"""Implementation of several embeddings."""
from logging import getLogger
from typing import Any, Dict, List

import torch
from pos import core
from pos.constants import BATCH_KEYS
from pos.data import get_initial_token_mask, map_to_index
from pos.data.batch import map_to_chars_batch
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.models.electra import ElectraConfig, ElectraModel

from . import interface

log = getLogger(__name__)


class ClassicWordEmbedding(interface.Encoder):
    """Classic word embeddings."""

    def __init__(self, key: str, vocab_map: core.VocabMap, embedding_dim: int, padding_idx=0, dropout=0.0):
        """Create one."""
        super().__init__(key)
        self.vocab_map = vocab_map
        self.embedding = nn.Embedding(len(vocab_map), embedding_dim, padding_idx=padding_idx)
        # Skip the first index, should be zero
        nn.init.xavier_uniform_(self.embedding.weight[1:, :])
        self.dropout = nn.Dropout(p=dropout)

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the sentence batch."""
        batch[BATCH_KEYS.TOKEN_IDS] = nn.utils.rnn.pad_sequence(
            [map_to_index(x, w2i=self.vocab_map.w2i) for x in batch[BATCH_KEYS.TOKENS]],
            batch_first=True,
        )
        return batch

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the embedding."""
        batch[self.key] = self.dropout(self.embedding(batch[BATCH_KEYS.TOKEN_IDS]))
        return batch

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self.embedding.weight.data.shape[1]


class CharacterEmbedding(ClassicWordEmbedding):
    """Character embedding. Has a distinct preprocessing step from classic."""

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the sentence batch."""
        batch[BATCH_KEYS.CHAR_IDS] = map_to_chars_batch(batch[BATCH_KEYS.TOKENS], self.vocab_map.w2i)
        return batch

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the embedding."""
        batch[self.key] = self.dropout(self.embedding(batch[BATCH_KEYS.CHAR_IDS]))
        return batch


class PretrainedEmbedding(ClassicWordEmbedding):
    """The Morphological Lexicion embeddings."""

    def __init__(
        self,
        key: str,
        vocab_map: core.VocabMap,
        embeddings: torch.Tensor,
        freeze=False,
        padding_idx=0,
        dropout=0.0,
    ):
        """Create one."""
        super().__init__(
            key=key,
            vocab_map=vocab_map,
            embedding_dim=1,
            padding_idx=padding_idx,
            dropout=dropout,
        )  # we overwrite the embedding
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze, padding_idx=padding_idx)

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the sentence batch."""
        batch[BATCH_KEYS.PRETRAINED_TOKEN_IDS] = nn.utils.rnn.pad_sequence(
            [map_to_index(x, w2i=self.vocab_map.w2i) for x in batch[BATCH_KEYS.TOKENS]],
            batch_first=True,
        )
        return batch

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the embedding."""
        batch[self.key] = self.dropout(self.embedding(batch[BATCH_KEYS.PRETRAINED_TOKEN_IDS]))
        return batch


class CharacterAsWordEmbedding(interface.Encoder):
    """A Character as Word Embedding."""

    def __init__(
        self,
        key: str,
        character_embedding: CharacterEmbedding,
        char_lstm_dim=64,
        char_lstm_layers=1,
        dropout=0.0,
    ):
        """Create one."""
        super().__init__(key)
        self.character_embedding_key = character_embedding.key
        # The character RNN
        self.rnn = nn.GRU(
            input_size=character_embedding.output_dim,
            hidden_size=char_lstm_dim,
            num_layers=char_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.emb_dropout = nn.Dropout(p=dropout)
        self.dropout = nn.Dropout(p=dropout)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.xavier_uniform_(param)
        self._output_dim = 2 * char_lstm_dim

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the sentence batch."""
        return batch

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the embedding."""
        # (b * seq, chars)
        char_embs = batch[self.character_embedding_key]
        # (b * seq, chars, f)
        self.rnn.flatten_parameters()
        out, hidden = self.rnn(char_embs)
        if type(self.rnn) == nn.LSTM:
            # If we are running an LSTM, just take the hidden, not the cell.
            hidden = hidden[0]
        # Warning! We return a tuple here, the last hidden state and the sequence.
        # Output documentation: (seq_len, batch, num_directions * hidden_size)
        # Batch first and our names: (b * seq, chars, num_directions * hidden_size)
        # Hidden documentation (GRU): (num_layers * num_directions, batch, hidden_size)
        # Batch is NOT placed first in the hidden.
        # We map it to (b * seq, hidden_size * num_layers * num_directions)
        batch[self.key] = (
            self.dropout(out),
            self.dropout(hidden.permute(1, 0, 2).reshape(out.shape[0], -1)),
        )
        return batch

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self._output_dim


class TransformerEmbedding(interface.Encoder):
    """An embedding of a sentence after going through a Transformer."""

    def __init__(self, key: str, path: str, dropout=0.0):
        """Initialize it be reading the config, model and tokenizer."""
        super().__init__(key)
        self.config = ElectraConfig.from_pretrained(path, output_hidden_states=True)
        self.model = ElectraModel(self.config)
        self.add_prefix_space = False
        if "roberta" in str(self.config.__class__).lower():  # type: ignore
            log.debug("Using prefix space")
            self.add_prefix_space = True
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(path, add_prefix_space=self.add_prefix_space)  # type: ignore
        # ELECTRA property
        self.num_layers = self.config.num_hidden_layers  # type: ignore
        self.hidden_dim = self.config.hidden_size  # type: ignore
        self.max_length = min(self.config.max_position_embeddings, self.tokenizer.model_max_length, 512)  # type: ignore
        self.dropout = nn.Dropout(p=dropout)

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the sentence batch."""
        preprocessed: Dict[str, List[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "initial_token_masks": [],
        }
        for sentence in batch[BATCH_KEYS.TOKENS]:
            encoded = self.tokenizer.encode_plus(  # type: ignore
                text=" ".join(sentence),
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            assert encoded["input_ids"].shape[1] == self.max_length, "The should be exactly the max_length defined."
            preprocessed["input_ids"].append(encoded["input_ids"][0])
            preprocessed["attention_mask"].append(encoded["attention_mask"][0])
            preprocessed["initial_token_masks"].append(
                torch.Tensor(get_initial_token_mask(encoded["offset_mapping"][0].tolist())).bool()
            )
        batch[BATCH_KEYS.SUBWORDS] = {key: torch.stack(value).to(core.device) for key, value in preprocessed.items()}
        return batch

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the embedding."""
        outputs = self.model(
            input_ids=batch[BATCH_KEYS.SUBWORDS]["input_ids"],
            attention_mask=batch[BATCH_KEYS.SUBWORDS]["attention_mask"],
            return_dict=True,
        )
        # Tuple[(b, s, f), ...]
        outputs["initial_token_masks"] = batch[BATCH_KEYS.SUBWORDS]["initial_token_masks"]
        batch[self.key] = get_emb_by_initial_token_masks(outputs)
        return batch

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self.hidden_dim


def get_emb_by_initial_token_masks(outputs: Dict[str, Any]) -> torch.Tensor:
    emb = outputs["hidden_states"][-1]  # Only last layer
    tokens_emb = []
    for b in range(emb.shape[0]):
        initial_token_mask = outputs["initial_token_masks"][b]
        output_sent = emb[b, :, :]
        tokens_emb.append(output_sent[initial_token_mask, :])
    padded = nn.utils.rnn.pad_sequence(tokens_emb, batch_first=True)
    return padded
