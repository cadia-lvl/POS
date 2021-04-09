"""Implementation of several embeddings."""
from typing import Any, Dict, Sequence

import torch
from pos import core
from pos.data import get_initial_token_mask, map_to_index
from pos.data.batch import map_to_chars_batch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from . import abltagger


class ClassingWordEmbedding(abltagger.Embedding):
    """Classic word embeddings."""

    def __init__(self, vocab_map: core.VocabMap, embedding_dim: int, padding_idx=0, dropout=0.0):
        """Create one."""
        super().__init__()
        self.vocab_map = vocab_map
        self.sparse_embedding = nn.Embedding(len(vocab_map), embedding_dim, padding_idx=padding_idx, sparse=True)
        # Skip the first index, should be zero
        nn.init.xavier_uniform_(self.sparse_embedding.weight[1:, :])
        self.dropout = nn.Dropout(p=dropout)

    def preprocess(self, batch: Sequence[core.Sentence]) -> torch.Tensor:
        """Preprocess the sentence batch."""
        return torch.nn.utils.rnn.pad_sequence(
            [map_to_index(x, w2i=self.vocab_map.w2i) for x in batch],
            batch_first=True,
        )

    def embed(self, batch: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
        """Apply the embedding."""
        return self.dropout(self.sparse_embedding(batch))

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self.sparse_embedding.weight.data.shape[1]


class CharacterEmbedding(ClassingWordEmbedding):
    """Character embedding. Has a distinct preprocessing step from classic."""

    def preprocess(self, batch: Sequence[core.Sentence]) -> torch.Tensor:
        """Preprocess the sentence batch."""
        return map_to_chars_batch(batch, self.vocab_map.w2i)


class PretrainedEmbedding(ClassingWordEmbedding):
    """The Morphological Lexicion embeddings."""

    def __init__(
        self,
        vocab_map: core.VocabMap,
        embeddings: torch.Tensor,
        freeze=False,
        padding_idx=0,
        dropout=0.0,
    ):
        """Create one."""
        super().__init__(
            vocab_map=vocab_map,
            embedding_dim=1,
            padding_idx=padding_idx,
            dropout=dropout,
        )  # we overwrite the embedding
        self.sparse_embedding = nn.Embedding.from_pretrained(
            embeddings, freeze=freeze, padding_idx=padding_idx, sparse=True
        )


class CharacterAsWordEmbedding(abltagger.Embedding):
    """A Character as Word Embedding."""

    def __init__(
        self,
        character_embedding: CharacterEmbedding,
        char_lstm_dim=64,
        char_lstm_layers=1,
        dropout=0.0,
    ):
        """Create one."""
        super().__init__()
        self.character_embedding = character_embedding
        # The character RNN
        self.rnn = nn.GRU(
            input_size=self.character_embedding.output_dim,
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

    def preprocess(self, batch: Sequence[core.Sentence]) -> torch.Tensor:
        """Preprocess the sentence batch."""
        return batch  # type: ignore

    def embed(self, batch: torch.Tensor, lengths: Sequence[int]) -> Any:
        """Apply the embedding."""
        # (b * seq, chars)
        char_embs = self.character_embedding(batch, lengths)
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
        return (
            self.dropout(out),
            self.dropout(hidden.permute(1, 0, 2).reshape(out.shape[0], -1)),
        )

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self._output_dim


class TransformerEmbedding(abltagger.Embedding):
    """An embedding of a sentence after going through a Transformer."""

    def __init__(self, model_path: str, dropout=0.0):
        """Initialize it be reading the config, model and tokenizer."""
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_path, config=self.config)
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_path)  # type: ignore
        # ELECTRA property
        self.num_layers = self.config.num_hidden_layers  # type: ignore
        self.hidden_dim = self.config.hidden_size  # type: ignore
        self.max_length = self.tokenizer.max_len_single_sentence  # type: ignore
        self.dropout = nn.Dropout(p=dropout)

    def preprocess(self, batch: Sequence[core.Sentence]) -> Dict[str, torch.Tensor]:
        """Preprocess the sentence batch."""
        preprocessed = {
            "input_ids": [],
            "attention_mask": [],
            "initial_token_masks": [],
        }
        for sentence in batch:
            encoded = self.tokenizer.encode_plus(
                text=list(sentence),
                is_split_into_words=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_prefix_space=True,
            )
            preprocessed["input_ids"].append(encoded["input_ids"][0])
            preprocessed["attention_mask"].append(encoded["attention_mask"][0])
            preprocessed["initial_token_masks"].append(
                torch.Tensor(get_initial_token_mask(encoded["offset_mapping"][0].tolist())).bool()
            )
        return {key: torch.stack(value).to(core.device) for key, value in preprocessed.items()}

    def embed(self, batch: Dict[str, torch.Tensor], lengths: Sequence[int]) -> Any:
        """Apply the embedding."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        # Tuple[(b, s, f), ...]
        outputs["initial_token_masks"] = batch["initial_token_masks"]
        return outputs

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self.hidden_dim
