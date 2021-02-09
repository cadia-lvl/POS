"""Implementation of several embeddings."""
from typing import Dict, Sequence

from torch import nn
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, AutoConfig

from pos import core
from pos.data import map_to_index, get_initial_token_mask
from pos.data.batch import map_to_chars_batch
from pos.data.constants import PAD
from pos.morphlex import MorphologicalLexicon
from . import abltagger


class ClassingWordEmbedding(abltagger.Embedding):
    """Classic word embeddings."""

    def __init__(
        self, vocab_map: core.VocabMap, embedding_dim: int, padding_idx=0, dropout=0.0
    ):
        """Create one."""
        super().__init__()
        self.vocab_map = vocab_map
        self.sparse_embedding = nn.Embedding(
            len(vocab_map), embedding_dim, padding_idx=padding_idx, sparse=True
        )
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


class CharacterEmbedding(abltagger.Embedding):
    """Character embedding."""

    def __init__(
        self, vocab_map: core.VocabMap, char_dim: int, padding_idx=0, dropout=0.0
    ):
        """Inititalize character embeddings."""
        super().__init__()
        self.sparse_embedding = nn.Embedding(
            len(vocab_map),
            char_dim,
            padding_idx=padding_idx,
            sparse=True,
        )
        self.dropout = nn.Dropout(p=dropout)

    def preprocess(self, batch: Sequence[core.Sentence]) -> torch.Tensor:
        """Preprocess the sentence batch."""
        return map_to_chars_batch(batch, self.vocab_map.w2i)

    def embed(self, batch: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
        """Apply the embedding."""
        return self.dropout(self.sparse_embedding(batch))

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self.sparse_embedding.weight.data.shape[1]


class MorphLexEmbedding(abltagger.Embedding):
    """Morphological lexicon embedding."""

    def __init__(
        self,
        morphlex: MorphologicalLexicon,
        vocab_map: core.VocabMap,
        tag_embedding_dim: int,
        padding_idx=0,
        dropout=0.0,
    ):
        """Inititialize."""
        super().__init__()
        # TODO: add lemmas
        self.morphlex = morphlex
        self.sparse_embedding = nn.Embedding(
            len(vocab_map), tag_embedding_dim, padding_idx=padding_idx, sparse=True
        )
        self.vocab_map = vocab_map
        self.dropout = nn.Dropout(p=dropout)

    def preprocess(self, batch: Sequence[core.Sentence]) -> torch.Tensor:
        """Preprocess the sentence batch."""
        # (b, tokens, tags)
        batch_tags = []
        for sent in batch:
            sent_tags = []
            for token in sent:
                results = self.morphlex.lookup_word(token)
                # (tags) - put a PAD, so the model can easily choose none.
                # TODO: add lemmas
                token_tags = map_to_index(
                    (PAD,) + tuple(tag for tag, _ in results), w2i=self.vocab_map.w2i
                )
                sent_tags.append(token_tags)
            batch_tags.append(sent_tags)
        max_num_tokens = max(len(sent) for sent in batch_tags)
        max_num_tags = max(tags.shape[0] for sent in batch_tags for tags in sent)
        # (b, tokens, tags)
        padded = torch.zeros(
            size=(len(batch), max_num_tokens, max_num_tags), device=core.device
        ).long()
        for s_idx, sent in enumerate(batch_tags):
            for t_idx, token in enumerate(sent):
                padded[s_idx : s_idx + 1, t_idx : t_idx + 1, : token.shape[0]] = token
        return padded

    def embed(self, batch: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
        """Apply the embedding."""
        return self.dropout(self.sparse_embedding(batch))

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self.sparse_embedding.weight.data.shape[1]


class CharacterAsWordEmbedding(abltagger.Embedding):
    """A Character as Word Embedding."""

    def __init__(
        self,
        vocab_map: core.VocabMap,
        character_embedding_dim=20,
        char_lstm_dim=64,
        char_lstm_layers=1,
        padding_idx=0,
        dropout=0.0,
    ):
        """Create one."""
        super().__init__()
        self.vocab_map = vocab_map
        self.sparse_embedding = nn.Embedding(
            len(vocab_map),
            character_embedding_dim,
            padding_idx=padding_idx,
            sparse=True,
        )
        nn.init.xavier_uniform_(self.sparse_embedding.weight[1:, :])
        # The character RNN
        self.rnn = nn.GRU(
            input_size=character_embedding_dim,
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
        return map_to_chars_batch(batch, self.vocab_map.w2i)

    def embed(self, batch: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
        """Apply the embedding."""
        # (b * seq, chars)
        char_embs = self.emb_dropout(self.sparse_embedding(batch))
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

    def __init__(self, model_path: str, layers="weights", dropout=0.0):
        """Initialize it be reading the config, model and tokenizer."""
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_path, config=self.config)
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_path
        )
        # ELECTRA property
        self.num_layers = self.config.num_hidden_layers
        self.hidden_dim = self.config.hidden_size
        self.max_length = self.tokenizer.max_len_single_sentence
        self.dropout = nn.Dropout(p=dropout)
        self.layers = layers
        if self.layers == "weights":
            self.layer_weights = nn.parameter.Parameter(torch.randn(self.num_layers))

    def preprocess(self, batch: Sequence[core.Sentence]) -> Dict[str, torch.Tensor]:
        """Preprocess the sentence batch."""
        preprocessed = {
            "input_ids": [],
            "attention_mask": [],
            "initial_token_masks": [],
        }
        for sentence in batch:
            encoded = self.tokenizer.encode_plus(
                text=sentence,
                is_split_into_words=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            preprocessed["input_ids"].append(encoded["input_ids"][0])
            preprocessed["attention_mask"].append(encoded["attention_mask"][0])
            preprocessed["initial_token_masks"].append(
                torch.Tensor(
                    get_initial_token_mask(encoded["offset_mapping"][0].tolist())
                ).bool()
            )
        return {
            key: torch.stack(value).to(core.device)
            for key, value in preprocessed.items()
        }

    def embed(
        self, batch: Dict[str, torch.Tensor], lengths: Sequence[int]
    ) -> torch.Tensor:
        """Apply the embedding."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        # Tuple[(b, s, f), ...]
        output = outputs["hidden_states"][-self.num_layers :]
        emb = outputs["last_hidden_state"]
        if self.layers == "weights":
            weights = nn.functional.softmax(self.layer_weights, dim=0)
            weighted_layers = output[0] * weights[0]
            for l in range(1, self.num_layers):
                weighted_layers += output[l] * weights[l]
            emb = weighted_layers
        # Now we map the subtokens to tokens.
        tokens_emb = []
        for b in range(emb.shape[0]):
            initial_token_mask = batch["initial_token_masks"][b]
            output_sent = emb[b, :, :]
            tokens_emb.append(output_sent[initial_token_mask, :])
        padded = torch.nn.utils.rnn.pad_sequence(tokens_emb, batch_first=True)
        return self.dropout(padded)

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self.hidden_dim
