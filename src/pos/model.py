"""The tagging Module."""
from enum import Enum
import logging
from typing import List, Mapping, Sequence, Tuple, Any, Dict, Optional, cast
import abc
import random

import torch
from torch import Tensor, stack, softmax, diagonal, cat, zeros_like, randn
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, AutoConfig

from . import core
from .core import Sentence, Sentences, VocabMap
from .data import (
    map_to_index,
    map_to_chars_batch,
    BATCH_KEYS,
    map_to_index_batch,
    get_initial_token_mask,
)


log = logging.getLogger(__name__)


class Modules(Enum):
    """To hold the module names."""

    Pretrained = "pretrained"
    Trained = "trained"
    MorphLex = "morphlex"
    CharactersToTokens = "chars"
    BiLSTM = "bilstm"
    BERT = "bert"
    Tagger = "tagger"
    Lemmatizer = "lemmatizer"


class BatchPostprocess(metaclass=abc.ABCMeta):
    """An interface to handle postprocessing for modules."""

    @abc.abstractmethod
    def postprocess(self, batch: Tensor, lengths: Sequence[int]) -> Sentences:
        """Postprocess the model output."""
        raise NotImplementedError


class BatchPreprocess(metaclass=abc.ABCMeta):
    """An interface to handle preprocessing for modules."""

    @abc.abstractmethod
    def preprocess(self, batch: Sequence[Sentence]) -> Tensor:
        """Preprocess the sentence batch."""
        raise NotImplementedError


class Embedding(BatchPreprocess, nn.Module, metaclass=abc.ABCMeta):
    """A module which accepts string inputs and embeds them to tensors."""

    def forward(self, batch: Sequence[Sentence], lengths: Sequence[int]) -> Tensor:
        """Run a generic forward pass for the Embeddings."""
        return self.embed(self.preprocess(batch), lengths).to(core.device)

    @abc.abstractmethod
    def embed(self, batch: Tensor, lengths: Sequence[int]) -> Tensor:
        """Apply the embedding."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension."""
        raise NotImplementedError


class ClassingWordEmbedding(Embedding):
    """Classic word embeddings."""

    def __init__(
        self, vocab_map: VocabMap, embedding_dim: int, padding_idx=0, dropout=0.0
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

    def preprocess(self, batch: Sequence[Sentence]) -> Tensor:
        """Preprocess the sentence batch."""
        return pad_sequence(
            [map_to_index(x, w2i=self.vocab_map.w2i) for x in batch],
            batch_first=True,
        )

    def embed(self, batch: Tensor, lengths: Sequence[int]) -> Tensor:
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
        vocab_map: VocabMap,
        embeddings: Tensor,
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


class CharacterAsWordEmbedding(Embedding):
    """A Character as Word Embedding."""

    def __init__(
        self,
        vocab_map: VocabMap,
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
        # The character BiLSTM
        self.char_bilstm = nn.LSTM(
            input_size=character_embedding_dim,
            hidden_size=char_lstm_dim,
            num_layers=char_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.emb_dropout = nn.Dropout(p=dropout)
        self.dropout = nn.Dropout(p=dropout)
        for name, param in self.char_bilstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.xavier_uniform_(param)
        self._output_dim = 2 * char_lstm_dim

    def preprocess(self, batch: Sequence[Sentence]) -> Tensor:
        """Preprocess the sentence batch."""
        return map_to_chars_batch(batch, self.vocab_map.w2i)

    def embed(self, batch: Tensor, lengths: Sequence[int]) -> Tensor:
        """Apply the embedding."""
        # (b * seq, chars)
        char_embs = self.emb_dropout(self.sparse_embedding(batch))
        # (b * seq, chars, f)
        self.char_bilstm.flatten_parameters()
        out, _ = self.char_bilstm(char_embs)
        return self.dropout(out)

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self._output_dim


class TransformerEmbedding(Embedding):
    """An embedding of a sentence after going through a Transformer."""

    def __init__(self, model_path: str, dropout=0.0):
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
        self.layer_weights = nn.parameter.Parameter(randn(self.num_layers))

    def preprocess(self, batch: Sequence[Sentence]) -> Dict[str, Tensor]:
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
                Tensor(
                    get_initial_token_mask(encoded["offset_mapping"][0].tolist())
                ).bool()
            )
        # Stack as batches
        try:
            return {
                key: stack(value).to(core.device) for key, value in preprocessed.items()
            }
        except RuntimeError as e:
            log.error(f"Unable to stack: {batch}")
            raise e

    def embed(self, batch: Dict[str, Tensor], lengths: Sequence[int]) -> Tensor:
        """Apply the embedding."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        # Tuple[(b, s, f), ...]
        output = outputs["hidden_states"][-self.num_layers :]
        weights = nn.functional.softmax(self.layer_weights)
        weighted_layers = output[0] * weights[0]
        for l in range(1, self.num_layers):
            weighted_layers += output[l] * weights[l]
        # Now we map the subtokens to tokens.
        tokens_emb = []
        for b in range(weighted_layers.shape[0]):
            initial_token_mask = batch["initial_token_masks"][b]
            output_sent = weighted_layers[b, :, :]
            tokens_emb.append(output_sent[initial_token_mask, :])
        padded = pad_sequence(tokens_emb, batch_first=True)
        return self.dropout(padded)

    @property
    def output_dim(self):
        """Return the output dimension."""
        return self.hidden_dim


class DotAttention(nn.Module):
    """Dot attention module as described in Luong et al. 2015."""

    def __init__(self):
        """Initialize it."""
        super().__init__()

    @staticmethod
    def score(hidden_decoder: Tensor, hidden_encoder: Tensor):
        """Compute the dot product as a score.

        Args:
            hidden_decoder: (b, f)
            hidden_encoder: (b, f)
        Returns:
            score: (b)
        """
        # both are (b, f)
        return diagonal(hidden_decoder.matmul(hidden_encoder.T))

    def forward(self, hidden_decoder: Tensor, hiddens_encoder: Tensor):
        """Forward pass of the model.

        Args:
            hidden_decoder: (b, f), b is the batch_size, f the features
            hiddens_encoder: (b, t, f) where t is timeseries to attend to.
        Returns:
            context: (b, f)
        """
        # (b, t)
        scores = stack(
            tuple(
                self.score(hidden_decoder, hiddens_encoder[:, i, :])
                for i in range(hiddens_encoder.shape[1])
            ),
            dim=1,
        )
        # The attention weights; (b, t)
        a = softmax(scores, dim=1)
        # (b, t, f)
        a_stack: Tensor = stack(tuple(a for _ in range(hidden_decoder.shape[1])), dim=2)
        weighted_hiddens = a_stack.mul(hiddens_encoder)
        # (b, f)
        context = weighted_hiddens.sum(dim=1)
        return context


class Decoder(BatchPostprocess, nn.Module, metaclass=abc.ABCMeta):
    """A module which accepts an sentence embedding and outputs another tensor."""

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def weight(self) -> int:
        """Return the decoder weight."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_targets(self, batch: Dict[BATCH_KEYS, Any]):
        """Add the decoder targets to the batch dictionary. SIDE-EFFECTS!."""

    @abc.abstractmethod
    def decode(
        self, encoded: Dict[Modules, Tensor], batch: Dict[BATCH_KEYS, Any]
    ) -> Tensor:
        """Run the decoder on the batch."""

    def forward(
        self, encoded: Dict[Modules, Tensor], batch: Dict[BATCH_KEYS, Any]
    ) -> Tensor:
        """Run a generic forward pass for the Embeddings."""
        self.add_targets(batch)
        return self.decode(encoded=encoded, batch=batch)


class CharacterDecoder(Decoder):
    """Cho et al., 2014 GRU RNN decoder which uses the context vector for each timestep. LSTM instead of GRU due to errors.

    Code adjusted from: https://github.com/bentrevett/pytorch-seq2seq/
    """

    def __init__(
        self,
        vocab_map: VocabMap,
        hidden_dim,
        emb_dim,
        char_attention=False,
        teacher_forcing=0.0,
        dropout=0.0,
        weight=1,
    ):
        """Initialize the model."""
        super().__init__()
        self.vocab_map = vocab_map
        self.teacher_forcing = (
            teacher_forcing  # if rand < teacher_forcing we will use teacher forcing.
        )
        self.hidden_dim = hidden_dim  # The internal dimension of the GRU model
        self._output_dim = len(
            vocab_map
        )  # The number of characters, these will be interpreted as logits.
        self._weight = weight
        self.char_attention = char_attention

        self.sparse_embedding = nn.Embedding(
            len(vocab_map), emb_dim, sparse=True
        )  # We map the input idx to vectors.
        # last character + sentence context + character attention
        rnn_in_dim = emb_dim + hidden_dim + (hidden_dim if self.char_attention else 0)
        self.rnn = nn.LSTM(
            rnn_in_dim,
            hidden_dim,
            batch_first=True,
        )

        # Map directly to characters
        self.fc_out = nn.Linear(hidden_dim, len(vocab_map))
        self.illegal_chars_output = {
            self.vocab_map.SOS_ID,
            self.vocab_map.PAD_ID,
            self.vocab_map.UNK_ID,
        }
        if self.char_attention:
            self.attention = DotAttention()
        self.dropout = nn.Dropout(dropout)  # Embedding dropout

    @property
    def output_dim(self) -> int:
        """Return the output dimension."""
        return self._output_dim

    @property
    def weight(self) -> int:
        """Return the decoder weight."""
        return self._weight

    def map_lemma_from_char_idx(self, char_idxs: List[int]) -> str:
        """Map a lemma from character indices."""
        chars = [
            self.vocab_map.i2w[char_idx]
            for char_idx in char_idxs
            if char_idx not in self.illegal_chars_output
        ]
        # If we find an EOS, we cut from there.
        if self.vocab_map.EOS in chars:
            eos_idx = chars.index(self.vocab_map.EOS)
            chars = chars[:eos_idx]
        return "".join(chars)

    def map_sentence_chars(
        self, sent: List[List[int]], sent_length: int
    ) -> Tuple[str, ...]:
        """Map a sentence characters from idx to strings and join to lemmas."""
        lemmas: List[str] = []
        for tok_num in range(sent_length):
            lemmas.append(self.map_lemma_from_char_idx(sent[tok_num]))
        return tuple(lemmas)

    def postprocess(self, batch: Tensor, lengths: Sequence[int]) -> Sentences:
        """Postprocess the model output."""
        # Get the character predictions
        char_preds = batch.argmax(dim=2)
        # Map to batch of sentences again.
        sent_char_preds = char_preds.view(size=(len(lengths), -1, char_preds.shape[-1]))
        as_list = sent_char_preds.tolist()

        sentence_lemmas = []
        for sent, sent_length in zip(as_list, lengths):
            sentence_lemmas.append(self.map_sentence_chars(sent, sent_length))
        return tuple(sentence_lemmas)

    def add_targets(self, batch: Dict[BATCH_KEYS, Any]):
        """Preprocess the sentence batch. HAS SIDE-EFFECTS!."""
        if BATCH_KEYS.LEMMAS in batch:
            batch[BATCH_KEYS.TARGET_LEMMAS] = map_to_chars_batch(
                batch[BATCH_KEYS.LEMMAS], self.vocab_map.w2i, add_sos=False
            )

    @staticmethod
    def _get_char_input_next_timestep(
        vocab_map: VocabMap,
        batch: Dict[BATCH_KEYS, Any],
        previous_predictions: Optional[Tensor] = None,
        teacher_forcing=0.5,
        training=True,
    ):
        """Get the next character timestep to feed the model."""
        if previous_predictions is None:  # First timestep
            # We assume that tokens and lemmas work equally well for the first timestep (PAD or SOS)
            return map_to_chars_batch(batch[BATCH_KEYS.TOKENS], vocab_map.w2i)[:, 0]
        previous_timestep = previous_predictions.shape[1] - 1  # -1 for 0 indexing
        # We have the targets, are training and teacher forcing is set
        if (
            BATCH_KEYS.TARGET_LEMMAS in batch
            and training
            and random.random() < teacher_forcing
        ):
            return batch[BATCH_KEYS.TARGET_LEMMAS][:, previous_timestep]
        # Otherwise, we will feed previous predictions.
        return previous_predictions[:, previous_timestep, :].argmax(dim=1)

    def decode(
        self, encoded: Dict[Modules, Tensor], batch: Dict[BATCH_KEYS, Any]
    ) -> Tensor:
        """Run the decoder on the batch."""
        context = torch.cat(
            tuple(emb for key, emb in encoded.items() if key in {Modules.BiLSTM}),
            dim=2,
        )
        # [batch_size * max_seq_len, emb_size] aka (b*s, f)
        context = context.reshape(shape=(-1, context.shape[2]))

        # [batch_size * max_seq_len, max_token_len, emb_size] aka (b, t, f)
        predictions: Optional[Tensor] = None

        # (1, b*s, f)
        hidden = context.unsqueeze(dim=0)
        # (1, b*s, f)
        cell = zeros_like(hidden)
        # We are training
        if BATCH_KEYS.TARGET_LEMMAS in batch:
            for _ in range(
                batch[BATCH_KEYS.TARGET_LEMMAS].shape[1]
            ):  # Iterate over characters
                # (b)
                next_char_input = self._get_char_input_next_timestep(
                    self.vocab_map,
                    batch,
                    predictions,
                    teacher_forcing=self.teacher_forcing,
                    training=self.training,
                ).to(core.device)
                # (b, f)
                emb_chars = self.dropout(self.sparse_embedding(next_char_input))
                rnn_in = torch.cat((emb_chars, context), dim=1)
                if self.char_attention:
                    char_attention = self.attention(
                        hidden.squeeze(),
                        encoded[Modules.CharactersToTokens],
                    )
                    rnn_in = cat((rnn_in, char_attention), dim=1)
                rnn_in = rnn_in.unsqueeze(1)  # Add the time-step
                output, (hidden, cell) = self.rnn(rnn_in, (hidden, cell))
                prediction = self.fc_out(output, dim=2)
                if predictions is None:
                    predictions = prediction
                else:
                    predictions = torch.cat((predictions, prediction), dim=1)

        # We decode
        # TODO: support decoding until EOS/PAD for all
        else:
            pass
        predictions = cast(Tensor, predictions)
        return predictions


class Tagger(Decoder):
    """A tagger; accept some tensor input and return logits over classes."""

    def __init__(self, vocab_map: VocabMap, input_dim, weight=1):
        """Initialize."""
        super().__init__()
        self.vocab_map = vocab_map
        output_dim = len(vocab_map)
        self.tagger = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.tagger.weight)
        self._output_dim = output_dim
        self._weight = weight

    @property
    def output_dim(self) -> int:
        """Return the output dimension."""
        return self._output_dim

    @property
    def weight(self) -> int:
        """Return the decoder weight."""
        return self._weight

    def decode(
        self, encoded: Dict[Modules, Tensor], batch: Dict[BATCH_KEYS, Any]
    ) -> Tensor:
        """Run the decoder on the batch."""
        context = torch.cat(
            tuple(emb for key, emb in encoded.items() if key in {Modules.BiLSTM}),
            dim=2,
        )
        return self.tagger(context)

    def add_targets(self, batch: Dict[BATCH_KEYS, Any]):
        """Add the decoder targets to the batch dictionary. SIDE-EFFECTS!."""
        if BATCH_KEYS.FULL_TAGS in batch:
            batch[BATCH_KEYS.TARGET_FULL_TAGS] = map_to_index_batch(
                batch[BATCH_KEYS.FULL_TAGS], self.vocab_map.w2i
            )

    def postprocess(self, batch: Tensor, lengths: Sequence[int]) -> Sentences:
        """Postprocess the model output."""
        idxs = batch.argmax(dim=2).tolist()

        tags = [
            tuple(
                self.vocab_map.i2w[tag_idx]
                for token_count, tag_idx in enumerate(sent)
                # All sentences are padded (at the right end) to be of equal length.
                # We do not want to return tags for the paddings.
                # We check the information about lengths and paddings.
                if token_count < lengths[sent_idx]
            )
            for sent_idx, sent in enumerate(idxs)
        ]
        return tuple(tags)


class Encoder(nn.Module):
    """The Pytorch module implementing the encoder."""

    def __init__(
        self,
        embeddings: Dict[Modules, Embedding],
        main_lstm_dim=64,  # The main LSTM dim will output with this dim
        main_lstm_layers=1,  # The main LSTM layers
        lstm_dropouts=0.0,
        input_dropouts=0.0,
        residual=True,
    ):
        """Initialize the module given the parameters."""
        super().__init__()
        self.residual = residual
        self.embeddings = nn.ModuleDict(
            {key.value: emb for key, emb in embeddings.items()}
        )

        bilstm_in_dim = sum(emb.output_dim for emb in self.embeddings.values())
        self.linear = nn.Linear(bilstm_in_dim, main_lstm_dim)  # type: ignore

        # BiLSTM over all inputs
        self.bilstm = nn.LSTM(
            input_size=main_lstm_dim,
            hidden_size=main_lstm_dim,
            num_layers=main_lstm_layers,
            dropout=lstm_dropouts,
            batch_first=True,
            bidirectional=True,
        )
        for name, param in self.bilstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError("Unknown parameter in lstm={name}")
        self.main_bilstm_out_dropout = nn.Dropout(p=input_dropouts)
        self.output_dim = main_lstm_dim * 2

    def forward(
        self, batch: Sequence[Sentence], lengths: Sequence[int]
    ) -> Dict[Modules, Tensor]:
        """Run a forward pass through the module. Input should be tensors."""
        # input is (batch_size=num_sentence, max_seq_len_in_batch=max(len(sentences)), max_word_len_in_batch + 1 + 1)
        # Embeddings
        embedded = {key: emb(batch, lengths) for key, emb in self.embeddings.items()}
        results = {Modules(key): emb for key, emb in embedded.items()}

        to_bilstm = {key: embedded[key] for key in self.embeddings.keys()}
        if Modules.CharactersToTokens.value in to_bilstm:
            last_timestep = to_bilstm[Modules.CharactersToTokens.value][
                :, -1, :
            ]  # Only the last timestep
            to_bilstm[Modules.CharactersToTokens.value] = last_timestep.reshape(
                len(lengths), -1, last_timestep.shape[-1]
            )
        embs_to_linear = torch.cat(list(to_bilstm.values()), dim=2)
        embs_to_bilstm = self.linear(embs_to_linear)

        # Pack the paddings
        packed = pack_padded_sequence(
            embs_to_bilstm,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        # Make sure that the parameters are contiguous.
        self.bilstm.flatten_parameters()
        # Ignore the hidden outputs
        packed_out, _ = self.bilstm(packed)
        # Unpack and ignore the lengths
        bilstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        bilstm_out = self.main_bilstm_out_dropout(bilstm_out)
        # Use residual connections
        if self.residual:
            bilstm_out = bilstm_out + cat((embs_to_bilstm, embs_to_bilstm), dim=2)
        results[Modules.BiLSTM] = bilstm_out

        return results


class ABLTagger(nn.Module):
    """The ABLTagger, consists of an Encoder(multipart) and a Tagger."""

    def __init__(self, encoder: Encoder, decoders: Dict[Modules, Decoder]):
        """Initialize the tagger."""
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict({key.value: emb for key, emb in decoders.items()})
        self.decoders = cast(Mapping[str, Decoder], self.decoders)

    def forward(self, batch: Dict[BATCH_KEYS, Any]) -> Dict[Modules, Tensor]:
        """Forward pass."""
        encoded: Dict[Modules, Tensor] = self.encoder(
            batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS]
        )
        return {
            Modules(key): decoder(encoded, batch)
            for key, decoder in self.decoders.items()
        }
