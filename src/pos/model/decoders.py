"""Implmementation of some decoders."""

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

import torch.nn as nn
import torch
from torch import Tensor, softmax
import numpy as np
from torch.nn.modules.sparse import Embedding
from . import abltagger
from pos.core import Sentences, VocabMap
from pos.data import (
    map_to_chars_batch,
    BATCH_KEYS,
    map_to_index_batch,
    SOS,
    PAD,
)
from .embeddings import CharacterEmbedding


class MultiplicativeAttention(nn.Module):
    """Multiplicative attention module as described in Luong et al. 2015."""

    def __init__(self, encoder_dim: int, decoder_dim: int):
        """Initialize it."""
        super().__init__()
        self.decoder_dim = decoder_dim
        self.W = nn.Linear(decoder_dim, encoder_dim, bias=False)

    def forward(self, query: Tensor, values: Tensor):
        """Forward pass of the model.

        Args:
            query: (b, f_d), b is the batch_size, f the features
            values: (b, t, f_e) where t is timeseries to attend to.

        Returns:
            context: (b, f)
        """
        # (b, t) = (b, f_d) @ (f_d, f_e) bmm (b, t, f_e).permute(0,2,1)
        # (b, 1, f_e) bmm (b, f_e, t)
        # (b, 1, t).squeeze()
        weights = self.W(query).unsqueeze(dim=1).bmm(values.permute(0, 2, 1)).squeeze()
        # (b, t)
        weights = weights / np.sqrt(self.decoder_dim)
        weights = softmax(weights, dim=1)
        # (b, 1, t) bmm (b, t, f_e).squeeze()
        return weights.unsqueeze(dim=1).bmm(values).squeeze()


class CharacterDecoder(abltagger.Decoder):
    """The lemmatizer."""

    MAX_SEQUENCE_ADDITIONAL = 20

    def __init__(
        self,
        vocab_map: VocabMap,
        hidden_dim,
        context_dim,
        character_embedding: CharacterEmbedding,
        num_layers=1,
        char_rnn_input_dim=0,
        attention_dim=0,
        char_attention=False,
        dropout=0.0,
        weight=1,
    ):
        """Initialize the model."""
        super().__init__()
        self.vocab_map = vocab_map
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim  # The internal dimension of the GRU model
        self.attention_dim = attention_dim
        self._output_dim = len(vocab_map)  # The number of characters, these will be interpreted as logits.
        self._weight = weight
        self.char_attention = char_attention
        self.char_rnn_input_dim = char_rnn_input_dim

        self.character_embedding = character_embedding
        # last character + sentence context + character attention
        rnn_in_dim = (
            self.character_embedding.output_dim
            + self.context_dim
            + (self.attention_dim if self.char_attention else 0)
            + self.char_rnn_input_dim
        )
        self.rnn = nn.LSTM(
            input_size=rnn_in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Map directly to characters
        self.output_embedding = Embedding.from_pretrained(
            embeddings=self.character_embedding.weight.data.transpose(0, 1), padding_idx=0
        )
        if self.char_attention:
            self.attention = MultiplicativeAttention(
                encoder_dim=self.attention_dim,
                decoder_dim=self.hidden_dim * self.num_layers,
            )
        self.dropout = nn.Dropout(dropout)  # Embedding dropout

    @property
    def illegal_chars_output(self) -> Set[int]:
        return {self.sos_idx, self.vocab_map.UNK_ID, self.pad_idx}

    @property
    def pad_idx(self) -> int:
        return self.vocab_map.PAD_ID

    @property
    def sos_idx(self) -> int:
        return self.vocab_map.SOS_ID

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
        chars = [self.vocab_map.i2w[char_idx] for char_idx in char_idxs if char_idx not in self.illegal_chars_output]
        # If we find an EOS, we cut from there.
        if self.vocab_map.EOS in chars:
            eos_idx = chars.index(self.vocab_map.EOS)
            chars = chars[:eos_idx]
        return "".join(chars)

    def map_sentence_chars(self, sent: List[List[int]], sent_length: int) -> Tuple[str, ...]:
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
    def make_sequence(device, token_idx: int, batch_size) -> Tensor:
        """Return the SOS sequence."""
        return Tensor([token_idx] * batch_size).long().to(device)

    def _get_char_input_next_timestep(
        self,
        timestep: int,
        previous_predictions: Tensor,
        max_timestep: int,
    ) -> Optional[Tensor]:
        """Get the next character (as an index for embedding) timestep to feed the model."""
        if timestep == max_timestep:
            return None
        sos_sequence = self.make_sequence(previous_predictions.device, self.sos_idx, previous_predictions.shape[0])
        if timestep == 0:  # First timestep
            return sos_sequence
        pad_sequence = self.make_sequence(previous_predictions.device, self.pad_idx, previous_predictions.shape[0])
        # Otherwise, we will feed previous predictions.
        last_timestep_idxs = previous_predictions[:, timestep - 1, :].argmax(dim=1)
        equal_sos = last_timestep_idxs == sos_sequence
        equal_pad = last_timestep_idxs == pad_sequence
        if (equal_pad + equal_sos).all():
            return None
        else:
            return last_timestep_idxs

    def decode(
        self, encoded: Dict[abltagger.Modules, Tensor], batch: Dict[BATCH_KEYS, Any], tags: Optional[Tensor]
    ) -> Tensor:
        """Run the decoder on the batch."""
        tag_embeddings = tags.clone().detach()
        b, s, f = tag_embeddings.shape
        # 1 for EOS
        c = (
            batch[BATCH_KEYS.TARGET_LEMMAS].shape[1]
            if BATCH_KEYS.TARGET_LEMMAS in batch
            else max(batch[BATCH_KEYS.TOKEN_CHARS_LENS]) + self.MAX_SEQUENCE_ADDITIONAL
        )
        # (b*s, f) = (num_tokens, features)
        tag_embeddings = tag_embeddings.reshape(b * s, f)
        # (layers, b*s, f)
        hidden = torch.zeros(size=(self.num_layers, b * s, self.hidden_dim), device=tag_embeddings.device)
        cell = torch.zeros(size=(self.num_layers, b * s, self.hidden_dim), device=tag_embeddings.device)
        # (b*s, t, f_out)
        predictions = torch.zeros(size=(b * s, c, self.output_dim), device=tag_embeddings.device)

        char_idx = 0
        # (b*s)
        next_char_input = self._get_char_input_next_timestep(
            timestep=char_idx,
            previous_predictions=predictions,
            max_timestep=c,
        )
        while next_char_input is not None:
            # (b*s, f)
            emb_chars = self.dropout(self.character_embedding(next_char_input))
            rnn_in = torch.cat((emb_chars, tag_embeddings), dim=1)
            if self.char_rnn_input_dim:
                # (b*s, f)
                rnn_in = torch.cat((rnn_in, encoded[abltagger.Modules.CharactersToTokens][1]), dim=1)
            if self.char_attention:
                # char rnn = (b*s, c, f)
                char_attention = self.attention(
                    hidden.view(hidden.shape[1], -1), encoded[abltagger.Modules.CharactersToTokens][0]
                )
                rnn_in = torch.cat((rnn_in, char_attention), dim=1)
            # (b*s, 1, f), a single timestep
            rnn_in = rnn_in.unsqueeze(1)
            output, (hidden, cell) = self.rnn(rnn_in, (hidden, cell))
            predictions[:, char_idx : char_idx + 1, :] = self.output_embedding(output)
            # For next iteration
            char_idx += 1
            next_char_input = self._get_char_input_next_timestep(
                timestep=char_idx,
                previous_predictions=predictions,
                max_timestep=c,
            )
        return predictions


class Tagger(abltagger.Decoder):
    """A tagger; accept some tensor input and return logits over classes."""

    def __init__(
        self,
        vocab_map: VocabMap,
        input_dim,
        weight=1,
        embedding=abltagger.Modules.BiLSTM,
    ):
        """Initialize."""
        super().__init__()
        self.vocab_map = vocab_map
        self._output_dim = len(vocab_map)
        self._weight = weight
        self.embedding = embedding
        # Classification head
        self.dense = nn.Linear(input_dim, input_dim)
        self.activation_fn = nn.ReLU()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.out_proj = nn.Linear(input_dim, self.output_dim)

    @property
    def output_dim(self) -> int:
        """Return the output dimension."""
        return self._output_dim

    @property
    def weight(self) -> int:
        """Return the decoder weight."""
        return self._weight

    def decode(
        self, encoded: Dict[abltagger.Modules, Any], batch: Dict[BATCH_KEYS, Any], decoders: Optional[Tensor]
    ) -> Tensor:
        """Run the decoder on the batch."""
        # Now we map the subtokens to tokens.
        if self.embedding == abltagger.Modules.BERT:
            x = abltagger.get_emb_by_initial_token_masks(encoded)
        elif self.embedding == abltagger.Modules.BiLSTM:
            x = encoded[abltagger.Modules.BiLSTM]
        else:
            raise NotImplementedError(f"{self.embedding} is not implemented in Tagger")
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)
        return self.out_proj(x)

    def add_targets(self, batch: Dict[BATCH_KEYS, Any]):
        """Add the decoder targets to the batch dictionary. SIDE-EFFECTS!."""
        if BATCH_KEYS.FULL_TAGS in batch:
            batch[BATCH_KEYS.TARGET_FULL_TAGS] = map_to_index_batch(batch[BATCH_KEYS.FULL_TAGS], self.vocab_map.w2i)

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
