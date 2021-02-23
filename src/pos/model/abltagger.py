"""The main classes used for creating models."""
from enum import Enum
import abc
from typing import Any, Dict, Mapping, Sequence, cast

import torch
from torch import nn

from pos.core import Sentence, Sentences
from pos.data import BATCH_KEYS


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
    def postprocess(self, batch: torch.Tensor, lengths: Sequence[int]) -> Sentences:
        """Postprocess the model output."""
        raise NotImplementedError


class BatchPreprocess(metaclass=abc.ABCMeta):
    """An interface to handle preprocessing for modules."""

    @abc.abstractmethod
    def preprocess(self, batch: Sequence[Sentence]) -> torch.Tensor:
        """Preprocess the sentence batch."""
        raise NotImplementedError


class Embedding(BatchPreprocess, nn.Module, metaclass=abc.ABCMeta):
    """A module which accepts string inputs and embeds them to tensors."""

    def forward(self, batch: Sequence[Sentence], lengths: Sequence[int]) -> torch.Tensor:
        """Run a generic forward pass for the Embeddings."""
        return self.embed(self.preprocess(batch), lengths)

    @abc.abstractmethod
    def embed(self, batch: torch.Tensor, lengths: Sequence[int]) -> Any:
        """Apply the embedding."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension."""
        raise NotImplementedError


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
    def decode(self, encoded: Dict[Modules, Any], batch: Dict[BATCH_KEYS, Any]) -> torch.Tensor:
        """Run the decoder on the batch."""

    def forward(self, encoded: Dict[Modules, torch.Tensor], batch: Dict[BATCH_KEYS, Any]) -> torch.Tensor:
        """Run a generic forward pass for the Embeddings."""
        self.add_targets(batch)
        return self.decode(encoded=encoded, batch=batch)


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
        self.embeddings = nn.ModuleDict({key.value: emb for key, emb in embeddings.items()})

        bilstm_in_dim = sum(emb.output_dim for emb in self.embeddings.values())
        if self.residual:
            self.linear = nn.Linear(bilstm_in_dim, main_lstm_dim)
            bilstm_in_dim = main_lstm_dim

        # BiLSTM over all inputs
        self.bilstm = nn.LSTM(
            input_size=bilstm_in_dim,
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

    def forward(self, batch: Sequence[Sentence], lengths: Sequence[int]) -> Dict[Modules, torch.Tensor]:
        """Run a forward pass through the module. Input should be tensors."""
        # input is (batch_size=num_sentence, max_seq_len_in_batch=max(len(sentences)), max_word_len_in_batch + 1 + 1)
        # Embeddings
        embedded = {Modules(key): emb(batch, lengths) for key, emb in self.embeddings.items()}

        to_bilstm = {Modules(key): embedded[Modules(key)] for key in self.embeddings.keys()}
        if Modules.CharactersToTokens in to_bilstm:
            last_hidden = to_bilstm[Modules.CharactersToTokens][1]
            # Reshape from (b*s, f) -> (b, s, f)
            to_bilstm[Modules.CharactersToTokens] = last_hidden.reshape(len(lengths), -1, last_hidden.shape[-1])
        if Modules.BERT in to_bilstm:
            to_bilstm[Modules.BERT] = get_emb_by_initial_token_masks(to_bilstm)
        embs_to_bilstm = torch.cat(list(to_bilstm.values()), dim=2)
        if self.residual:
            embs_to_bilstm = self.linear(embs_to_bilstm)

        # Pack the paddings
        packed = torch.nn.utils.rnn.pack_padded_sequence(
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
        bilstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        bilstm_out = self.main_bilstm_out_dropout(bilstm_out)
        # Use residual connections
        if self.residual:
            bilstm_out = bilstm_out + torch.cat((embs_to_bilstm, embs_to_bilstm), dim=2)
        embedded[Modules.BiLSTM] = bilstm_out

        return embedded


class ABLTagger(nn.Module):
    """The ABLTagger, consists of an Encoder(multipart) and a Tagger."""

    def __init__(self, encoder: Encoder, decoders: Dict[Modules, Decoder]):
        """Initialize the tagger."""
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict({key.value: emb for key, emb in decoders.items()})
        self.decoders = cast(Mapping[str, Decoder], self.decoders)

    def forward(self, batch: Dict[BATCH_KEYS, Any]) -> Dict[Modules, torch.Tensor]:
        """Forward pass."""
        encoded: Dict[Modules, torch.Tensor] = self.encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        return {Modules(key): decoder(encoded, batch) for key, decoder in self.decoders.items()}


def get_emb_by_initial_token_masks(encoded) -> torch.Tensor:
    emb = encoded[Modules.BERT]["hidden_states"][-1]  # Only last layer
    tokens_emb = []
    for b in range(emb.shape[0]):
        initial_token_mask = encoded[Modules.BERT]["initial_token_masks"][b]
        output_sent = emb[b, :, :]
        tokens_emb.append(output_sent[initial_token_mask, :])
    padded = torch.nn.utils.rnn.pad_sequence(tokens_emb, batch_first=True)
    return padded
