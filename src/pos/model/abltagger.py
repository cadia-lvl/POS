"""The main classes used for creating models."""
import abc
from enum import Enum
from typing import Any, Dict, Optional, Sequence, cast

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
    Lemmatizer = "character_decoder"


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
    def decode(
        self, encoded: Dict[Modules, Any], batch: Dict[BATCH_KEYS, Any], tags: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Run the decoder on the batch."""

    def forward(self, encoded, batch, tags=None) -> torch.Tensor:
        """Run a generic forward pass for the Embeddings."""
        self.add_targets(batch)
        return self.decode(encoded, batch, tags)


def get_emb_by_initial_token_masks(encoded) -> torch.Tensor:
    emb = encoded[Modules.BERT]["hidden_states"][-1]  # Only last layer
    tokens_emb = []
    for b in range(emb.shape[0]):
        initial_token_mask = encoded[Modules.BERT]["initial_token_masks"][b]
        output_sent = emb[b, :, :]
        tokens_emb.append(output_sent[initial_token_mask, :])
    padded = nn.utils.rnn.pad_sequence(tokens_emb, batch_first=True)
    return padded
