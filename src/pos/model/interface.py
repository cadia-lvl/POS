"""The main classes used for creating models."""
import abc
from typing import Any, Dict

from pos.core import Sentences
from torch import nn


class BatchPostprocess(metaclass=abc.ABCMeta):
    """An interface to handle postprocessing for modules."""

    @abc.abstractmethod
    def postprocess(self, batch: Dict[str, Any]) -> Sentences:
        """Postprocess the model output."""
        raise NotImplementedError


class BatchPreprocess(metaclass=abc.ABCMeta):
    """An interface to handle preprocessing for modules."""

    @abc.abstractmethod
    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the sentence batch."""
        raise NotImplementedError


class Encoder(BatchPreprocess, nn.Module, metaclass=abc.ABCMeta):
    """A module which accepts string inputs and embeds them to tensors."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run a generic forward pass for the Embeddings."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension."""
        raise NotImplementedError


class Decoder(BatchPostprocess, nn.Module, metaclass=abc.ABCMeta):
    """A module which accepts an sentence embedding and outputs another tensor."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

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
    def add_targets(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Add the decoder targets to the batch dictionary."""


class EncodersDecoders(nn.Module):
    """Multiple encoders and decoders."""

    def __init__(self, encoders: Dict[str, Encoder], decoders: Dict[str, Decoder]):
        """Initialize the model. The Lemmatizer depends on the Tagger."""
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass."""
        for encoder in self.encoders.values():
            batch = encoder.preprocess(batch)  # type: ignore
            batch = encoder(batch)
        for decoder in self.decoders.values():
            batch = decoder.add_targets(batch)  # type: ignore
            batch = decoder(batch)
        results = {key: batch[key] for key in self.decoders.keys()}  # type: ignore
        return results
