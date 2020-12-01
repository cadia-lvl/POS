"""The API for the tagging model."""
import pickle
from typing import Union, cast, Sequence
import logging

import torch

from .core import FieldedDataset
from .train import tag_data_loader
from .data import collate_fn

log = logging.getLogger(__name__)


class Tagger:
    """A Tagger is the interface towards a POS tagger model.

    Args:
        model_path: Path to the "tagger.pt".
        dictionaries: Path to the "dictionaries.pickle".
        device: The (computational) device to use. Either "cpu" for CPU or "cuda:x" for GPU. X is an integer from 0 and up and refers to which GPU to use.
    """

    def __init__(self, model_file=None, dictionaries_file=None, device="cpu"):
        """Initialize a Tagger. Reads the given files."""
        log.info("Setting device.")
        self.device = torch.device(device)
        log.info("Reading dictionaries")
        with open(dictionaries_file, "rb") as f:
            self.dictionaries = pickle.load(f)
        log.info("Reading model file")
        self.model = torch.load(model_file, map_location=self.device)

    def tag_sent(self, sent: Sequence[str]) -> Sequence[str]:
        """Tag a (single) sentence. To tag multiple sentences at once (faster) use "tag_bulk".

        Args:
            sent: A tokenized sentence; a Sequence[str].

        Returns: The POS tags a Sequence[str].
        """
        return self.tag_bulk([sent], batch_size=1)[0]

    def tag_bulk(
        self,
        dataset: Union[Sequence[Sequence[str]], FieldedDataset],
        batch_size=16,
    ):
        """Tag multiple sentence. This is a faster alternative to "tag_sent", used for batch processing.

        Args:
            dataset: A collection of tokenized sentence; a List[List[str]], Tuple[Tuple[str]] or SimpleDataset.

        Returns: The POS tags a Tuple[Tuple[str]] or SimpleDataset.
        """
        dataset = cast_types(dataset)
        log.info("Predicting tags")
        # Initialize DataLoader
        # with collate_fn - that function needs the dict
        # The dict needs to be loaded based on some files
        dl = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=batch_size,
        )

        predicted_tags = tag_data_loader(self.model, dl)
        log.info("Done predicting!")
        return predicted_tags


def cast_types(
    sentences: Union[Sequence[Sequence[str]], FieldedDataset]
) -> FieldedDataset:
    """Convert list/tuple to TokenizedDataset."""
    if type(sentences) == FieldedDataset:
        return cast(FieldedDataset, sentences)
    elif type(sentences) == list or type(sentences) == tuple:
        sentences = tuple(sentence for sentence in sentences)
        return FieldedDataset(sentences, fields=["tokens"])
    else:
        raise TypeError(f"Invalid type={type(sentences)} to taggin model")
