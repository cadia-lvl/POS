"""The API for the tagging model."""
from typing import Union, cast
import logging

import torch

import pos.core as core
from pos.model import ABLTagger, Modules
from pos.core import FieldedDataset, Fields, Sentence, Sentences, set_device
from pos.train import tag_data_loader
from pos.data import collate_fn, chunk_dataset, dechunk_dataset

log = logging.getLogger(__name__)


class Tagger:
    """A Tagger is the interface towards a POS tagger model.

    Args:
        model_path: Path to the "tagger.pt".
        device: The (computational) device to use. Either "cpu" for CPU or "cuda:x" for GPU. X is an integer from 0 and up and refers to which GPU to use.
    """

    def __init__(self, model_file=None, device="cpu"):
        """Initialize a Tagger. Reads the given files."""
        log.info("Setting device.")
        set_device(gpu_flag="cpu" != device)
        log.info("Reading model file...")
        self.model: ABLTagger = torch.load(model_file, map_location=core.device)

    def tag_sent(self, sent: Sentence) -> Sentence:
        """Tag a (single) sentence. To tag multiple sentences at once (faster) use "tag_bulk".

        Args:
            sent: A tokenized sentence; a Tuple[str, ...] (a tuple of strings)

        Returns: The POS tags a Tuple[str, ...] where the first element in the tuple corresponds to the first token in the input sentence.
        """
        return self.tag_bulk((sent,), batch_size=1)[0]

    def tag_bulk(
        self,
        dataset: Union[Sentences, FieldedDataset],
        batch_size=16,
    ) -> Sentences:
        """Tag multiple sentence. This is a faster alternative to "tag_sent", used for batch processing.

        Args:
            dataset: A collection of tokenized sentence; a Tuple[Tuple[str, ...], ...] or FieldedDataset.
            batch_size: The number of sentences to process at once. Set it to as high as possible without blowing up the memory.

        Returns: The POS tags a Tuple[Tuple[str, ...], ...] or FieldedDataset.
        """
        ds = cast_types(dataset)
        chunked_ds = chunk_dataset(
            ds,
            tokenizer=self.model.encoder.embeddings[Modules.BERT.value].tokenizer,
            max_sequence_length=self.model.encoder.embeddings[
                Modules.BERT.value
            ].max_length,
        )
        log.info("Predicting tags")
        # Initialize DataLoader
        # with collate_fn - that function needs the dict
        # The dict needs to be loaded based on some files
        dl = torch.utils.data.DataLoader(
            chunked_ds,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=batch_size,
        )

        _, values = tag_data_loader(self.model, dl)
        log.info("Done predicting!")

        if Modules.Tagger in values:
            chunked_ds = chunked_ds.add_field(values[Modules.Tagger], Fields.Tags)
        if Modules.Lemmatizer in values:
            chunked_ds = chunked_ds.add_field(values[Modules.Lemmatizer], Fields.Lemmas)
        # Dechunk
        dechunk_ds = dechunk_dataset(ds, chunked_ds)
        # TODO: Add lemmas
        return dechunk_ds.get_field(field=Fields.Tags)


def cast_types(sentences: Union[Sentences, FieldedDataset]) -> FieldedDataset:
    """Convert list/tuple to TokenizedDataset."""
    if type(sentences) == FieldedDataset:
        return cast(FieldedDataset, sentences)
    else:
        sentences = cast(Sentences, sentences)
        sentences = (sentences,)  # type: ignore
        return FieldedDataset(sentences, fields=(Fields.Tokens,))  # type: ignore
