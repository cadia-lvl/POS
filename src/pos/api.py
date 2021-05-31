"""The API for the tagging model."""
import json
import logging
import pickle
from pathlib import Path
from typing import Tuple, Union, cast

import torch
from torch.utils.data import DataLoader

import pos.core as core
from pos.constants import Modules
from pos.core import FieldedDataset, Fields, Sentence, Sentences, set_device
from pos.data import chunk_dataset, dechunk_dataset
from pos.model import EncodersDecoders
from pos.model.utils import build_model
from pos.train import tag_data_loader

log = logging.getLogger(__name__)


class Tagger:
    """A Tagger is the interface towards a POS tagger model.

    Args:
        model_path: Path to the "tagger.pt".
        device: The (computational) device to use. Either "cpu" for CPU or "cuda:x" for GPU. X is an integer from 0 and up and refers to which GPU to use.
    """

    def __init__(self, model_dir: str, device="cpu"):
        """Initialize a Tagger. Reads the given files."""
        log.info("Setting device.")
        set_device(gpu_flag="cpu" != device)
        log.info("Reading model file...")
        model_path = Path(model_dir)
        assert model_path.exists(), "Model dir not found"
        with open(model_path / "hyperparamters.json") as f:
            kwargs = json.load(f)
            kwargs["trained"] = model_dir
        with open(model_path / "dictionaries.pickle", "rb") as f:
            dicts = pickle.load(f)
        # Þetta mun klikka þegar það á að búa til transformer
        self.model: EncodersDecoders = build_model(kwargs=kwargs, dicts=dicts)
        self.model.load_state_dict(torch.load(model_path / "tagger.pt", map_location=core.device))

    def _infer(self, ds: FieldedDataset, batch_size=16) -> FieldedDataset:
        # If we have a BERT model, we need to chunk
        chunked_ds = ds
        if Modules.BERT in self.model.encoders:
            chunked_ds = chunk_dataset(
                ds,
                tokenizer=self.model.encoders[Modules.BERT].tokenizer,  # type: ignore
                max_sequence_length=self.model.encoders[Modules.BERT].max_length,
            )
        log.info("Predicting tags")
        # Initialize DataLoader
        dl = DataLoader(
            chunked_ds,
            collate_fn=ds.collate_fn,  # type: ignore
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
        if chunked_ds != ds:
            return dechunk_dataset(ds, chunked_ds)
        return chunked_ds

    def tag_sent(self, sent: Sentence) -> Sentence:
        """Tag a (single) sentence. To tag multiple sentences at once (faster) use "tag_bulk".

        Args:
            sent: A tokenized sentence; a Tuple[str, ...] (a tuple of strings)

        Returns: The POS tags a Tuple[str, ...] where the first element in the tuple corresponds to the first token in the input sentence.
        """
        return self.tag_bulk((tuple(sent),), batch_size=1)[0]

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
        return self._infer(ds, batch_size=batch_size).get_field(field=Fields.Tags)

    def lemma_sent(self, sent: Sentence, tags: Sentence) -> Sentence:
        """Lemmatize a (single) sentence. To lemmatize multiple sentences at once (faster) use "lemma_bulk".

        Args:
            sent: A tokenized sentence; a Tuple[str, ...] (a tuple of strings)
            tags: The POS tags of the sentence; a Tuple[str, ...] (a tuple of strings)

        Returns: The lemmas for a sentence, a Tuple[str, ...] where the first element in the tuple corresponds to the first token in the input sentence.
        """
        ds = FieldedDataset(((sent,), (tags,)), fields=(Fields.Tokens, Fields.GoldTags))
        return self.lemma_bulk(ds, batch_size=1)[0]

    def lemma_bulk(
        self,
        dataset: Union[Tuple[Sentences, Sentences], FieldedDataset],
        batch_size=16,
    ) -> Sentences:
        """Lemmatize multiple sentence. This is a faster alternative to "lemma_sent", used for batch processing.

        Args:
            dataset: A collection of tokenized (first) sentence and their tags; Tuple[Sentences, Sentences], Sentences=Tuple[Sentence, ...], Sentence=Tuple[str, ...] or FieldedDataset.
            batch_size: The number of sentences to process at once. Set it to as high as possible without blowing up the memory.

        Returns: The lemmas a Tuple[Tuple[str, ...], ...] or FieldedDataset.
        """
        if type(dataset) == tuple:
            dataset = cast(Tuple[Sentences, Sentences], dataset)
            ds = FieldedDataset(dataset, fields=(Fields.Tokens, Fields.GoldTags))
        elif type(dataset) == FieldedDataset:
            dataset = cast(FieldedDataset, dataset)
            ds = dataset
        else:
            raise ValueError("Bad input type. Use Tuple[Sentences, Sentences] or FieldedDataset")

        return self._infer(ds, batch_size=batch_size).get_field(Fields.Lemmas)


def cast_types(sentences: Union[Sentences, FieldedDataset]) -> FieldedDataset:
    """Convert list/tuple to TokenizedDataset."""
    if type(sentences) == FieldedDataset:
        return cast(FieldedDataset, sentences)
    else:
        sentences = cast(Sentences, sentences)
        return FieldedDataset((sentences,), fields=(Fields.Tokens,))
