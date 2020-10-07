"""The API for the tagging model."""
import pickle
from typing import Union, List, Tuple, cast
from functools import partial
import logging

import torch

from . import types, data

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

    def initialize_dataloader(self, batch_size=16):
        """
        Initialize a dataloader, which preprocesses the data before sending it to the model.
        
        This function can be extended or adjusted using inheritance.
        """
        # We set all the parameters to the function except the "dataset".
        return partial(
            data.data_loader,
            device=self.device,
            dictionaries=self.dictionaries,
            shuffle=False,
            w_emb="pretrained",
            c_emb="standard",
            m_emb="standard",
            batch_size=batch_size,
        )

    def tag_sent(self, sent: Union[Tuple[str], List[str], types.SimpleDataset]):
        """Tag a (single) sentence. To tag multiple sentences at once (faster) use "tag_bulk".
        
        Args:
            sent: A tokenized sentence; a List[str], Tuple[str] or SimpleDataset.
            
        Returns: The POS tags a Tuple[str] or SimpleDataset.
        """
        # Convert list/tuple to SimpleDataset
        if type(sent) == types.SimpleDataset:
            pass
        elif type(sent) == list or type(sent) == tuple:
            sent = cast(list, sent)  # Either is fine for mypy
            sent = types.SimpleDataset((types.Symbols(sent),))
        else:
            raise TypeError(f"Invalid type={type(sent)} to taggin model")

        log.info("Predicting tags")
        predicted_tags = self.model.tag_sents(
            self.initialize_dataloader(batch_size=1)(dataset=sent),
            dictionaries=self.dictionaries,
            criterion=None,
        )
        log.info("Done predicting!")
        # If we were given a single sentence, let's unpack it.
        if len(predicted_tags) == 1:
            return predicted_tags[0]
        # Otherwise we just return the SimpleDataset
        else:
            predicted_tags = cast(types.SimpleDataset, predicted_tags)
            return predicted_tags

    def tag_bulk(
        self,
        dataset: Union[Tuple[Tuple[str]], List[List[str]], types.SimpleDataset],
        batch_size=16,
    ):
        """Tag multiple sentence. This is a faster alternative to "tag_sent", used for batch processing.
        
        Args:
            dataset: A collection of tokenized sentence; a List[List[str]], Tuple[Tuple[str]] or SimpleDataset.
            
        Returns: The POS tags a Tuple[Tuple[str]] or SimpleDataset.
        """
        # Convert list/tuple to SimpleDataset
        if type(dataset) == types.SimpleDataset:
            pass
        elif type(dataset) == list or type(dataset) == tuple:
            dataset = cast(list, dataset)  # Either is fine for mypy
            dataset = types.SimpleDataset((types.Symbols(sent) for sent in dataset))
        else:
            raise TypeError(f"Invalid type={type(dataset)} to taggin model")

        log.info("Predicting tags")
        predicted_tags = self.model.tag_sents(
            self.initialize_dataloader(batch_size=batch_size)(dataset=dataset),
            dictionaries=self.dictionaries,
            criterion=None,
        )
        log.info("Done predicting!")
        predicted_tags = cast(types.SimpleDataset, predicted_tags)
        return predicted_tags

