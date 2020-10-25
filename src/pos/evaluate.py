"""A collection of types and function used to evaluate the performance of a tagger."""
from typing import Tuple, List, Dict, Set, Any
import logging
from collections import Counter
from pathlib import Path
import pickle

from .core import DoubleTaggedDataset, Vocab, VocabMap, SequenceTaggingDataset
from .data import Modules

log = logging.getLogger(__name__)


class Experiment:
    """An Experiment holds information about an experiment, i.e. predicted tags and vocabularies."""

    def __init__(
        self,
        predictions: DoubleTaggedDataset,
        train_vocab: Vocab,
        dicts: Dict[Modules, VocabMap],
    ):
        """Initialize an experiment given the predictions and vocabulary."""
        self.predictions = predictions
        # Get the tokens and retrive the vocab.
        self.test_vocab = Vocab.from_symbols(self.predictions.unpack()[0])
        self.known_vocab = train_vocab.intersection(self.test_vocab)
        self.dicts = dicts

        log.info("Creating vocabs")
        morphlex_vocab = get_by_key_backup(
            "m_map", Modules.MorphLex, self.dicts
        ).intersection(self.test_vocab)
        wemb_vocab = get_by_key_backup(
            "w_map", Modules.WordEmbeddings, self.dicts
        ).intersection(self.test_vocab)
        # fmt: off
        self.unknown_vocab = self.test_vocab.difference(self.known_vocab)  # pylint: disable=no-member
        self.known_wemb_vocab = self.known_vocab.intersection(wemb_vocab).difference(morphlex_vocab)
        self.known_wemb_morphlex_vocab = self.known_vocab.intersection(wemb_vocab).intersection(morphlex_vocab)
        self.known_morphlex_vocab = self.known_vocab.intersection(morphlex_vocab).difference(wemb_vocab)
        self.seen_vocab = self.known_vocab.difference(wemb_vocab).difference(morphlex_vocab)
        self.unknown_wemb_vocab = self.unknown_vocab.intersection(wemb_vocab).difference(morphlex_vocab)
        self.unknown_wemb_morphlex_vocab = self.unknown_vocab.intersection(wemb_vocab).intersection(morphlex_vocab)
        self.unknown_morphlex_vocab = self.unknown_vocab.intersection(morphlex_vocab).difference(wemb_vocab)
        self.unseen_vocab = self.unknown_vocab.difference(wemb_vocab).difference(morphlex_vocab)
        # fmt: on

    @staticmethod
    def from_file(path: Path, predictions=None):
        """Create an Experiment from a given path of an experimental results. Optionally point to a specific prediction ds."""
        log.info(f"Reading experiment={path}")
        log.info("Reading predictions")
        if predictions:
            predictions = DoubleTaggedDataset.from_file(predictions)
        else:
            predictions = DoubleTaggedDataset.from_file(str(path / "predictions.tsv"))
        log.info("Reading dicts")
        with (path / "dictionaries.pickle").open("rb") as f:
            dicts = pickle.load(f)
        log.info("Reading training vocab")
        train_vocab = Vocab.from_file(path / "known_toks.txt")
        log.info(f"Done reading experiment={path}")
        return Experiment(predictions=predictions, train_vocab=train_vocab, dicts=dicts)

    @staticmethod
    def from_predictions(
        predicted_tags, test_ds: SequenceTaggingDataset, dicts: Dict[Modules, VocabMap]
    ):
        """Create an Experiment from predicted tags, test_dataset and dictionaries."""
        train_vocab = Vocab(dicts[Modules.WordEmbeddings].w2i.keys())
        predicted_ds = DoubleTaggedDataset(
            tuple((tokens, tags, predicted_tags))
            for tokens, tags, predicted_tags in zip(*test_ds.unpack(), predicted_tags)
        )
        return Experiment(predicted_ds, train_vocab=train_vocab, dicts=dicts)

    def accuracy(self, vocab: Set[str] = None) -> Tuple[float, int]:
        """Calculate the accuracy given a vocabulary to filter on. If nothing is provided, we do not filter."""
        if vocab is None:
            total = sum(
                (
                    1
                    for tokens, gold_tags, predicted_tags in self.predictions
                    for _, gold, predicted in zip(tokens, gold_tags, predicted_tags)
                )
            )
            correct = sum(
                (
                    predicted == gold
                    for tokens, gold_tags, predicted_tags in self.predictions
                    for _, gold, predicted in zip(tokens, gold_tags, predicted_tags)
                )
            )
            return (correct / total, total)
        # We need to filter based on the given vocabulary
        total = sum(
            (
                1
                for tokens, gold_tags, predicted_tags in self.predictions
                for token, gold, predicted in zip(tokens, gold_tags, predicted_tags)
                if token in vocab
            )
        )
        if total == 0:
            return (0.0, 0)
        correct = sum(
            (
                predicted == gold
                for tokens, gold_tags, predicted_tags in self.predictions
                for token, gold, predicted in zip(tokens, gold_tags, predicted_tags)
                if token in vocab
            )
        )
        return (correct / total, total)

    def all_accuracy(self):
        """Return all accuracies."""
        return (
            self.accuracy(),
            self.accuracy(self.unknown_vocab),
            self.accuracy(self.known_vocab),
            self.accuracy(self.known_wemb_vocab),
            self.accuracy(self.known_wemb_morphlex_vocab),
            self.accuracy(self.known_morphlex_vocab),
            self.accuracy(self.seen_vocab),
            self.accuracy(self.unknown_wemb_vocab),
            self.accuracy(self.unknown_wemb_morphlex_vocab),
            self.accuracy(self.unknown_morphlex_vocab),
            self.accuracy(self.unseen_vocab),
        )

    def error_profile(self):
        """Return an error profile with counts of errors (tagger > gold)."""
        return Counter(
            f"{predicted} > {gold}"
            for tokens, gold_tags, predicted_tags in self.predictions
            for token, gold, predicted in zip(tokens, gold_tags, predicted_tags)
            if gold != predicted
        )


def get_by_key_backup(key1, key2, dictionary: Dict[Any, VocabMap]) -> Set:
    """Fetch value from dict by key1 first, then key2 or return empty set."""
    if key1 in dictionary:
        return set(dictionary[key1].w2i.keys())
    elif key2 in dictionary:
        return set(dictionary[key2].w2i.keys())
    else:
        return set()


def get_average(accuracies: List[Tuple[float, int]]) -> Tuple[float, int]:
    """Get the average percentage and total of a list of accuracies."""
    return (
        sum(accuracy[0] for accuracy in accuracies) / len(accuracies),
        int(round(sum(accuracy[1] for accuracy in accuracies) / len(accuracies))),
    )


def all_accuracy_average(experiments: List[Experiment]) -> List[Tuple[float, int]]:
    """Return the average of all accuracies."""
    all_accuracy = [experiment.all_accuracy() for experiment in experiments]
    return [get_average(accuracies) for accuracies in zip(*all_accuracy)]
