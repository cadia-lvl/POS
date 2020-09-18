"""A collection of types and function used to evaluate the performance of a tagger."""
from typing import Tuple, List, Dict, Set
import logging
from collections import Counter
from pathlib import Path
import pickle

from .types import PredictedDataset, Vocab, VocabMap

log = logging.getLogger()


class Experiment:
    """An Experiment holds information about an experiment, i.e. predicted tags and vocabularies."""

    def __init__(
        self,
        predictions: PredictedDataset,
        train_vocab: Vocab,
        dicts: Dict[str, VocabMap],
    ):
        """Initialize an experiment given the predictions and vocabulary."""
        self.predictions = predictions
        # Get the tokens and retrive the vocab.
        self.test_vocab = Vocab.from_symbols(self.predictions.unpack()[0])
        self.known_vocab = train_vocab.intersection(self.test_vocab)
        self.dicts = dicts

        log.info("Creating vocabs")
        if "m_map" in self.dicts:
            morphlex_vocab = set(self.dicts["m_map"].w2i.keys()).intersection(
                self.test_vocab
            )
        else:
            morphlex_vocab = set()
        if "w_map" in self.dicts:
            wemb_vocab = set(self.dicts["w_map"].w2i.keys()).intersection(
                self.test_vocab
            )
        else:
            wemb_vocab = set()
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
            predictions = PredictedDataset.from_file(predictions)
        else:
            predictions = PredictedDataset.from_file(str(path / "predictions.tsv"))
        log.info("Reading dicts")
        with (path / "dictionaries.pickle").open("rb") as f:
            dicts = pickle.load(f)
        log.info("Reading training vocab")
        train_vocab = Vocab.from_file(path / "known_toks.txt")
        log.info(f"Done reading experiment={path}")
        return Experiment(predictions=predictions, train_vocab=train_vocab, dicts=dicts)

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
