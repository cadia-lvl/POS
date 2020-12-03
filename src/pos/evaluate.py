"""A collection of types and function used to evaluate the performance of a tagger."""
from typing import Callable, Tuple, List, Dict, Set, Any, Union
import logging
from collections import Counter
from pathlib import Path
import pickle

from .core import Sentences, Vocab, VocabMap, FieldedDataset, Dicts, Fields

log = logging.getLogger(__name__)


class Experiment:
    """An Experiment holds information about an experiment, i.e. predicted tags and vocabularies."""

    def __init__(
        self,
        predictions: FieldedDataset,
        train_vocab: Vocab,
        dicts: Dict[Dicts, VocabMap],
    ):
        """Initialize an experiment given the predictions and vocabulary."""
        self.predictions = predictions
        # Get the tokens and retrive the vocab.
        self.test_vocab = Vocab.from_symbols(self.predictions.get_field(Fields.Tokens))
        self.known_vocab = train_vocab.intersection(self.test_vocab)
        self.dicts = dicts

        log.info("Creating vocabs")
        morphlex_vocab = get_by_key_backup(
            "m_map", Dicts.MorphLex, self.dicts
        ).intersection(self.test_vocab)
        wemb_vocab = get_by_key_backup("w_map", Dicts.Tokens, self.dicts).intersection(
            self.test_vocab
        )
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
    def from_file(path: Path):
        """Create an Experiment from a given path of an experimental results. Optionally point to a specific prediction ds."""
        log.info(f"Reading experiment={path}")
        log.info("Reading predictions")
        predictions = FieldedDataset.from_file(
            str(path), (Fields.Tokens, Fields.GoldTags, Fields.Tags)
        )
        log.info("Reading dicts")
        with (path / "dictionaries.pickle").open("rb") as f:
            dicts = pickle.load(f)
        log.info("Reading training vocab")
        train_vocab = Vocab.from_file(path / "known_toks.txt")
        log.info(f"Done reading experiment={path}")
        return Experiment(predictions=predictions, train_vocab=train_vocab, dicts=dicts)

    @staticmethod
    def from_tag_predictions(
        test_ds: FieldedDataset, dicts: Dict[Dicts, VocabMap], tags: Sentences
    ):
        """Create an Experiment from predicted tags, test_dataset and dictionaries."""
        train_vocab = Vocab(dicts[Dicts.Tokens].w2i.keys())
        test_ds = test_ds.add_field(tags, Fields.Tags)
        return Experiment(test_ds, train_vocab=train_vocab, dicts=dicts)

    def calculate_accuracy(self, vocab: Set[str] = None) -> Tuple[float, int]:
        """Calculate the accuracy given a vocabulary to filter on. If nothing is provided, we do not filter."""
        if vocab is None:
            total = sum(
                (
                    1
                    for tokens, gold_tags, predicted_tags in zip(
                        self.predictions.get_field(Fields.Tokens),
                        self.predictions.get_field(Fields.GoldTags),
                        self.predictions.get_field(Fields.Tags),
                    )
                    for _ in zip(tokens, gold_tags, predicted_tags)
                )
            )
            correct = sum(
                (
                    predicted == gold
                    for tokens, gold_tags, predicted_tags in zip(
                        self.predictions.get_field(Fields.Tokens),
                        self.predictions.get_field(Fields.GoldTags),
                        self.predictions.get_field(Fields.Tags),
                    )
                    for _, gold, predicted in zip(tokens, gold_tags, predicted_tags)
                )
            )
            return (correct / total, total)
        # We need to filter based on the given vocabulary
        total = sum(
            (
                1
                for tokens, gold_tags, predicted_tags in zip(
                    self.predictions.get_field(Fields.Tokens),
                    self.predictions.get_field(Fields.GoldTags),
                    self.predictions.get_field(Fields.Tags),
                )
                for token, _, _ in zip(tokens, gold_tags, predicted_tags)
                if token in vocab
            )
        )
        if total == 0:
            return (0.0, 0)
        correct = sum(
            (
                predicted == gold
                for tokens, gold_tags, predicted_tags in zip(
                    self.predictions.get_field(Fields.Tokens),
                    self.predictions.get_field(Fields.GoldTags),
                    self.predictions.get_field(Fields.Tags),
                )
                for token, gold, predicted in zip(tokens, gold_tags, predicted_tags)
                if token in vocab
            )
        )
        return (correct / total, total)

    @staticmethod
    def all_accuracy_closure(
        test_ds: FieldedDataset, dicts: Dict[Dicts, VocabMap]
    ) -> Callable[[Sentences], Tuple[Dict[str, float], Dict[str, int]]]:
        """Create an Experiment from predicted tags, test_dataset and dictionaries."""

        def calculate_accuracy(
            tags: Sentences,
        ) -> Tuple[Dict[str, float], Dict[str, int]]:
            """Closure."""
            return Experiment.from_tag_predictions(test_ds, dicts, tags).all_accuracy()

        return calculate_accuracy

    def all_accuracy(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Return all accuracies."""

        def _all_accuracy(index):
            return {
                "Total": self.calculate_accuracy()[index],
                "Unknown": self.calculate_accuracy(self.unknown_vocab)[index],
                "Known": self.calculate_accuracy(self.known_vocab)[index],
                "Known-Wemb": self.calculate_accuracy(self.known_wemb_vocab)[index],
                "Known-Wemb+Morph": self.calculate_accuracy(
                    self.known_wemb_morphlex_vocab
                )[index],
                "Known-Morph": self.calculate_accuracy(self.known_morphlex_vocab)[
                    index
                ],
                "Seen": self.calculate_accuracy(self.seen_vocab)[index],
                "Unknown-Wemb": self.calculate_accuracy(self.unknown_wemb_vocab)[index],
                "Unknown-Wemb+Morph": self.calculate_accuracy(
                    self.unknown_wemb_morphlex_vocab
                )[index],
                "Unknown-Morph": self.calculate_accuracy(self.unknown_morphlex_vocab)[
                    index
                ],
                "Unseen": self.calculate_accuracy(self.unseen_vocab)[index],
            }

        accuracy = _all_accuracy(0)
        total = _all_accuracy(1)
        return accuracy, total  # type: ignore

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


def get_average(accuracies: List[Dict[str, Union[float, int]]]) -> Dict[str, float]:
    """Get the average percentage and total of a list of accuracies."""
    length = len(accuracies)
    keys = list(accuracies[0].keys())
    totals = {}
    for key in keys:
        totals[key] = 0
        for accuracy in accuracies:
            totals[key] += accuracy[key]
    return {key: total / length for key, total in totals.items()}


def all_accuracy_average(
    experiments: List[Experiment],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return the average of all accuracies."""
    all_accuracies = []
    all_totals = []
    for experiment in experiments:
        accuracies, totals = experiment.all_accuracy()
        all_accuracies.append(accuracies)
        all_totals.append(totals)
    return (get_average(all_accuracies), get_average(all_totals))
