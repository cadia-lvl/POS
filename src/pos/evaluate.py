"""A collection of types and function used to evaluate the performance of a tagger."""
from typing import Callable, Iterable, Tuple, List, Dict, Set, Any, Union
import logging
from collections import Counter
from pathlib import Path
import pickle
from statistics import stdev

from .core import Sentences, Vocab, VocabMap, FieldedDataset, Dicts, Fields

log = logging.getLogger(__name__)


class Experiment:
    """An Experiment holds information about an experiment, i.e. predicted tags, lemmas and vocabularies."""

    def __init__(
        self,
        predictions: FieldedDataset,
        train_tokens: Vocab,
        morphlex_tokens: Vocab,
        pretrained_vocab: Vocab,
        train_lemmas: Vocab = None,
    ):
        """Initialize an experiment given the predictions and vocabulary."""
        self.predictions = predictions
        # Get the tokens and retrive the vocab.
        self.test_tokens = Vocab.from_symbols(self.predictions.get_field(Fields.Tokens))
        self.known_tokens = train_tokens.intersection(self.test_tokens)
        if Fields.Lemmas in self.predictions.fields:
            self.test_lemmas = Vocab.from_symbols(
                self.predictions.get_field(Fields.GoldLemmas)
            )
            self.known_lemmas = train_lemmas.intersection(self.test_lemmas)
        log.debug("Creating vocabs")
        morphlex_tokens = morphlex_tokens.intersection(self.test_tokens)  # type: ignore
        pretrained_tokens = pretrained_vocab.intersection(self.test_tokens)  # type: ignore
        # fmt: off
        self.unknown_tokens = self.test_tokens.difference(self.known_tokens)  # pylint: disable=no-member
        self.train_pretrained_tokens = self.known_tokens.intersection(pretrained_tokens).difference(morphlex_tokens)
        self.train_pretrained_morphlex_tokens = self.known_tokens.intersection(pretrained_tokens).intersection(morphlex_tokens)
        self.train_morphlex_tokens = self.known_tokens.intersection(morphlex_tokens).difference(pretrained_tokens)
        self.train_tokens_only = self.known_tokens.difference(pretrained_tokens).difference(morphlex_tokens)
        self.test_pretrained_tokens = self.unknown_tokens.intersection(pretrained_tokens).difference(morphlex_tokens)
        self.test_pretrained_morphlex_tokens = self.unknown_tokens.intersection(pretrained_tokens).intersection(morphlex_tokens)
        self.test_morphlex_tokens = self.unknown_tokens.intersection(morphlex_tokens).difference(pretrained_tokens)
        self.test_tokens_only = self.unknown_tokens.difference(pretrained_tokens).difference(morphlex_tokens)
        # fmt: on

    @staticmethod
    def from_file(path: Path, morphlex_path: Path, pretrained_path: Path):
        """Create an Experiment from a given directory of an experimental results."""
        log.debug(f"Reading experiment={path}")
        log.debug("Reading predictions")
        predictions = FieldedDataset.from_file(
            str(path / "predictions.tsv"), (Fields.Tokens, Fields.GoldTags, Fields.Tags)
        )
        log.debug("Reading vocabs")
        morphlex_vocab = Vocab.from_file(str(morphlex_path))
        pretrained_vocab = Vocab.from_file(str(pretrained_path))
        log.debug("Reading training vocab")
        train_vocab = Vocab.from_file(str(path / "known_toks.txt"))
        log.debug(f"Done reading experiment={path}")
        return Experiment(
            predictions=predictions,
            train_tokens=train_vocab,
            morphlex_tokens=morphlex_vocab,
            pretrained_vocab=pretrained_vocab,
        )

    @staticmethod
    def from_predictions(
        test_ds: FieldedDataset,
        predictions: Sentences,
        train_tokens: Vocab,
        morphlex_tokens: Vocab,
        pretrained_tokens: Vocab,
        train_lemmas: Vocab = None,
        field=Fields.Tags,
    ):
        """Create an Experiment from predicted tags, test_dataset and dictionaries."""
        test_ds = test_ds.add_field(predictions, field)
        return Experiment(
            test_ds,
            train_tokens=train_tokens,
            morphlex_tokens=morphlex_tokens,
            pretrained_vocab=pretrained_tokens,
            train_lemmas=train_lemmas,
        )

    @staticmethod
    def tag_accuracy_closure(
        test_ds: FieldedDataset,
        train_vocab: Vocab,
        morphlex_vocab: Vocab,
        pretrained_vocab: Vocab,
    ) -> Callable[[Sentences], Tuple[Dict[str, float], Dict[str, int]]]:
        """Create an Experiment from predicted tags, test_dataset and dictionaries."""

        def calculate_accuracy(
            tags: Sentences,
        ) -> Tuple[Dict[str, float], Dict[str, int]]:
            """Closure."""
            return Experiment.from_predictions(
                test_ds,
                tags,
                train_tokens=train_vocab,
                morphlex_tokens=morphlex_vocab,
                pretrained_tokens=pretrained_vocab,
            ).tagging_accuracy()

        return calculate_accuracy

    @staticmethod
    def lemma_accuracy_closure(
        test_ds: FieldedDataset,
        train_tokens: Vocab,
        morphlex_tokens: Vocab,
        pretrained_tokens: Vocab,
        train_lemmas: Vocab,
    ) -> Callable[[Sentences], Tuple[Dict[str, float], Dict[str, int]]]:
        """Create an Experiment from predicted tags, test_dataset and dictionaries."""

        def calculate_accuracy(
            lemmas: Sentences,
        ) -> Tuple[Dict[str, float], Dict[str, int]]:
            """Closure."""
            return Experiment.from_predictions(
                test_ds,
                lemmas,
                train_tokens=train_tokens,
                morphlex_tokens=morphlex_tokens,
                pretrained_tokens=pretrained_tokens,
                train_lemmas=train_lemmas,
                field=Fields.Lemmas,
            ).lemma_accuracy()

        return calculate_accuracy

    def accuracy(
        self, vocab: Set[str] = None, gold_field=Fields.GoldTags, pred_field=Fields.Tags
    ) -> Tuple[float, int]:
        """Calculate the accuracy given a vocabulary to filter on. If nothing is provided, we do not filter."""

        def in_vocabulary(token, vocab):
            """Filter condition."""
            # Empty vocab implies all
            if vocab is None:
                return True
            else:
                return token in vocab

        total = sum(
            (
                1
                for tokens in self.predictions.get_field(Fields.Tokens)
                for token in tokens
                if in_vocabulary(token, vocab)
            )
        )
        if total == 0:
            return (0.0, 0)
        correct = sum(
            (
                predicted == gold
                for tokens, golds, predicted in zip(
                    self.predictions.get_field(Fields.Tokens),
                    self.predictions.get_field(gold_field),
                    self.predictions.get_field(pred_field),
                )
                for token, gold, predicted in zip(tokens, golds, predicted)
                if in_vocabulary(token, vocab)
            )
        )
        return (correct / total, total)

    def tagging_accuracy(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Return all accuracies."""

        def _tagging_accuracy(index):
            return {
                "Total": self.accuracy()[index],
                "Unknown": self.accuracy(self.unknown_tokens)[index],
                "Known": self.accuracy(self.known_tokens)[index],
                "Known-Wemb": self.accuracy(self.train_pretrained_tokens)[index],
                "Known-Wemb+Morph": self.accuracy(
                    self.train_pretrained_morphlex_tokens
                )[index],
                "Known-Morph": self.accuracy(self.train_morphlex_tokens)[index],
                "Seen": self.accuracy(self.train_tokens_only)[index],
                "Unknown-Wemb": self.accuracy(self.test_pretrained_tokens)[index],
                "Unknown-Wemb+Morph": self.accuracy(
                    self.test_pretrained_morphlex_tokens
                )[index],
                "Unknown-Morph": self.accuracy(self.test_morphlex_tokens)[index],
                "Unseen": self.accuracy(self.test_tokens_only)[index],
            }

        accuracy = _tagging_accuracy(0)
        total = _tagging_accuracy(1)
        return accuracy, total  # type: ignore

    def lemma_accuracy(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Return all lemma accuracies."""

        def _tagging_accuracy(index):
            return {
                "Total": self.accuracy(
                    gold_field=Fields.GoldLemmas, pred_field=Fields.Lemmas
                )[index],
                "Unknown": self.accuracy(
                    self.unknown_tokens,
                    gold_field=Fields.GoldLemmas,
                    pred_field=Fields.Lemmas,
                )[index],
                "Known": self.accuracy(
                    self.known_tokens,
                    gold_field=Fields.GoldLemmas,
                    pred_field=Fields.Lemmas,
                )[index],
            }

        accuracy = _tagging_accuracy(0)
        total = _tagging_accuracy(1)
        return accuracy, total  # type: ignore

    def error_profile(self):
        """Return an error profile with counts of errors (tagger > gold)."""
        return Counter(
            f"{predicted} > {gold}"
            for tokens, gold_tags, predicted_tags in self.predictions
            for token, gold, predicted in zip(tokens, gold_tags, predicted_tags)
            if gold != predicted
        )


def get_average(
    accuracies: List[Dict[str, Union[float, int]]]
) -> Dict[str, Tuple[float, float]]:
    """Get the average (accuracy, std_dev) and (total, std_dev) of a list of accuracies."""
    length = len(accuracies)
    keys = list(accuracies[0].keys())
    totals: Dict[str, Tuple[float, float]] = {}
    for key in keys:
        totals[key] = (
            average([accuracy[key] for accuracy in accuracies]),
            stdev([accuracy[key] for accuracy in accuracies])
            if len(accuracies) >= 2
            else 0.0,
        )
    return totals


def average(values: List[Union[int, float]]) -> float:
    """Return the average."""
    return sum(values) / len(values)


def all_accuracy_average(
    experiments: List[Experiment],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Return the average of all accuracies."""
    all_accuracies = []
    all_totals = []
    for experiment in experiments:
        accuracies, totals = experiment.tagging_accuracy()
        all_accuracies.append(accuracies)
        all_totals.append(totals)
    return (
        get_average(all_accuracies),
        get_average(
            all_totals,
        ),
    )


def collect_experiments(
    directory: str, morphlex_vocab: str, pretrained_vocab: str
) -> List[Experiment]:
    """Collect model predictions in the directory. If the directory contains other directories, it will recurse into it."""
    experiments: List[Experiment] = []
    root = Path(directory)
    directories = [d for d in root.iterdir() if d.is_dir()]
    if directories:
        experiments.extend(
            [
                experiment
                for d in directories
                for experiment in collect_experiments(
                    str(d),
                    morphlex_vocab=morphlex_vocab,
                    pretrained_vocab=pretrained_vocab,
                )
            ]
        )
        return experiments
    # No directories found
    else:
        return [
            Experiment.from_file(root, Path(morphlex_vocab), Path(pretrained_vocab))
        ]


def format_results(
    results: Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]
) -> str:
    """Format the Accuracy results for pretty printing."""

    def rows(
        results: Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]
    ) -> Iterable[str]:
        accuracies, totals = results
        keys = accuracies.keys()
        for key in keys:
            yield f"{key:<20}: {accuracies[key][0]*100:>02.2f} ±{accuracies[key][1]*100:>02.2f}, {totals[key][0]:>} ±{totals[key][1]:>}"

    return "\n".join(rows(results))