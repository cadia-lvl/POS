"""A collection of types and function used to evaluate the performance of a tagger."""
import json
from typing import Callable, Iterable, Tuple, List, Dict, Set, Any, Union
import logging
from collections import Counter
from pathlib import Path
from statistics import stdev

from .core import Sentences, Vocab, FieldedDataset, Fields

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
        if train_lemmas is not None:
            self.test_lemmas = Vocab.from_symbols(
                self.predictions.get_field(Fields.GoldLemmas)
            )
            self.known_lemmas = train_lemmas.intersection(self.test_lemmas)
            self.unknown_lemmas = self.test_lemmas.difference(self.known_lemmas)
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
        # Start by reading what type of model it is
        with open(str(path / "hyperparamters.json")) as f:
            hyperparams = json.load(f)
        fields = (Fields.Tokens,)
        if hyperparams["tagger"] and hyperparams["lemmatizer"]:
            fields = fields + (
                Fields.GoldTags,
                Fields.GoldLemmas,
                Fields.Tags,
                Fields.Lemmas,
            )
        elif hyperparams["tagger"] and not hyperparams["lemmatizer"]:
            fields = fields + (Fields.GoldTags, Fields.Tags)
        elif not hyperparams["tagger"] and hyperparams["lemmatizer"]:
            fields = fields + (Fields.GoldLemmas, Fields.Lemmas)
        else:
            raise ValueError("Bad hyperparameters, no tagger nor lemmatizer")
        train_lemmas = None
        if Fields.Lemmas in fields:
            train_lemmas = Vocab.from_file(str(path / "known_lemmas.txt"))
        log.debug("Reading predictions")
        predictions = FieldedDataset.from_file(str(path / "predictions.tsv"), fields)
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
            train_lemmas=train_lemmas,
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
        self,
        vocab: Set[str] = None,
        comparison_field=Fields.Tokens,
        gold_field=Fields.GoldTags,
        pred_field=Fields.Tags,
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
                for tokens in self.predictions.get_field(comparison_field)
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
                    self.predictions.get_field(comparison_field),
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

        def _lemma_accuracy(index):
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
                "Unknown-Lemmas": self.accuracy(
                    self.unknown_lemmas,
                    comparison_field=Fields.GoldLemmas,
                    gold_field=Fields.GoldLemmas,
                    pred_field=Fields.Lemmas,
                )[index],
                "Known-Lemmas": self.accuracy(
                    self.known_lemmas,
                    comparison_field=Fields.GoldLemmas,
                    gold_field=Fields.GoldLemmas,
                    pred_field=Fields.Lemmas,
                )[index],
            }

        accuracy = _lemma_accuracy(0)
        total = _lemma_accuracy(1)
        return accuracy, total  # type: ignore

    def error_profile(self, type="tags"):
        """Return an error profile with counts of errors (tagger > gold)."""
        if type == "tags":
            gold_field = Fields.GoldTags
            pred_field = Fields.Tags
        else:
            gold_field = Fields.GoldLemmas
            pred_field = Fields.Lemmas
        return Counter(
            f"{predicted} > {gold}"
            for gold_tags, predicted_tags in zip(
                self.predictions.get_field(gold_field),
                self.predictions.get_field(pred_field),
            )
            for gold, predicted in zip(gold_tags, predicted_tags)
            if gold != predicted
        )

    def lemma_tag_confusion_matrix(self):
        """Count the number of errors made when tag is right and lemma is wrong..."""
        def error_name(gold_tag, tag, gold_lemma, lemma):
            if gold_tag == tag and gold_lemma == lemma:
                return "Both right"
            elif gold_tag != tag and gold_lemma == lemma:
                return "Tag wrong, lemma right"
            elif gold_tag == tag and gold_lemma != lemma:
                return "Tag right, lemma wrong"
            else:  # Both wrong
                return "Both wrong"

        confusion = Counter(
            error_name(gold_tag, tag, gold_lemma, lemma)
            for gold_tags, tags, gold_lemmas, lemmas in zip(
                self.predictions.get_field(Fields.GoldTags),
                self.predictions.get_field(Fields.Tags),
                self.predictions.get_field(Fields.GoldLemmas),
                self.predictions.get_field(Fields.Lemmas),
            )
            for gold_tag, tag, gold_lemma, lemma in zip(
                gold_tags, tags, gold_lemmas, lemmas
            )
        )
        return confusion


def get_average(
    accuracies: List[Dict[str, Union[float, int]]]
) -> Dict[str, Tuple[float, float]]:
    """Get the average (accuracy, std_dev) and (total, std_dev) of a list of accuracies."""
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


# accuracy, std_dev
Statistics = Tuple[float, float]
# "known-wemb" -> statistic
Measures = Dict[str, Statistics]


def all_accuracy_average(
    experiments: List[Experiment],
    type: str = "tags",
) -> Tuple[Measures, Measures]:
    """Return the average of all accuracies."""
    all_tag_accuracies = []
    all_totals = []
    for experiment in experiments:
        if type == "tags":
            accuracies, totals = experiment.tagging_accuracy()
        else:
            accuracies, totals = experiment.lemma_accuracy()
        all_tag_accuracies.append(accuracies)
        all_totals.append(totals)
    return (
        get_average(all_tag_accuracies),
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


def format_results(results: Tuple[Measures, Measures]) -> str:
    """Format the Accuracy results for pretty printing."""

    def rows(results: Tuple[Measures, Measures]) -> Iterable[str]:
        accuracies, totals = results
        keys = accuracies.keys()
        for key in keys:
            yield f"{key:<20}: {accuracies[key][0]*100:>02.2f} ±{accuracies[key][1]*100:>02.2f}, {totals[key][0]:>} ±{totals[key][1]:>}"

    return "\n".join(rows(results))


ErrorProfile = Counter


def format_profile(errors: ErrorProfile, up_to=60) -> str:
    """Format the tag error profile for pretty printing."""
    total_errors = sum(errors.values())
    formatted = "Rank\tPred > Correct\tfreq\t%\n"
    formatted += "\n".join(
        f"{index + 1}\t{key}\t{value}\t{value/total_errors*100:.2f}"
        for index, (key, value) in enumerate(errors.most_common(up_to))
    )
    return formatted
