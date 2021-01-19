"""A collection of types and function used to evaluate the performance of a tagger."""
from dataclasses import Field
import json
from os import stat
from typing import Callable, Iterable, Optional, Tuple, List, Dict, Set, Any, Union
import logging
from collections import Counter
from pathlib import Path
from statistics import stdev

from .core import Sentences, Vocab, FieldedDataset, Fields

log = logging.getLogger(__name__)

# accuracy, std_dev
Measure = Union[int, float]
Measure_std = Tuple[Measure, float]
# "known-wemb" -> statistic
Measures = Dict[str, Measure]
Measures_std = Dict[str, Measure_std]


class ExternalVocabularies:
    """External Vocabularies read external files which are static between different evaluations."""

    def __init__(self, morphlex_tokens: Vocab, pretrained_tokens: Vocab):
        """Initialize."""
        self.morphlex_tokens = morphlex_tokens
        self.pretrained_tokens = pretrained_tokens

    @staticmethod
    def from_files(morphlex_path: Path, pretrained_path: Path):
        """Read the files."""
        morphlex_tokens = Vocab.from_file(str(morphlex_path))
        pretrained_tokens = Vocab.from_file(str(pretrained_path))
        return ExternalVocabularies(morphlex_tokens, pretrained_tokens)


class Evaluation:
    """Evaluation of a model on a test set."""

    def __init__(self, train_vocab: Vocab, test_dataset: FieldedDataset):
        """Initialize."""
        self.train_tokens = train_vocab
        self.test_tokens = Vocab.from_symbols(test_dataset.get_field(Fields.Tokens))
        self.test_dataset = test_dataset
        self.known_tokens = self.train_tokens.intersection(self.test_tokens)
        self.unknown_tokens = self.test_tokens.difference(
            self.known_tokens
        )  # pylint: disable=no-member

    @staticmethod
    def accuracy(
        predictions: FieldedDataset,
        vocab: Optional[Set[str]],
        comparison_field: Fields,
        gold_field: Fields,
        pred_field=Fields,
    ) -> Tuple[float, int]:
        """Calculate the accuracy given a vocabulary to filter on. If nothing is provided, we do not filter.

        Args:
            predictions: The predictions along with the correct values.
            vocab: A vocabulary to whitelist elements based on.
            comparison_field: The field in the predictions to compare if it is contained in vocab.
            gold_field: The field in the predictions to consider correct.
            pred_field: The field in the predictions to consider predictions.
        """

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
                for tokens in predictions.get_field(comparison_field)
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
                    predictions.get_field(comparison_field),
                    predictions.get_field(gold_field),
                    predictions.get_field(pred_field),
                )
                for token, gold, predicted in zip(tokens, golds, predicted)
                if in_vocabulary(token, vocab)
            )
        )
        return (correct / total, total)

    @staticmethod
    def error_profile(
        predictions: FieldedDataset, gold_field: Fields, pred_field: Field
    ):
        """Return an error profile with counts of errors (pred > correct/gold)."""
        return Counter(
            f"{predicted} > {gold}"
            for gold_tags, predicted_tags in zip(
                predictions.get_field(gold_field),
                predictions.get_field(pred_field),
            )
            for gold, predicted in zip(gold_tags, predicted_tags)
            if gold != predicted
        )


class TaggingEvaluation(Evaluation):
    """Tagging evaluation of a model."""

    def __init__(
        self,
        external_vocabs: ExternalVocabularies,
        **kw,
    ):
        """Initialize."""
        super().__init__(**kw)
        morphlex_tokens = external_vocabs.morphlex_tokens.intersection(self.test_tokens)  # type: ignore
        pretrained_tokens = external_vocabs.pretrained_tokens.intersection(self.test_tokens)  # type: ignore
        # fmt: off
        self.train_pretrained_tokens = self.known_tokens.intersection(pretrained_tokens).difference(morphlex_tokens)
        self.train_pretrained_morphlex_tokens = self.known_tokens.intersection(pretrained_tokens).intersection(morphlex_tokens)
        self.train_morphlex_tokens = self.known_tokens.intersection(morphlex_tokens).difference(pretrained_tokens)
        self.train_tokens_only = self.known_tokens.difference(pretrained_tokens).difference(morphlex_tokens)
        self.test_pretrained_tokens = self.unknown_tokens.intersection(pretrained_tokens).difference(morphlex_tokens)
        self.test_pretrained_morphlex_tokens = self.unknown_tokens.intersection(pretrained_tokens).intersection(morphlex_tokens)
        self.test_morphlex_tokens = self.unknown_tokens.intersection(morphlex_tokens).difference(pretrained_tokens)
        self.test_tokens_only = self.unknown_tokens.difference(pretrained_tokens).difference(morphlex_tokens)
        # fmt: on

    def _tagging_accuracy(
        self, predictions: FieldedDataset
    ) -> Tuple[Measures, Measures]:
        """Return all tagging accuracies. Assumes that predicted tags have been set."""
        accuracy_results = {
            "Total": self.accuracy(
                predictions,
                None,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Unknown": self.accuracy(
                predictions,
                self.unknown_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Known": self.accuracy(
                predictions,
                self.known_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Known-Wemb": self.accuracy(
                predictions,
                self.train_pretrained_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Known-Wemb+Morph": self.accuracy(
                predictions,
                self.train_pretrained_morphlex_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Known-Morph": self.accuracy(
                predictions,
                self.train_morphlex_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Seen": self.accuracy(
                predictions,
                self.train_tokens_only,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Unknown-Wemb": self.accuracy(
                predictions,
                self.test_pretrained_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Unknown-Wemb+Morph": self.accuracy(
                predictions,
                self.test_pretrained_morphlex_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Unknown-Morph": self.accuracy(
                predictions,
                self.test_morphlex_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
            "Unseen": self.accuracy(
                predictions,
                self.test_tokens_only,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldTags,
                pred_field=Fields.Tags,
            ),
        }

        accuracy = {key: result[0] for key, result in accuracy_results.items()}
        total = {key: result[1] for key, result in accuracy_results.items()}
        return accuracy, total

    def tagging_accuracy(self, tags):
        """Calculate the tagging accuracy, given some tags."""
        if Fields.Tags in self.test_dataset.fields:
            raise RuntimeError(
                "Unable to evaluate predictions. Predicted tags are already present."
            )

        test_ds = self.test_dataset.add_field(tags, Fields.Tags)
        return self._tagging_accuracy(test_ds)

    def tagging_profile(self, predictions: FieldedDataset):
        """Error profile for tagging."""
        return self.error_profile(
            predictions, gold_field=Fields.GoldTags, pred_field=Fields.Tags
        )


class LemmatizationEvaluation(Evaluation):
    """Lemmatization evaluation of a model."""

    def __init__(
        self,
        train_lemmas: Vocab,
        **kw,
    ):
        """Initialize."""
        super().__init__(**kw)
        self.train_lemmas = train_lemmas
        self.test_lemmas = Vocab.from_symbols(
            self.test_dataset.get_field(Fields.GoldLemmas)
        )
        self.known_lemmas = self.train_lemmas.intersection(self.test_lemmas)
        self.unknown_lemmas = self.test_lemmas.difference(
            self.known_lemmas
        )  # pylint: disable=no-member

    def _lemma_accuracy(self, predictions: FieldedDataset) -> Tuple[Measures, Measures]:
        """Return all lemma accuracies."""
        accuracy_results = {
            "Total": self.accuracy(
                predictions=predictions,
                vocab=None,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldLemmas,
                pred_field=Fields.Lemmas,
            ),
            "Unknown": self.accuracy(
                predictions,
                self.unknown_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldLemmas,
                pred_field=Fields.Lemmas,
            ),
            "Known": self.accuracy(
                predictions,
                self.known_tokens,
                comparison_field=Fields.Tokens,
                gold_field=Fields.GoldLemmas,
                pred_field=Fields.Lemmas,
            ),
            "Unknown-Lemmas": self.accuracy(
                predictions,
                self.unknown_lemmas,
                comparison_field=Fields.GoldLemmas,
                gold_field=Fields.GoldLemmas,
                pred_field=Fields.Lemmas,
            ),
            "Known-Lemmas": self.accuracy(
                predictions,
                self.known_lemmas,
                comparison_field=Fields.GoldLemmas,
                gold_field=Fields.GoldLemmas,
                pred_field=Fields.Lemmas,
            ),
        }

        accuracy = {key: result[0] for key, result in accuracy_results.items()}
        total = {key: result[1] for key, result in accuracy_results.items()}
        return accuracy, total

    def lemma_accuracy(self, lemmas: Sentences):
        """Calculate the lemmatization accuracy, given some lemmas."""
        if Fields.Lemmas in self.test_dataset.fields:
            raise RuntimeError(
                "Unable to evaluate predictions. Predicted lemmas are already present."
            )

        test_ds = self.test_dataset.add_field(lemmas, Fields.Lemmas)
        return self._lemma_accuracy(test_ds)

    def lemma_profile(self, predictions: FieldedDataset):
        """Error profile for lemmatization."""
        return self.error_profile(
            predictions, gold_field=Fields.GoldLemmas, pred_field=Fields.Lemmas
        )


class TaggingLemmatizationEvaluation(TaggingEvaluation, LemmatizationEvaluation):
    """Lemmatization and tagging evaluation of model."""

    def __init__(
        self,
        test_dataset: FieldedDataset,
        train_vocab: Vocab,
        external_vocabs: ExternalVocabularies,
        train_lemmas: Vocab,
    ):
        """Initialize it."""
        super().__init__(
            test_dataset=test_dataset,
            train_vocab=train_vocab,
            external_vocabs=external_vocabs,
            train_lemmas=train_lemmas,
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
                self.test_dataset.get_field(Fields.GoldTags),
                self.test_dataset.get_field(Fields.Tags),
                self.test_dataset.get_field(Fields.GoldLemmas),
                self.test_dataset.get_field(Fields.Lemmas),
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


def all_accuracy_average(
    experiments: List[
        Union[
            TaggingEvaluation, LemmatizationEvaluation, TaggingLemmatizationEvaluation
        ]
    ],
    type: str = "tags",
) -> Tuple[Measures, Measures]:
    """Return the average of all accuracies."""
    all_tag_accuracies = []
    all_totals = []
    for experiment in experiments:
        if type == "tags":
            accuracies, totals = experiment._tagging_accuracy()
        else:
            accuracies, totals = experiment._lemma_accuracy()
        all_tag_accuracies.append(accuracies)
        all_totals.append(totals)
    return (
        get_average(all_tag_accuracies),
        get_average(
            all_totals,
        ),
    )


def collect_evaluation(
    predictions: Path,
    fields: Tuple[Fields, ...],
):
    """Collect the necessary files for an evaluation."""
    ds = FieldedDataset.from_file(str(predictions), fields=fields)


# def collect_experiments(
#     directory: str, morphlex_vocab: str, pretrained_vocab: str
# ) -> List[Experiment]:
#     """Collect model predictions in the directory. If the directory contains other directories, it will recurse into it."""
#     experiments: List[Experiment] = []
#     root = Path(directory)
#     directories = [d for d in root.iterdir() if d.is_dir()]
#     if directories:
#         experiments.extend(
#             [
#                 experiment
#                 for d in directories
#                 for experiment in collect_experiments(
#                     str(d),
#                     morphlex_vocab=morphlex_vocab,
#                     pretrained_vocab=pretrained_vocab,
#                 )
#             ]
#         )
#         return experiments
#     # No directories found
#     else:
#         return [
#             Experiment.from_file(root, Path(morphlex_vocab), Path(pretrained_vocab))
#         ]


def format_result(results: Tuple[Measures, Measures]) -> str:
    """Format the Accuracy results for pretty printing."""

    def rows(results: Tuple[Measures, Measures]) -> Iterable[str]:
        accuracies, totals = results
        keys = accuracies.keys()
        for key in keys:
            yield f"{key:<20}: {accuracies[key]*100:>02.2f}, {totals[key]:>}"

    return "\n".join(rows(results))


def format_results(results: Tuple[Measures_std, Measures_std]) -> str:
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
