"""A collection of types and function used to evaluate the performance of a tagger."""
from collections import Counter
from typing import Tuple, List, Dict, Set
import logging
from pathlib import Path
import pickle

from . import data

log = logging.getLogger()

Example = Tuple[str, str, str, int]


class TagExamples:
    """A class to represent tags and their occurrances."""

    def __init__(self, tag: str):
        """Initialize the object."""
        self.tag = tag
        self.examples: List[Tuple[str, str, int]] = []

    def add_example(self, token, predicted, line_id) -> None:
        """Add an example of tag usage. A usage is when it is the correct tag."""
        self.examples.append((token, predicted, line_id))

    def __len__(self) -> int:
        """Return the number of examples of this tag."""
        return len(self.examples)

    def incorrect_count(self) -> int:
        """Return the number of occurance a tag is incorrectly predicted."""
        count = 0
        for _, p, _ in self.examples:
            if self.tag != p:
                count += 1
        return count

    def errors(self) -> Counter:
        """Return a Counter for the types of errors for a given tag."""
        errors: Counter = Counter()
        for _, p, _ in self.examples:
            if self.tag != p:
                errors.update([f"{p} -> {self.tag}"])
        return errors


def flatten_data(in_data: data.PredictedDataset) -> List[Example]:
    """Flatten the predictions to a sequence of tokens along with their tags."""
    return [
        (tok, gold, pred, sent_idx)
        for tok, gold, pred, sent_idx, _ in in_data.as_sequence()
    ]


def analyse_examples(flat_data: List[Example]) -> Dict[str, TagExamples]:
    """Run TagExamples on each token and tag."""
    examples: Dict[str, TagExamples] = {}
    for tok, gold, pred, line_id in flat_data:
        try:
            examples[gold].add_example(tok, pred, line_id)
        except KeyError:
            examples[gold] = TagExamples(gold)
            examples[gold].add_example(tok, pred, line_id)
    return examples


def calculate_accuracy(examples: Dict[str, TagExamples]) -> Tuple[float, int]:
    """Calculate the total and incorrect count of the TagExamples."""
    total = 0
    incorrect = 0
    for gold, tag_example in examples.items():
        total += len(tag_example)
        incorrect += tag_example.incorrect_count()
    if total == 0:
        return (0.0, 0)
    return ((total - incorrect) / total, total)


def all_errors(examples: Dict[str, TagExamples]) -> Counter:
    """Return a Counter of all the errors."""
    error_counter: Counter = Counter()
    for gold, tag_example in examples.items():
        error_counter.update(tag_example.errors())
    return error_counter


def get_vocab(examples: Dict[str, TagExamples]) -> Set[str]:
    """Return the vocabulary of the TagExamples."""
    vocab: Set[str] = set()
    for tag_example in examples.values():
        vocab.update(set(example[0] for example in tag_example.examples))
    return vocab


def filter_examples(
    examples: Dict[str, TagExamples], vocab: Set[str]
) -> Dict[str, TagExamples]:
    """Filter the TagExamples based on a vocabulary."""
    filtered: Dict[str, TagExamples] = {}
    for tag, tag_example in examples.items():
        for example in tag_example.examples:
            if example[0] in vocab:
                try:
                    filtered[tag].add_example(*example)
                except KeyError:
                    filtered[tag] = TagExamples(tag)
                    filtered[tag].add_example(*example)
    return filtered


class Experiment:
    """An Experiment holds information about an experiment, i.e. predicted tags and vocabularies."""

    def __init__(
        self,
        predictions: data.PredictedDataset,
        train_vocab: data.Vocab,
        dicts: Dict[str, data.VocabMap],
    ):
        """Initialize an experiment given the predictions and vocabulary."""
        self.examples = analyse_examples(flatten_data(predictions))
        self.test_vocab = get_vocab(self.examples)
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
        self.unknown_vocab = self.test_vocab.difference(self.known_vocab)
        # fmt: off
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
        """Create an Experiment from a given path of an experimental results."""
        log.info(f"Reading experiment={path}")
        log.info("Reading predictions")
        predictions = data.PredictedDataset.from_file(str(path / "predictions.tsv"))
        log.info("Reading dicts")
        with (path / "dictionaries.pickle").open("rb") as f:
            dicts = pickle.load(f)
        log.info("Reading training vocab")
        train_vocab = data.Vocab.from_file(path / "known_toks.txt")
        log.info(f"Done reading experiment={path}")
        return Experiment(predictions=predictions, train_vocab=train_vocab, dicts=dicts)

    def accuracy(self, vocab: Set[str] = None) -> Tuple[float, int]:
        """Calculate the accuracy given a vocabulary to filter on. If nothing is provided, we do not filter."""
        if vocab is None:
            return calculate_accuracy(self.examples)
        else:
            return calculate_accuracy(filter_examples(self.examples, vocab=vocab))

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
