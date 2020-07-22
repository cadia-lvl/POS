"""A collection of types and function used to evaluate the performance of a tagger."""
from collections import Counter
from typing import Tuple, List, Dict, Set
import logging
from dataclasses import dataclass
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


def flatten_data(
    in_data: Tuple[data.DataSent, data.DataSent, data.DataSent]
) -> List[Example]:
    """Flatten the predictions to a sequence of tokens along with their tags."""
    line_id = 0
    flat = []
    for sent_toks, sent_gold, sent_pred in zip(*in_data):
        line_id += 1
        for tok, gold, pred in zip(sent_toks, sent_gold, sent_pred):
            flat.append((tok, gold, pred, line_id))
            line_id += 1
    return flat


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


def calculate_accuracy(examples: Dict[str, TagExamples]) -> Tuple[int, int]:
    """Calculate the total and incorrect count of the TagExamples."""
    total = 0
    incorrect = 0
    for gold, tag_example in examples.items():
        total += len(tag_example)
        incorrect += tag_example.incorrect_count()
    return (incorrect, total)


def all_errors(examples: Dict[str, TagExamples]) -> Counter:
    """Return a Counter of all the errors."""
    error_counter: Counter = Counter()
    for gold, tag_example in examples.items():
        error_counter.update(tag_example.errors())
    return error_counter


def get_vocab(examples: Dict[str, TagExamples]) -> Set[str]:
    """Return the vocabulary of the TagExamples."""
    vocab = set()
    for tag_example in examples.values():
        for example in tag_example.examples:
            vocab.add(example[0])
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


@dataclass(init=False)
class Experiment:
    """A class to hold information about an experiment, i.e. predicted tags."""

    path: Path
    examples: Dict[str, TagExamples]
    dicts: Dict[str, data.VocabMap]
    train_vocab: Set[str]

    def __init__(self, path: Path):
        """Initialize an experiment given a path. Read the predictions and vocabulary."""
        self.path = path
        self.examples = analyse_examples(
            flatten_data(data.read_tsv(str(path / "predictions.tsv")))
        )
        with (path / "dictionaries.pickle").open("rb") as f:
            self.dicts = pickle.load(f)
        with (path / "known_toks.txt").open("r") as f_known:
            self.train_vocab = set(line.strip() for line in f_known)

    def test_vocab(self):
        """Return the vocabulary of the test data."""
        return get_vocab(self.examples)

    def wemb_vocab(self):
        """Return the vocabulary in the training data."""
        return set(self.dicts["w_map"].w2i.keys())

    def morphlex_vocab(self):
        """Return the vocabulary in the morphological lexicon."""
        return set(self.dicts["m_map"].w2i.keys())

    def both_vocab(self):
        """Return the vocabulary in the morphological lexicon and wemb."""
        return self.wemb_vocab().union(self.morphlex_vocab())

    def unk_vocab(self):
        """Return the vocabulary in the test data minus the morphological lexicon and training."""
        return self.test_vocab().difference(self.train_vocab)

    def accuracy(self, vocab: Set[str] = None):
        """Calculate the accuracy given a vocabulary to filter on. If nothing is provided, we do not filter."""
        if vocab is None:
            return calculate_accuracy(self.examples)
        else:
            return calculate_accuracy(filter_examples(self.examples, vocab=vocab))

    def __str__(self) -> str:
        """Return nicely formatted information about the experiment."""
        total_incorrect, total_total = self.accuracy()
        unk_incorrect, unk_total = self.accuracy(self.unk_vocab())
        known_incorrect, known_total = self.accuracy(self.train_vocab)
        return f"{format_acc_total('total', total_total, total_incorrect)}, \
                {format_acc_total('unk', unk_total, unk_incorrect)}, \
                {format_acc_total('known', known_total, known_incorrect)}, \
                "


def format_acc_total(name: str, total: float, incorrect: float) -> str:
    """Format accuracy and total incorrect."""
    return f"{name}={((total - incorrect)/total)*100:2.3f}% / {incorrect:.0f}"


def calculate_average_acc(
    experiments: List[Experiment], filter: str
) -> Tuple[float, float]:
    """Calculate the average accuracy over a collection of experiments for the most common filters."""
    if filter == "none":
        total = [experiment.accuracy() for experiment in experiments]
    elif filter == "unk":
        total = [
            experiment.accuracy(experiment.unk_vocab()) for experiment in experiments
        ]
    elif filter == "known":
        total = [
            experiment.accuracy(experiment.train_vocab) for experiment in experiments
        ]
    return sum(x[0] for x in total) / len(total), sum(x[1] for x in total) / len(total)


def report_experiments(experiments: List[Experiment]) -> str:
    """Return a nicely formatted aggregate of a collection of experiments."""
    total_incorrect, total_total = calculate_average_acc(experiments, "none")
    unk_incorrect, unk_total = calculate_average_acc(experiments, "unk")
    known_incorrect, known_total = calculate_average_acc(experiments, "known")
    return f"{format_acc_total('total', total_total, total_incorrect)}, \
            {format_acc_total('unk', unk_total, unk_incorrect)}, \
            {format_acc_total('known', known_total, known_incorrect)}, \
            "
