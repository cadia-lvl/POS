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


@dataclass(init=False)
class Experiment:
    """An Experiment holds information about an experiment, i.e. predicted tags and vocabularies.
    
    train_vocab: The vocabulary which was seen during training and is also present in testing.
    """

    path: Path
    examples: Dict[str, TagExamples]
    dicts: Dict[str, data.VocabMap]
    train_vocab: Set[str]
    test_vocab: Set[str]

    def __init__(self, path: Path):
        """Initialize an experiment given a path. Read the predictions and vocabulary."""
        self.path = path
        log.info(f"Reading experiment={path}")
        self.examples = analyse_examples(
            flatten_data(data.PredictedDataset.from_file(str(path / "predictions.tsv")))
        )
        with (path / "dictionaries.pickle").open("rb") as f:
            self.dicts = pickle.load(f)
        self.test_vocab = get_vocab(self.examples)
        with (path / "known_toks.txt").open("r") as f_known:
            self.train_vocab = set(line.strip() for line in f_known).intersection(
                self.test_vocab
            )
        log.info(f"Done reading experiment={path}")

    def wemb_vocab(self):
        """Return the vocabulary in the training data."""
        if "w_map" in self.dicts:
            return set(self.dicts["w_map"].w2i.keys()).intersection(self.test_vocab)
        return set()

    def pretrained_wemb_vocab_only(self):
        """Wemb only vocab is the vocabulary of the pretrained wemb vocab in the test vocab minus all others."""
        return (
            self.wemb_vocab()
            .difference(self.train_vocab)
            .difference(self.morphlex_vocab())
        )

    def wemb_vocab_only(self):
        """Wemb only vocab is the vocabulary of the wemb vocab in the test vocab minus morphlex."""
        return self.wemb_vocab().difference(self.morphlex_vocab())

    def morphlex_vocab(self):
        """Morphlex vocab is the vocabulary of the morphlex in the test vocab."""
        if "m_map" in self.dicts:
            return set(self.dicts["m_map"].w2i.keys()).intersection(self.test_vocab)
        return set()

    def morphlex_vocab_only(self):
        """Morphlex only vocab is the vocabulary of the morphlex vocab in the test vocab minus all others."""
        return (
            self.morphlex_vocab()
            .difference(self.train_vocab)
            .difference(self.wemb_vocab())
        )

    def unseen_vocab(self):
        """Unseen vocab is the vocabulary of the test vocab minus everything."""
        return self.unk_vocab().difference(self.both_vocab())

    def both_vocab(self):
        """Return the vocabulary in the morphological lexicon and wemb."""
        return self.wemb_vocab().union(self.morphlex_vocab())

    def unk_vocab(self):
        """Unk vocabulary is the test vocabulary minus training."""
        return self.test_vocab.difference(self.train_vocab)

    def accuracy(self, vocab: Set[str] = None) -> Tuple[float, int]:
        """Calculate the accuracy given a vocabulary to filter on. If nothing is provided, we do not filter."""
        if vocab is None:
            return calculate_accuracy(self.examples)
        else:
            return calculate_accuracy(filter_examples(self.examples, vocab=vocab))

    def __str__(self) -> str:
        """Return nicely formatted information about the experiment."""
        return f"\
|t-V|={len(self.test_vocab)}, \
|k-V|={len(self.train_vocab)}, \
|u-V|={len(self.unk_vocab())}, \
|m-only-V|={len(self.morphlex_vocab_only())}, \
|w-only-V|={len(self.wemb_vocab_only())}, \
|p-only-V|={len(self.pretrained_wemb_vocab_only())}\n\
{format_acc_total('t-acc', *self.accuracy())}, \
{format_acc_total('u-acc', *self.accuracy(self.unk_vocab()))}, \
{format_acc_total('k-acc', *self.accuracy(self.train_vocab))}, \
{format_acc_total('m-only-acc', *self.accuracy(self.morphlex_vocab_only()))}, \
{format_acc_total('w-only-acc', *self.accuracy(self.wemb_vocab_only()))}, \
{format_acc_total('p-only-acc', *self.accuracy(self.pretrained_wemb_vocab_only()))}, \
{format_acc_total('c-only-acc', *self.accuracy(self.unseen_vocab()))}, \
"


def format_acc_total(name: str, acc: float, total: int) -> str:
    """Format accuracy and total incorrect."""
    return f"{name:<5}={acc*100:>02.2f}% / {total:>4d}"


def calculate_average_acc(
    experiments: List[Experiment], filter_str: str
) -> Tuple[float, int]:
    """Calculate the average accuracy over a collection of experiments for the most common filters."""
    if filter_str == "none":
        total = [experiment.accuracy() for experiment in experiments]
    elif filter_str == "unk":
        total = [
            experiment.accuracy(experiment.unk_vocab()) for experiment in experiments
        ]
    elif filter_str == "known":
        total = [
            experiment.accuracy(experiment.train_vocab) for experiment in experiments
        ]
    elif filter_str == "m-only":
        total = [
            experiment.accuracy(experiment.morphlex_vocab_only())
            for experiment in experiments
        ]
    elif filter_str == "w-only":
        total = [
            experiment.accuracy(experiment.wemb_vocab_only())
            for experiment in experiments
        ]
    elif filter_str == "p-only":
        total = [
            experiment.accuracy(experiment.pretrained_wemb_vocab_only())
            for experiment in experiments
        ]
    elif filter_str == "c-only":
        total = [
            experiment.accuracy(experiment.unseen_vocab()) for experiment in experiments
        ]
    else:
        raise ValueError(f"Unkown filter_str={filter_str}")
    return (
        sum(x[0] for x in total) / len(total),
        int(round(sum(x[1] for x in total) / len(total))),
    )


def report_experiments(name: str, experiments: List[Experiment]) -> str:
    """Return a nicely formatted aggregate of a collection of experiments."""
    return f"\
{name:<15}: \
{format_acc_total('tot', *calculate_average_acc(experiments, 'none'))}, \
{format_acc_total('unk', *calculate_average_acc(experiments, 'unk'))}, \
{format_acc_total('kno', *calculate_average_acc(experiments, 'known'))}, \
{format_acc_total('mor', *calculate_average_acc(experiments, 'm-only'))}, \
{format_acc_total('wmb', *calculate_average_acc(experiments, 'w-only'))}, \
{format_acc_total('pre', *calculate_average_acc(experiments, 'p-only'))}, \
{format_acc_total('chr', *calculate_average_acc(experiments, 'c-only'))}, \
"
