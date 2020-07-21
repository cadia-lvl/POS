"""A collection of types and function used to evaluate the performance of a tagger."""
from collections import Counter
from typing import Tuple, List, Dict, Set
import logging

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


def calculate_accuracy(examples: Dict[str, TagExamples]) -> float:
    """Calculate the accuracy of the TagExamples."""
    total = 0
    incorrect = 0
    for gold, tag_example in examples.items():
        total += len(tag_example)
        incorrect += tag_example.incorrect_count()
    return (total - incorrect) / total


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
