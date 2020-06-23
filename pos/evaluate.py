from collections import Counter
from typing import Tuple, List, Dict, Set
import logging

from . import data

log = logging.getLogger()

Example = Tuple[str, str, str, int]


class TagExamples:
    def __init__(self, tag: str):
        self.tag = tag
        self.examples: List[Tuple[str, str, int]] = []

    def add_example(self, token, predicted, line_id) -> None:
        self.examples.append((token, predicted, line_id))

    def __len__(self) -> int:
        return len(self.examples)

    def incorrect_count(self) -> int:
        count = 0
        for _, p, _ in self.examples:
            if self.tag != p:
                count += 1
        return count

    def errors(self) -> Counter:
        errors: Counter = Counter()
        for _, p, _ in self.examples:
            if self.tag != p:
                errors.update([f"{p} -> {self.tag}"])
        return errors


def flatten_data(
    in_data: Tuple[data.DataSent, data.DataSent, data.DataSent]
) -> List[Example]:
    line_id = 0
    flat = []
    for sent_toks, sent_gold, sent_pred in zip(*in_data):
        line_id += 1
        for tok, gold, pred in zip(sent_toks, sent_gold, sent_pred):
            flat.append((tok, gold, pred, line_id))
            line_id += 1
    return flat


def analyse_examples(flat_data: List[Example]) -> Dict[str, TagExamples]:
    examples: Dict[str, TagExamples] = {}
    for tok, gold, pred, line_id in flat_data:
        try:
            examples[gold].add_example(tok, pred, line_id)
        except KeyError:
            examples[gold] = TagExamples(gold)
            examples[gold].add_example(tok, pred, line_id)
    return examples


def calculate_accuracy(examples: Dict[str, TagExamples]) -> float:
    total = 0
    incorrect = 0
    for gold, tag_example in examples.items():
        total += len(tag_example)
        incorrect += tag_example.incorrect_count()
    return (total - incorrect) / total


def all_errors(examples: Dict[str, TagExamples]) -> Counter:
    error_counter: Counter = Counter()
    for gold, tag_example in examples.items():
        error_counter.update(tag_example.errors())
    return error_counter


def get_vocab(examples: Dict[str, TagExamples]) -> Set[str]:
    vocab = set()
    for tag_example in examples.values():
        for example in tag_example.examples:
            vocab.add(example[0])
    return vocab


def filter_examples(
    examples: Dict[str, TagExamples], vocab: Set[str]
) -> Dict[str, TagExamples]:
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
