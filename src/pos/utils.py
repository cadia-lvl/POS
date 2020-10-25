"""Utilities."""
from typing import List, Tuple, Sequence
import logging

log = logging.getLogger(__name__)


def read_tsv(f) -> Sequence[Tuple[Sequence[str], ...]]:
    """Read a .tsv file which defines sequences with empty lines and return a  list of the Symbols."""
    examples: List[Tuple[List[str], ...]] = []
    example: List[List[str]] = []

    def map_example(example: List[List[str]]) -> Tuple[List[str], ...]:
        return tuple([list(field) for field in zip(*example)])

    for line in f:
        line = line.strip()
        # We read a blank line and buffer is not empty - sentence has been read.
        if not line and len(example) != 0:
            examples.append(map_example(example))
            example.clear()
        else:
            example.append(line.split())
    # For the last sentence
    if len(example) != 0:
        log.info("No newline at end of file, handling it.")
        examples.append(map_example(example))
    return examples


def write_tsv(f, data: Tuple[Sequence[Sequence[str]], ...]):
    """Write a tsv in many columns."""
    for sentence in zip(*data):
        for tok_tags in zip(*sentence):
            f.write("\t".join(tok_tags) + "\n")
        f.write("\n")
