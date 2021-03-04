"""Utilities."""
from typing import List, Tuple, Sequence, Iterable, Optional
import logging

log = logging.getLogger(__name__)


def tokens_to_sentences(tsv_lines: Iterable[Optional[Tuple[str, ...]]]) -> Iterable[Tuple[Tuple[str, ...], ...]]:
    """Accept a sequence of tuples (token, tag, ...) and returns a sequence of sentences (tokens, tags).

    An end of sentence is marked with a None element.
    """

    def pack_sentence(example_list: List[Tuple[str, ...]]) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(column_values) for column_values in zip(*example_list))

    example: List[Tuple[str, ...]] = []
    for line in tsv_lines:
        if line is None:
            if len(example) != 0:
                packed = pack_sentence(example)
                yield packed
                example.clear()
            # Otherwise pass silently
        else:
            example.append(line)


def read_tsv(f, sep="\t") -> Iterable[Optional[Tuple[str, ...]]]:
    """Read a .tsv file and return a tuple based on each line. Empty lines are None.

    None is appended at then end if the last line in the file is not empty.
    """
    empty_sent = True
    for line in f:
        line = line.strip()
        if not line:
            empty_sent = True
            yield None
        else:
            empty_sent = False
            yield line.split(sep)
    if not empty_sent:
        yield None
        log.info("No newline at end of file, handling it.")


def sentences_to_tokens(sentences: Iterable[Tuple[Sequence[str], ...]]) -> Iterable[Optional[Tuple[str, ...]]]:
    """Convert sentences to tuples of tokens/tags."""
    for sentence in sentences:
        yield tuple(zip(*sentence))  # type: ignore
        yield None


def write_tsv(f, data: Iterable[Tuple[str, ...]]):
    """Write a tsv in many columns."""
    for line in data:
        f.write("\t".join(line) + "\n")
    f.write("\n")
