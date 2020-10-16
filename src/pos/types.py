"""Store constants used in different parts of the project."""
from enum import Enum
from typing import Tuple, Set, Iterable, List, Dict, Optional
import logging

log = logging.getLogger(__name__)


class w_emb(Enum):
    """The supported word embeddings."""

    NONE = "none"
    STANDARD = "standard"
    PRETRAINED = "pretrained"
    ELECTRA = "electra"


# Some types

Symbol = str
Symbols = Tuple[Symbol, ...]


class Vocab(set):
    """A Vocab is an unordered set of symbols."""

    @staticmethod
    def from_symbols(sentences: Iterable[Symbols]):
        """Create a Vocab from a sequence of Symbols."""
        return Vocab((tok for sent in sentences for tok in sent))

    @staticmethod
    def from_file(filepath):
        """Create a Vocab from a file with a sequence of Symbols."""
        with open(filepath) as f:
            return Vocab(
                (symbol for line in f.readlines() for symbol in line.strip().split())
            )


class SimpleDataset(tuple):
    """A SimpleDataset is a sequence of Symbols: tokens or tags."""

    @staticmethod
    def from_file(filepath):
        """Construct a Dataset from a file."""
        with open(filepath) as f:
            sentences = read_tsv(f, cols=1)
        return SimpleDataset(sentences)


class TaggedSentence(tuple):
    """A TaggedSentence is pair of tokens and tags."""


class Dataset(tuple):
    """A Dataset is a sequence of tagged sentences: ( (tokens, tags), (tokens, tags), )."""

    @staticmethod
    def from_file(filepath):
        """Construct a Dataset from a file."""
        with open(filepath) as f:
            sentences = read_tsv(f, cols=2)
        return Dataset(sentences)

    def unpack(self) -> Tuple[SimpleDataset, SimpleDataset]:
        """Unpack a Dataset to two SimpleDataset(s): Tokens and tags."""
        tokens = SimpleDataset(
            tokens for tokens, _ in self  # pylint: disable=not-an-iterable
        )
        tags = SimpleDataset(
            tags for _, tags in self  # pylint: disable=not-an-iterable
        )
        return (tokens, tags)


class PredictedSentence(Tuple[Symbols, Symbols, Symbols]):
    """A PredictedSentence is a tuple of tokens, gold-tags and predicted-tags."""


class PredictedDataset(Tuple[PredictedSentence, ...]):
    """A PredictedDataset is sequence of PredictedSentences."""

    def unpack(self) -> Tuple[SimpleDataset, SimpleDataset, SimpleDataset]:
        """Unpack a PredictedDataset to three SimpleDataset(s): Tokens, tags and predicted tags."""
        tokens = SimpleDataset(
            tokens for tokens, _, _ in self  # pylint: disable=not-an-iterable
        )
        tags = SimpleDataset(
            tags for _, tags, _ in self  # pylint: disable=not-an-iterable
        )
        pred_tags = SimpleDataset(
            pred_tags for _, _, pred_tags in self  # pylint: disable=not-an-iterable
        )
        return (tokens, tags, pred_tags)

    def as_sequence(self) -> Iterable[Tuple[str, str, str, int, int]]:
        """Represent the PredictedDataset as a sequence of predictions, along with sentence and word index (0-based)."""
        for sent_index, sentence in enumerate(self):
            for word_index, symbols in enumerate(zip(*sentence)):
                yield symbols[0], symbols[1], symbols[2], sent_index, word_index

    @staticmethod
    def from_file(filepath):
        """Construct a PredictedDataset from a file."""
        with open(filepath) as f:
            sentences = read_tsv(f, cols=3)
        return PredictedDataset(sentences)


def write_tsv(f, data: Tuple[SimpleDataset, ...]):
    """Write a tsv in many columns."""
    for sentence in zip(*data):
        for tok_tags in zip(*sentence):
            f.write("\t".join(tok_tags) + "\n")
        f.write("\n")


def read_tsv(f, cols=2) -> List[Tuple[Symbols, ...]]:
    """Read a single .tsv file with one, two or three columns and returns a list of the Symbols."""

    def add_sentence(
        sent_tokens: List[str], sent_tags: List[str], model_tags: List[str]
    ):
        """Add a sentence to the list. HAS SIDE-EFFECTS."""
        if cols == 1:
            sentences.append((tuple(sent_tokens),))
        elif cols == 2:
            sentences.append((tuple(sent_tokens), tuple(sent_tags)))
        elif cols == 3:
            sentences.append((tuple(sent_tokens), tuple(sent_tags), tuple(model_tags),))
        else:
            raise ValueError(f"Invalid number of cols={cols}")

    sentences: List[Tuple[Symbols, ...]] = []
    sent_tokens: List[str] = []
    sent_tags: List[str] = []
    model_tags: List[str] = []
    for line in f:
        line = line.strip()
        # We read a blank line and buffer is not empty - sentence has been read.
        if not line and len(sent_tokens) != 0:
            add_sentence(sent_tokens, sent_tags, model_tags)
            sent_tokens = []
            sent_tags = []
            model_tags = []
        else:
            symbols = line.split()
            if cols >= 1:
                sent_tokens.append(symbols[0])
            if cols >= 2:
                sent_tags.append(symbols[1])
            if cols == 3:
                model_tags.append(symbols[2])
    # For the last sentence
    if len(sent_tokens) != 0:
        log.info("No newline at end of file, handling it.")
        add_sentence(sent_tokens, sent_tags, model_tags)
    return sentences


class VocabMap:
    """A VocabMap stores w2i and i2w for dictionaries."""

    w2i: Dict[Symbol, int]
    i2w: Dict[int, Symbol]

    def __init__(
        self, vocab: Vocab, special_tokens: Optional[List[Tuple[str, int]]] = None
    ):
        """Build a vocabulary mapping from the provided vocabulary, needs to start at index=0.

        If special_tokens is given, will add these tokens first and start from the next index of the highest index provided.
        """
        self.w2i = {}
        next_idx = 0
        if special_tokens:
            for symbol, idx in special_tokens:
                self.w2i[symbol] = idx
                next_idx = max((idx + 1, next_idx))
        for idx, symbol in enumerate(vocab, start=next_idx):
            self.w2i[symbol] = idx
        self.i2w = {i: w for w, i in self.w2i.items()}

    def __len__(self):
        """Return the length of the dictionary."""
        return len(self.w2i)
