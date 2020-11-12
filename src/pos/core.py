"""The main abstractions in the project."""
from enum import Enum
from typing import Tuple, Set, Iterable, List, Dict, Optional, Sequence, cast
import logging

from torch.utils.data import Dataset

from .utils import read_tsv, tokens_to_sentences

log = logging.getLogger(__name__)

Tokens = Sequence[str]
Tags = Sequence[str]


class Modules(Enum):
    """An enum to name all model parts."""

    BERT = "bert"
    CharsAsWord = "c_map"
    Pretrained = "p_map"
    FullTag = "t_map"
    WordEmbeddings = "w_map"
    MorphLex = "m_map"
    Lengths = "lens"


class SequenceTaggingDataset(Dataset):
    """A dataset to hold pairs of tokens and tags."""

    def __init__(
        self, examples: Sequence[Tuple[Tokens, Tags]],
    ):
        """Initialize a dataset given a sequence of examples."""
        self.examples = examples

    def __getitem__(self, idx):
        """Support itemgetter."""
        return self.examples[idx]

    def __len__(self):
        """Support len."""
        return len(self.examples)

    def __iter__(self):
        """Support iteration."""
        return iter(self.examples)

    def __add__(self, other):
        """Support addition."""
        return SequenceTaggingDataset(self.examples + other.examples)

    @staticmethod
    def from_file(filepath: str,):
        """Initialize a dataset given a filepath."""
        with open(filepath) as f:
            examples = tuple(tokens_to_sentences(read_tsv(f)))
        if len(examples) != 0:
            # We expect to get List[Tokens, Tags]
            assert len(examples[0]) == 2
        examples = cast(Tuple[Tuple[Sequence[str], Sequence[str]]], examples)
        return SequenceTaggingDataset(examples)

    def unpack(self) -> Tuple[Sequence[Tokens], Sequence[Tags]]:
        """Unpack to Tokens and tags."""
        return (tuple(tokens for tokens, _ in self), tuple(tags for _, tags in self))


class TokenizedDataset(Dataset):
    """A dataset to hold tokenized text."""

    def __init__(
        self, examples: Sequence[Tokens],
    ):
        """Initialize a dataset given a sequence of examples."""
        self.examples = examples

    def __getitem__(self, idx):
        """Support itemgetter."""
        return self.examples[idx]

    def __len__(self):
        """Support len."""
        return len(self.examples)

    def __iter__(self):
        """Support iteration."""
        return iter(self.examples)

    @staticmethod
    def from_file(filepath: str,):
        """Initialize a dataset given a filepath."""
        with open(filepath) as f:
            examples = tuple(tokens_to_sentences(read_tsv(f)))
        if len(examples) != 0:
            # We expect to get List[Tokens, Tags]
            assert len(examples[0]) == 1
        return TokenizedDataset([example[0] for example in examples])


class DoubleTaggedDataset(Dataset):
    """A PredictedDataset is sequence of PredictedSentences."""

    def __init__(
        self, examples: Sequence[Tuple[Tokens, Tags, Tags]],
    ):
        """Initialize the Dataset."""
        self.examples = examples

    def __getitem__(self, idx):
        """Support itemgetter."""
        return self.examples[idx]

    def __len__(self):
        """Support len."""
        return len(self.examples)

    def __iter__(self):
        """Support iteration."""
        return iter(self.examples)

    def unpack(self) -> Tuple[Sequence[Tokens], Sequence[Tokens], Sequence[Tokens]]:
        """Unpack a PredictedDataset to three SimpleDataset(s): Tokens, tags and predicted tags."""
        return (
            tuple(tokens for tokens, _, _ in self),
            tuple(tags for _, tags, _ in self),
            tuple(preds for _, _, preds in self),
        )

    def as_sequence(self) -> Iterable[Tuple[str, str, str, int, int]]:
        """Represent the PredictedDataset as a sequence of predictions, along with sentence and word index (0-based)."""
        for sent_index, sentence in enumerate(self):
            for word_index, symbols in enumerate(zip(*sentence)):
                yield symbols[0], symbols[1], symbols[2], sent_index, word_index

    @staticmethod
    def from_file(filepath):
        """Construct a PredictedDataset from a file."""
        with open(filepath) as f:
            examples = tuple(tokens_to_sentences(read_tsv(f)))
        return DoubleTaggedDataset(examples)


class Vocab(set):
    """A Vocab is an unordered set of symbols."""

    @staticmethod
    def from_symbols(sentences: Iterable[Tokens]):
        """Create a Vocab from a sequence of Symbols."""
        return Vocab((tok for sent in sentences for tok in sent))

    @staticmethod
    def from_file(filepath):
        """Create a Vocab from a file with a sequence of Symbols."""
        with open(filepath) as f:
            return Vocab(
                (symbol for line in f.readlines() for symbol in line.strip().split())
            )


class VocabMap:
    """A VocabMap stores w2i and i2w for dictionaries."""

    w2i: Dict[str, int]
    i2w: Dict[int, str]

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

