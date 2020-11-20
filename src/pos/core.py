"""The main abstractions in the project."""
from enum import Enum
from typing import Tuple, Set, Iterable, List, Dict, Optional, Sequence, cast, Any
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
    Lemmatizer = "lemma"


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

    # To pad in batches
    PAD = "<pad>"
    PAD_ID = 0
    # For unkown words in testing
    UNK = "<unk>"
    UNK_ID = 1
    # For EOS and SOS in char BiLSTM
    EOS = "</s>"
    EOS_ID = 2
    SOS = "<s>"
    SOS_ID = 3
    UNK_PAD = [(UNK, UNK_ID), (PAD, PAD_ID)]
    UNK_PAD_EOS_SOS = [
        (UNK, UNK_ID),
        (PAD, PAD_ID),
        (EOS, EOS_ID),
        (SOS, SOS_ID),
    ]

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


class TokenizedDataset(Dataset):
    """A dataset to hold tokenized text."""

    def __init__(
        self, examples: Sequence[Any],
    ):
        """Initialize a dataset given a sequence of examples."""
        self.examples = examples

    @staticmethod
    def from_file(filepath: str,):
        """Initialize a dataset given a filepath."""
        with open(filepath) as f:
            examples = tuple(tokens_to_sentences(read_tsv(f)))
        if len(examples) != 0:
            # We expect to get List[Tokens]
            assert len(examples[0]) == 1
        return TokenizedDataset([example[0] for example in examples])

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
        return self.__class__(self.examples + other.examples)

    def unpack(self) -> Tuple[Sequence[Tokens], ...]:
        """Unpack to Tokens and tags."""
        return (tuple(tokens for tokens in self),)

    def get_vocab(self) -> Vocab:
        """Return the Vocabulary in the dataset."""
        return Vocab.from_symbols(self.unpack()[0])

    def get_vocab_map(self, special_tokens=None) -> VocabMap:
        """Return the VocabularyMapping in the dataset."""
        return VocabMap(self.get_vocab(), special_tokens=special_tokens)

    def get_char_vocab(self) -> Vocab:
        """Return the character Vocabulary in the dataset."""
        return Vocab.from_symbols((tok for sent in self.unpack()[0] for tok in sent))

    def get_char_vocab_map(self, special_tokens=None) -> VocabMap:
        """Return the character VocabularyMapping in the dataset."""
        return VocabMap(self.get_char_vocab(), special_tokens=special_tokens)


class SequenceTaggingDataset(TokenizedDataset):
    """A dataset to hold pairs of tokens and tags."""

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

    def unpack(self) -> Tuple[Sequence[Tokens], ...]:
        """Unpack to Tokens and tags."""
        return (tuple(tokens for tokens, _ in self), tuple(tags for _, tags in self))

    def get_tag_vocab(self) -> Vocab:
        """Return the tag Vocabulary in the dataset."""
        return Vocab.from_symbols(self.unpack()[1])

    def get_tag_vocab_map(self, special_tokens) -> VocabMap:
        """Return the tag VocabularyMapping in the dataset."""
        return VocabMap(self.get_tag_vocab(), special_tokens=special_tokens)


class DoubleTaggedDataset(SequenceTaggingDataset):
    """A dataset containing two tags per token."""

    def unpack(self) -> Tuple[Sequence[Tokens], ...]:
        """Unpack to Tokens and other token tags."""
        return (
            tuple(tokens for tokens, _, _ in self),
            tuple(tags for _, tags, _ in self),
            tuple(preds for _, _, preds in self),
        )

    @staticmethod
    def from_file(filepath):
        """Construct from a file."""
        with open(filepath) as f:
            examples = tuple(tokens_to_sentences(read_tsv(f)))
        return DoubleTaggedDataset(examples)

