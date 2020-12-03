"""The main abstractions in the project."""
from enum import Enum
from typing import (
    Iterator,
    Tuple,
    Iterable,
    List,
    Dict,
    Optional,
    Sequence,
    Union,
)
import logging

from torch.utils.data import Dataset

from .utils import read_tsv, tokens_to_sentences, write_tsv

log = logging.getLogger(__name__)

Sentence = Tuple[str, ...]
Sentences = Tuple[Sentence, ...]


class Dicts(Enum):
    """An enum to name all model parts."""

    Chars = "c_map"
    Pretrained = "p_map"
    FullTag = "t_map"
    Tokens = "w_map"
    MorphLex = "m_map"


class Vocab(set):
    """A Vocab is an unordered set of symbols."""

    @staticmethod
    def from_symbols(sentences: Iterable[Union[Sentence, str]]):
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


class Fields:
    """Common fields used."""

    Tokens = "tokens"
    Tags = "tags"
    Lemmas = "lemmas"
    GoldTags = "gold_tags"
    GoldLemmas = "gold_lemmas"


class FieldedDataset(Dataset):
    """A generic dataset built from group tsv lines."""

    def __init__(self, data: Tuple[Sentences, ...], fields: Tuple[str, ...]):
        """Initialize the dataset."""
        self.data: Tuple[Sentences, ...] = data
        self.fields: Tuple[str, ...] = fields

    def __getitem__(self, idx) -> Tuple[Sentence, ...]:
        """Support itemgetter."""
        return tuple(data_field[idx] for data_field in self.data)

    def __len__(self) -> int:
        """Support len."""
        return len(self.data[0])

    def __iter__(self) -> Iterator[Tuple[Sentence, ...]]:
        """Support iteration."""
        return zip(*self.data)

    def __add__(self, other):
        """Support addition."""
        assert self.fields == other.fields
        return self.__class__(self.data + other.data, self.fields)

    def get_field(self, field=Fields.Tokens) -> Sentences:
        """Get the field."""
        return self.data[self.fields.index(field)]

    def get_vocab(self, field=Fields.Tokens) -> Vocab:
        """Return the Vocabulary in the dataset."""
        return Vocab.from_symbols(self.get_field(field))

    def get_vocab_map(self, special_tokens=None, field=Fields.Tokens) -> VocabMap:
        """Return the VocabularyMapping in the dataset."""
        return VocabMap(self.get_vocab(field), special_tokens=special_tokens)

    def get_char_vocab(self, field=Fields.Tokens) -> Vocab:
        """Return the character Vocabulary in the dataset."""
        return Vocab.from_symbols(
            (tok for sent in self.get_field(field) for tok in sent)
        )

    def get_char_vocab_map(self, special_tokens=None, field=Fields.Tokens) -> VocabMap:
        """Return the character VocabularyMapping in the dataset."""
        return VocabMap(self.get_char_vocab(field), special_tokens=special_tokens)

    def get_tag_vocab_map(self, special_tokens=None, field=Fields.GoldTags) -> VocabMap:
        """Return the VocabularyMapping in the dataset."""
        return VocabMap(self.get_vocab(field), special_tokens=special_tokens)

    def add_field(self, data_field: Sequence[Sentence], field: str):
        """Return a new FieldDataset which has an added data_field."""
        return FieldedDataset(self.data + (data_field,), self.fields + (field,))

    def _iter_for_tsv(self):
        """Iterate for TSV which includes empty lines between sentences."""
        yield_empty = False
        for field_sentences in self:
            if yield_empty:
                # Yield an empty tuple for empty lines between sentences.
                yield tuple()
            for fields in zip(*field_sentences):
                yield fields
                yield_empty = True

    def to_tsv_file(self, path: str):
        """Write the dataset to a file as TSV."""
        with open(path, mode="w") as f:
            write_tsv(f, self._iter_for_tsv())

    @staticmethod
    def from_file(filepath: str, fields: Tuple[str, ...] = None):
        """Construct from a file. By default we assume first there are Tokens, GoldTags, GoldLemmas."""
        with open(filepath) as f:
            examples = tuple(zip(*tuple(tokens_to_sentences(read_tsv(f)))))
        if not fields:
            fields = tuple()
            if len(examples) >= 1:
                fields = fields + (Fields.Tokens,)
            if len(examples) >= 2:
                fields = fields + (Fields.GoldTags,)
            if len(examples) >= 3:
                fields = fields + (Fields.GoldLemmas,)
        return FieldedDataset(examples, fields=fields)
