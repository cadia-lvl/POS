"""The main abstractions in the project."""
import logging
import random
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch import device as t_device
from torch import set_num_threads
from torch.utils.data import Dataset

from pos.constants import BATCH_KEYS
from pos.utils import read_tsv, tokens_to_sentences, write_tsv

log = logging.getLogger(__name__)

Sentence = Tuple[str, ...]
Sentences = Tuple[Sentence, ...]

device = None


def set_device(gpu_flag=False):
    """Set the torch device."""
    if gpu_flag and torch.cuda.is_available():
        device_set = t_device("cuda")  # type: ignore
        # Torch will use the allocated GPUs from environment variable CUDA_VISIBLE_DEVICES
        log.info(f"Using {torch.cuda.device_count()} GPUs {torch.cuda.get_device_name()}")
    else:
        device_set = t_device("cpu")  # type: ignore
        threads = 1
        # Set the number of threads to use for CPU
        set_num_threads(threads)
        log.info(f"Using {threads} CPU threads")
    global device
    device = device_set


def set_seed(seed=42):
    """Set the seed on all platforms. 0 for no specific seeding."""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


class Dicts(Enum):
    """An enum to name all mappings."""

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
    def from_file(filepath: str):
        """Create a Vocab from a file with a sequence of Symbols."""
        with open(filepath) as f:
            return Vocab((symbol for line in f.readlines() for symbol in line.strip().split()))


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

    def __init__(self, vocab: Set[str], special_tokens: Optional[List[Tuple[str, int]]] = None):
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
        # the data is columnar
        assert len(data) == len(
            fields
        ), f"The data {len(data)} and fields {len(fields)} do not contain the same number of elements"
        # All sequences are of the same length
        lengths = -1
        for field in fields:
            if lengths == -1:
                lengths = self._get_field_length(field)
            else:
                assert lengths == self._get_field_length(field), "All fields should be of the same size"

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
        new_data = tuple(data + other_data for data, other_data in zip(self.data, other.data))
        return self.__class__(new_data, self.fields)

    def _get_field_length(self, field: str) -> Tuple[int, ...]:
        """Return the field length."""
        return tuple(len(sentence) for sentence in self.data[self.fields.index(field)])

    def get_lengths(self) -> Tuple[int, ...]:
        """Return the sentence lengths."""
        return self._get_field_length(self.fields[0])

    def _shorten_field_length(self, field, lengths: Tuple[int, ...]) -> Sentences:
        """Shorten the field based on lengths."""
        elements = self.get_field(field)
        # lengths, x
        adjusted_sentences = [tuple() for _ in range(len(lengths))]
        index = 0
        for element in elements:
            length = lengths[index]
            # Just right
            if len(element) == length:
                adjusted_sentences[index] = element
                index += 1
            # the sentence is too long
            elif len(element) > length:
                partial_element = element
                while len(partial_element) > 0:
                    # shorten it according to the lengths until done
                    length = lengths[index]
                    part, partial_element = (
                        partial_element[:length],
                        partial_element[length:],
                    )
                    adjusted_sentences[index] = part
                    index += 1
            else:
                log.error(f"Shortening but element too short {element}, {len(element)}, {length}")
                raise ValueError("Bad lengths")
        return tuple(adjusted_sentences)

    def _lengthen_field_length(self, field, lengths: Tuple[int, ...]) -> Sentences:
        """Lengthen field length back to original."""
        elements = self.get_field(field)
        # lengths, x
        adjusted_sentences = [tuple() for _ in range(len(lengths))]
        index = 0
        elements_it = iter(elements)
        for element in elements_it:
            length = lengths[index]
            # Just right
            if len(element) == length:
                adjusted_sentences[index] = element
                index += 1

            else:
                while len(adjusted_sentences[index]) != length:
                    if len(adjusted_sentences[index]) == 0:
                        # set it
                        adjusted_sentences[index] = element
                    else:
                        # splice it
                        adjusted_sentences[index] = adjusted_sentences[index] + element
                    if len(adjusted_sentences[index]) != length:
                        element = next(elements_it)
                index += 1
        return tuple(adjusted_sentences)

    def _adjust_field_length(self, field, lengths: Tuple[int, ...], shorten=True) -> Sentences:
        if shorten:
            return self._shorten_field_length(field, lengths)
        else:
            return self._lengthen_field_length(field, lengths)

    def adjust_lengths(self, lengths: Tuple[int, ...], shorten):
        """Adjust the lengths of the dataset according to the given lengths."""
        adjusted_data = []
        for field in self.fields:
            adjusted_data.append(self._adjust_field_length(field, lengths, shorten))
        return FieldedDataset(tuple(adjusted_data), self.fields)

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
        return Vocab.from_symbols((tok for sent in self.get_field(field) for tok in sent))

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

    def collate_fn(self, batch: Sequence[Tuple[Sentence, ...]]) -> Dict[str, Any]:
        """Map the inputs to batches."""
        batch_dict = {}
        for idx, field in enumerate(self.fields):
            if field == Fields.Tokens:
                batch_dict[BATCH_KEYS.TOKENS] = tuple(element[idx] for element in batch)
            elif field == Fields.GoldTags:
                batch_dict[BATCH_KEYS.FULL_TAGS] = tuple(element[idx] for element in batch)
            elif field == Fields.GoldLemmas:
                batch_dict[BATCH_KEYS.LEMMAS] = tuple(element[idx] for element in batch)
        batch_dict[BATCH_KEYS.TOKEN_CHARS_LENS] = tuple(
            len(token) for sent in batch_dict[BATCH_KEYS.TOKENS] for token in sent
        )
        batch_dict[BATCH_KEYS.LENGTHS] = tuple(len(x) for x in batch_dict[BATCH_KEYS.TOKENS])
        return batch_dict

    @staticmethod
    def from_file(filepath: str, fields: Tuple[str, ...] = None, sep="\t"):
        """Construct from a file. By default we assume first there are Tokens, GoldTags, GoldLemmas."""
        with open(filepath) as f:
            sentences = tuple(tokens_to_sentences(read_tsv(f, sep=sep)))
            examples = tuple(zip(*sentences))
        if not fields:
            fields = tuple()
            if len(examples) >= 1:
                fields = fields + (Fields.Tokens,)
            if len(examples) >= 2:
                fields = fields + (Fields.GoldLemmas,)
            if len(examples) >= 3:
                fields = fields + (Fields.GoldTags,)
            if len(examples) >= 4:
                raise ValueError("Unable to guess fields in TSV file. Please set 'fields'")
        return FieldedDataset(examples, fields=fields)
