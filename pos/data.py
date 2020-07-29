"""Data preparation and reading."""
from dataclasses import dataclass
from typing import (
    List,
    Tuple,
    Set,
    Dict,
    Optional,
    Union,
    Iterable,
    cast,
    Sequence,
    Callable,
)
import logging

from tqdm import tqdm
import numpy as np
import torch
import random

log = logging.getLogger()

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

# Some types

Symbol = str


class Symbols(Tuple[Symbol, ...]):
    """A Symbol is a sequence of symbols in a sentence: tokens or tags."""


class Vocab(Set[str]):
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


class SimpleDataset(Tuple[Symbols, ...]):
    """A SimpleDataset is a sequence of Symbols: tokens or tags."""


class TaggedSentence(Tuple[Symbols, Symbols]):
    """A TaggedSentence is pair of tokens and tags."""


class Dataset(Tuple[TaggedSentence, ...]):
    """A Dataset is a sequence of tagged sentences: ( (tokens, tags), (tokens, tags), )."""

    @staticmethod
    def from_file(filepath):
        """Construct a Dataset from a file."""
        sentences = read_tsv(filepath, cols=2)
        return Dataset(sentences)

    def unpack(self) -> Tuple[SimpleDataset, SimpleDataset]:
        """Unpack a Dataset to two DataSent(s); Tokens and Tags."""
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

    @staticmethod
    def from_file(filepath):
        """Construct a PredictedDataset from a file."""
        sentences = read_tsv(filepath, cols=3)
        return PredictedDataset(sentences)


def write_tsv(output, data: Tuple[SimpleDataset, ...]):
    """Write a tsv in many columns."""
    with open(output, "w+") as f:
        for sent_tok_tags in zip(*data):
            for tok_tags in zip(*sent_tok_tags):
                f.write("\t".join(tok_tags) + "\n")
            f.write("\n")


def read_tsv(filepath, cols=2) -> List[Tuple[Symbols, ...]]:
    """Read a single .tsv file with one, two or three columns and returns a list of the Symbols."""

    def add_sentence(sent_tokens, sent_tags, model_tags):
        """Add a sentence to the list. HAS SIDE-EFFECTS."""
        if cols == 1:
            sentences.append(Symbols(tuple(sent_tokens)))
        if cols == 2:
            sentences.append((Symbols(tuple(sent_tokens)), Symbols(tuple(sent_tags))))
        elif cols == 3:
            sentences.append(
                (
                    Symbols(tuple(sent_tokens)),
                    Symbols(tuple(sent_tags)),
                    Symbols(tuple(model_tags)),
                )
            )
        else:
            raise ValueError(f"Invalid number of cols={cols}")

    sentences: List[Tuple[Symbols, ...]] = []
    with open(filepath) as f:
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


@dataclass()
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


def wemb_str_to_emb_pair(line: str) -> Tuple[str, List[float]]:
    """Map a word-embedding string to the key and values.

    Word-embeddings string is formatted as:
    A line is a token and the corresponding embedding.
    Between the token and the embedding there is a space " ".
    Between each value in the embedding there is a space " ".
    """
    values = line.strip().split(" ")
    return (values[0], [float(n) for n in values[1:]])


def bin_str_to_emb_pair(line: str) -> Tuple[str, List[float]]:
    """Map a bin-embedding string to the key and values.

    Each line is a token and the corresponding embedding.
    Between the token and the embedding there is a ";".
    Between each value in the embedding there is a comma ",".
    """
    key, vector = line.strip().split(";")
    # We stip out '[' and ']'
    return (key, [float(n) for n in vector[1:-1].split(",")])


def emb_pairs_to_dict(
    lines: Sequence[str], f: Callable[[str], Tuple[str, List[float]]]
) -> Dict[str, List[float]]:
    """Map a sequence of strings which are embeddings using f to dictionary."""
    embedding_dict: Dict[str, List[float]] = dict()
    for line in tqdm(lines):
        key, values = f(line)
        embedding_dict[key] = values
    return embedding_dict


def map_embedding(
    embedding_dict: Dict[str, Union[List[float], List[int]]],
    filter_on: Optional[Set[str]] = None,
    special_tokens: Optional[List[Tuple[str, int]]] = None,
) -> Tuple[VocabMap, np.array]:
    """Accept an embedding dict and returns the read embedding (np.array) and the VocabMap based on the embedding dict.

    filter_on: If provided, will only return mappings for given words (if present in the file). If not provided, will read all the file.
    First element will be all zeroes for UNK.
    Returns: The embeddings as np.array and VocabMap for the embedding.
    """
    if special_tokens is None:
        special_tokens = []
    # find out how long the embeddings are, we assume all have the same length.
    length_of_embeddings = len(list(embedding_dict.values())[0])
    # words_to_add are the words in the dict, filtered
    words_to_add = Vocab(set())
    if filter_on is not None:
        log.info(f"Filtering on #symbols={len(filter_on)}")
        for filter_word in filter_on:
            # If the word is present in the file we use it.
            if filter_word in embedding_dict:
                words_to_add.add(filter_word)  # pylint: disable=no-member
    else:
        words_to_add.update(embedding_dict.keys())  # pylint: disable=no-member
    # All special tokens are treated equally as zeros
    # If the token is already present in the dict, we will overwrite it.
    for token, _ in special_tokens:
        # We treat PAD as 0s
        if token == PAD:
            embedding_dict[token] = [0 for _ in range(length_of_embeddings)]
        # Others we treat as -1, this is not perfect and the implications for BIN are unknown.
        else:
            # TODO: fix so that this is 0. To do this, we also need to read lens in model
            embedding_dict[token] = [-1 for _ in range(length_of_embeddings)]

    embeddings = np.zeros(
        shape=(len(words_to_add) + len(special_tokens), length_of_embeddings)
    )

    vocab_map = VocabMap(words_to_add, special_tokens=special_tokens)
    for symbol, idx in vocab_map.w2i.items():
        embeddings[idx] = embedding_dict[symbol]

    log.info(f"Embedding: final shape={embeddings.shape}")
    return vocab_map, embeddings


def data_loader(
    dataset: Union[Dataset, SimpleDataset],
    device: torch.device,
    dictionaries: Dict[str, VocabMap],
    shuffle=True,
    w_emb="standard",
    c_emb="standard",
    m_emb="standard",
    batch_size=16,
) -> Iterable[Dict[str, Optional[torch.Tensor]]]:
    """Perpare the data according to parameters and return batched Tensors."""
    if shuffle:
        dataset_l = list(dataset)
        random.shuffle(dataset_l)
        # TODO: fix constructor, we don't know that it is a Dataset
        dataset = Dataset(dataset_l)  # type: ignore
    length = len(dataset)
    for ndx in range(0, length, batch_size):
        batch = dataset[ndx : min(ndx + batch_size, length)]
        batch_y: Optional[SimpleDataset] = None
        if type(dataset) is Dataset:
            batch = Dataset(batch)
            batch_x, batch_y = batch.unpack()
        elif type(dataset) is SimpleDataset:
            batch = cast(SimpleDataset, batch)
            batch_x = batch
        else:
            raise ValueError(f"Unsupported dataset type={type(batch)}")

        batch_w: Optional[torch.Tensor] = None
        batch_c: Optional[torch.Tensor] = None
        batch_m: Optional[torch.Tensor] = None
        batch_t: Optional[torch.Tensor] = None
        log.debug(batch_x)
        batch_lens = torch.tensor(  # pylint: disable=not-callable
            [len(sent) for sent in batch_x]
        ).to(device, dtype=torch.int64)
        if w_emb == "standard" or w_emb == "pretrained":
            # We need the w_map
            w2i = dictionaries["w_map"].w2i
            batch_w = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.tensor(  # pylint: disable=not-callable
                        [w2i[token] if token in w2i else w2i[UNK] for token in sent]
                    )
                    for sent in batch_x
                ],
                batch_first=True,
                padding_value=w2i[PAD],
            ).to(device)
            # First pad, then map to index
        elif w_emb == "none":
            # Nothing to do.
            pass
        else:
            raise ValueError(f"Unsupported w_emb={w_emb}")
        if m_emb == "standard" or m_emb == "extra":
            w2i = dictionaries["m_map"].w2i
            batch_m = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.tensor(  # pylint: disable=not-callable
                        [w2i[token] if token in w2i else w2i[UNK] for token in sent]
                    )
                    for sent in batch_x
                ],
                batch_first=True,
                padding_value=w2i[PAD],
            ).to(device)
        elif m_emb == "none":
            # Nothing to do.
            pass
        else:
            raise ValueError(f"Unsupported m_emb={m_emb}")
        if c_emb == "standard":
            from . import model

            w2i = dictionaries["c_map"].w2i
            sents_padded = []
            for sent in batch_x:
                try:
                    sents_padded.append(
                        torch.nn.utils.rnn.pad_sequence(
                            [
                                torch.tensor(  # pylint: disable=not-callable
                                    [w2i[SOS]]
                                    + [
                                        w2i[char] if char in w2i else w2i[UNK]
                                        for char in token
                                    ]
                                    + [w2i[EOS]]
                                )
                                for token in sent
                            ],
                            batch_first=True,
                            padding_value=w2i[PAD],
                        )
                    )
                except IndexError:
                    log.error(f"Invalid sequence={sent}, in batch={batch_x}")
            max_words = max((t.shape[0] for t in sents_padded))
            max_chars = max((t.shape[1] for t in sents_padded))
            sents_padded = [
                model.copy_into_larger_tensor(
                    t, t.new_zeros(size=(max_words, max_chars))
                )
                for t in sents_padded
            ]
            batch_c = torch.nn.utils.rnn.pad_sequence(
                sents_padded, batch_first=True, padding_value=w2i[PAD]
            ).to(device)
        elif c_emb == "none":
            # Nothing to do.
            pass
        else:
            raise ValueError(f"Unsupported c_emb={c_emb}")

        if batch_y is not None:
            w2i = dictionaries["t_map"].w2i
            batch_t = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.tensor(  # pylint: disable=not-callable
                        [w2i[token] if token in w2i else w2i[UNK] for token in sent]
                    )
                    for sent in batch_y
                ],
                batch_first=True,
                padding_value=w2i[PAD],
            ).to(device)
        yield {
            "w": batch_w,
            "c": batch_c,
            "m": batch_m,
            "t": batch_t,
            "lens": batch_lens,
        }
