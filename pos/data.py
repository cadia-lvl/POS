"""Data preparation and reading."""
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Set, Dict, Optional, Union, Iterable, NewType, cast
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
Sent = NewType("Sent", Tuple[str, ...])
TaggedSent = NewType("TaggedSent", Tuple[Sent, Sent])


class Vocab(set):
    """A class to hold a vocabulary as an unordered set of symbols."""


class Dataset(tuple):
    """A class to hold tagged sentences: ( (tokens, tags), (tokens, tags), )."""


class DataSent(tuple):
    """A class to hold (untagged) sentences: ( tokens, tokens, )."""


def write_tsv(output, data: Tuple[DataSent, ...]):
    """Write a tsv in many columns."""
    with open(output, "w+") as f:
        for sent_tok_tags in zip(*data):
            for tok_tags in zip(*sent_tok_tags):
                f.write("\t".join(tok_tags) + "\n")
            f.write("\n")


def read_tsv(input) -> Tuple[DataSent, DataSent, DataSent]:
    """Read a tsv, two or three columns."""
    tokens = []
    tags = []
    model_tags = []
    with open(input) as f:
        sent_tokens: List[str] = []
        sent_tags: List[str] = []
        sent_model_tags: List[str] = []
        for line in f:
            line = line.strip()
            # We read a blank line - sentence has been read.
            if not line:
                tokens.append(Sent(tuple(sent_tokens)))
                tags.append(Sent(tuple(sent_tags)))
                model_tags.append(Sent(tuple(sent_model_tags)))
                sent_tokens = []
                sent_tags = []
                sent_model_tags = []
            else:
                tok_tags = line.split()
                sent_tokens.append(tok_tags[0])
                sent_tags.append(tok_tags[1])
                if len(tok_tags) == 3:
                    sent_model_tags.append(tok_tags[2])

    # The file does not neccessarily end with a blank line
    if len(sent_tokens) != 0:
        tokens.append(Sent(tuple(sent_tokens)))
        tags.append(Sent(tuple(sent_tags)))
        model_tags.append(Sent(tuple(sent_model_tags)))
    return (DataSent(tuple(tokens)), DataSent(tuple(tags)), DataSent(tuple(model_tags)))


def read_pos_tsv(filepath) -> Dataset:
    """Read a single .tsv file with two columns and returns a Dataset."""
    sentences = []
    with open(filepath) as f:
        sent_tokens: List[str] = []
        sent_tags: List[str] = []
        for line in f:
            line = line.strip()
            # We read a blank line and buffer is not empty - sentence has been read.
            if not line and len(sent_tokens) != 0:
                sentences.append(
                    TaggedSent((Sent(tuple(sent_tokens)), Sent(tuple(sent_tags))))
                )
                sent_tokens = []
                sent_tags = []
            else:
                token, tag = line.split()
                sent_tokens.append(token)
                sent_tags.append(tag)
    # For the last sentence
    if len(sent_tokens) != 0:
        log.info("No newline at end of file, handling it.")
        sentences.append(TaggedSent((Sent(tuple(sent_tokens)), Sent(tuple(sent_tags)))))
    return Dataset(sentences)


def read_datasent(file_stream) -> DataSent:
    """Read a filestream, with token per line and new-line separating sentences."""
    sentences = []
    sent_tokens: List[str] = []
    for line in file_stream:
        line = line.strip()
        # We read a blank line and buffer is not empty - sentence has been read.
        if not line and len(sent_tokens) != 0:
            sentences.append(Sent(tuple(sent_tokens)))
            sent_tokens = []
        else:
            token = line
            sent_tokens.append(token)
    # For the last sentence
    if len(sent_tokens) != 0:
        log.info("No newline at end of file, handling it.")
        sentences.append(Sent(tuple(sent_tokens)))
    return DataSent(sentences)


def get_vocab(sentences: Iterable[Sent]) -> Vocab:
    """Iterate over sentences and extract tokens/tags."""
    return Vocab((tok for sent in sentences for tok in sent))


def get_tok_freq(sentences: Iterable[Sent]) -> Counter:
    """Gather token/tag frequencies."""
    return Counter((tok for sent in sentences for tok in sent))


def read_vocab(vocab_file) -> Vocab:
    """Read a vocab file and return Vocab."""
    with open(vocab_file) as f:
        return get_vocab([Sent(tuple(line.strip().split())) for line in f.readlines()])


def read_datasets(data_paths: List[str]) -> Dataset:
    """Read the given paths and returns a combined dataset of all paths.

    The files should be .tsv with two columns, first column is the word, second column is the POS.
    Sentences are separated with a blank line.
    """
    log.info(f"Reading files={data_paths}")
    return Dataset(
        (tagged_sent for path in data_paths for tagged_sent in read_pos_tsv(path))
    )


def unpack_dataset(dataset: Dataset) -> Tuple[DataSent, DataSent]:
    """Unpack a Dataset to two DataSent(s); Tokens and Tags."""
    tokens = DataSent(tokens for tokens, _ in dataset)
    tags = DataSent(tags for _, tags in dataset)
    return (tokens, tags)


@dataclass()
class VocabMap:
    """For w2i and i2w storage for different dictionaries."""

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


def read_word_embedding(embedding_data: Iterable[str],) -> Dict[str, List[float]]:
    """Read an embedding file and return the embedding mapping.

    Formatted as:
    Each line is a token and the corresponding embedding.
    Between the token and the embedding there is a space " ".
    Between each value in the embedding there is a space " ".

    See called function for further parameters and return value.
    """
    log.info("Reading word embeddings")
    embedding_dict: Dict[str, List[float]] = dict()
    it = iter(embedding_data)
    # pop the number of vectors and dimension
    next(it)
    for line in tqdm(it):
        values = line.strip().split(" ")
        token = values[0]
        vector = values[1:]
        embedding_dict[token] = [float(n) for n in vector]
    return embedding_dict


def read_bin_embedding(embedding_data: Iterable[str],) -> Dict[str, List[int]]:
    """Read an embedding file and return the embedding mapping.

    Formatted as:
    Each line is a token and the corresponding embedding.
    Between the token and the embedding there is a ";".
    Between each value in the embedding there is a comma ",".

    See called function for further parameters and return value.
    """
    log.info("Reading bin embeddings")
    embedding_dict: Dict[str, List[int]] = dict()
    for line in tqdm(embedding_data):
        key, vector = line.strip().split(";")
        # We stip out '[' and ']'
        embedding_dict[key] = [int(n) for n in vector[1:-1].split(",")]
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
                words_to_add.add(filter_word)
    else:
        words_to_add.update(embedding_dict.keys())
    # All special tokens are treated equally as zeros
    # If the token is already present in the dict, we will overwrite it.
    for token, _ in special_tokens:
        # We treat PAD as 0s
        if token == PAD:
            embedding_dict[token] = [0 for _ in range(length_of_embeddings)]
        # Others we treat as -1, this is not perfect and the implications for BIN are unknown.
        else:
            embedding_dict[token] = [-1 for _ in range(length_of_embeddings)]

    embeddings = np.zeros(
        shape=(len(words_to_add) + len(special_tokens), length_of_embeddings)
    )

    vocab_map = VocabMap(words_to_add, special_tokens=special_tokens)
    for symbol, idx in vocab_map.w2i.items():
        embeddings[idx] = embedding_dict[symbol]

    log.info(f"Embedding: final shape={embeddings.shape}")
    return vocab_map, embeddings


def create_mappers(
    dataset: Dataset,
    w_emb="standard",
    c_emb="standard",
    m_emb="standard",
    pretrained_word_embeddings_file=None,
    morphlex_embeddings_file=None,
    known_chars: Vocab = None,
) -> Tuple[Dict[str, VocabMap], Dict[str, np.array]]:
    """Prepare the mappers for processing the data based on the parameters.

    Returns:
        The dictionaries which map symbol<->idx.
        Other configuration such as read embeddings to be loaded by the model.
    """
    dictionaries: Dict[str, VocabMap] = {}
    extras: Dict[str, np.array] = {}

    if w_emb == "pretrained":
        with open(pretrained_word_embeddings_file) as f:
            embedding_dict = read_word_embedding(f)
        w_map, w_embedding = map_embedding(
            embedding_dict=embedding_dict,  # type: ignore
            filter_on=None,
            special_tokens=[(UNK, UNK_ID), (PAD, PAD_ID)],
        )
        dictionaries["w_map"] = w_map
        extras["word_embeddings"] = w_embedding
    elif w_emb == "standard":
        dictionaries["w_map"] = VocabMap(
            get_vocab((x for x, y in dataset)),
            special_tokens=[(PAD, PAD_ID), (UNK, UNK_ID)],
        )
    elif w_emb == "none":
        # Nothing to do
        pass
    else:
        raise ValueError(f"Unknown w_emb={w_emb}")

    dictionaries["t_map"] = VocabMap(
        get_vocab((y for x, y in dataset)),
        special_tokens=[(PAD, PAD_ID), (UNK, UNK_ID),],
    )
    if c_emb == "standard" and known_chars is not None:
        dictionaries["c_map"] = VocabMap(
            known_chars,
            special_tokens=[
                (UNK, UNK_ID),
                (PAD, PAD_ID),
                (EOS, EOS_ID),
                (SOS, SOS_ID),
            ],
        )
    elif c_emb == "none":
        # Nothing to do
        pass
    else:
        raise ValueError(f"Unkown c_emb={c_emb}")
    if m_emb == "standard" or m_emb == "extra":
        with open(morphlex_embeddings_file) as f:
            embedding_dict = read_bin_embedding(f)  # type: ignore
        m_map, m_embedding = map_embedding(
            embedding_dict=embedding_dict,  # type: ignore
            filter_on=None,
            special_tokens=[(UNK, UNK_ID), (PAD, PAD_ID)],
        )
        dictionaries["m_map"] = m_map
        extras["morph_lex_embeddings"] = m_embedding
    elif m_emb == "none":
        # Nothing to do
        pass
    else:
        raise ValueError(f"Unkown c_emb={c_emb}")

    return dictionaries, extras


def data_loader(
    dataset: Union[Dataset, DataSent],
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
        dataset = Dataset(dataset_l)
    length = len(dataset)
    for ndx in range(0, length, batch_size):
        batch = dataset[ndx : min(ndx + batch_size, length)]
        batch_y: Optional[DataSent] = None
        if type(dataset) is Dataset:
            batch = cast(Dataset, batch)
            batch_x, batch_y = unpack_dataset(batch)
        elif type(dataset) is DataSent:
            batch_x = DataSent(batch)
        else:
            raise ValueError(f"Unsupported dataset type={type(batch)}")

        batch_w: Optional[torch.Tensor] = None
        batch_c: Optional[torch.Tensor] = None
        batch_m: Optional[torch.Tensor] = None
        batch_t: Optional[torch.Tensor] = None
        log.debug(batch_x)
        batch_lens = torch.tensor([len(sent) for sent in batch_x]).to(
            device, dtype=torch.int64
        )
        if w_emb == "standard" or w_emb == "pretrained":
            # We need the w_map
            w2i = dictionaries["w_map"].w2i
            batch_w = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.tensor(
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
                    torch.tensor(
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
                                torch.tensor(
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
                    torch.tensor(
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
