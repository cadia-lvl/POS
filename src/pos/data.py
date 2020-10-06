"""Data preparation and reading."""
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
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
import random

from . import flair_embeddings as flair
from .types import Vocab, Dataset, SimpleDataset, VocabMap


log = logging.getLogger(__name__)

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
        # Others we treat also as 0
        else:
            embedding_dict[token] = [0 for _ in range(length_of_embeddings)]

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
    shuffle: bool,
    w_emb: str,
    c_emb: str,
    m_emb: str,
    batch_size: int,
) -> Iterable[Dict[str, Optional[torch.Tensor]]]:
    """Perpare the data according to parameters and return batched Tensors."""
    if shuffle and type(dataset) == Dataset:
        dataset_l = list(deepcopy(dataset))
        random.shuffle(dataset_l)
        dataset = Dataset(dataset_l)  # type: ignore
    length = len(dataset)
    for ndx in range(0, length, batch_size):
        batch = dataset[ndx : min(ndx + batch_size, length)]
        batch_y: Optional[SimpleDataset] = None
        if type(dataset) is Dataset:
            batch = cast(Dataset, batch)
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
        elif w_emb == "electra":
            batch_w = torch.nn.utils.rnn.pad_sequence(
                [flair.electra_embedding(sent) for sent in batch_x],
                batch_first=True,
                padding_value=0.0,  # Pad with 0.0
            ).to(device)

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
