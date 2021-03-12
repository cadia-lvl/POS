"""Reading pretrained files."""
import logging
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from torch import Tensor, zeros
from tqdm import tqdm

from pos.core import Vocab, VocabMap

from .constants import PAD, PAD_ID, UNK, UNK_ID

log = logging.getLogger(__name__)


def wemb_str_to_emb_pair(line: str) -> Tuple[str, Tensor]:
    """Map a word-embedding string to the key and values.

    Word-embeddings string is formatted as:
    A line is a token and the corresponding embedding.
    Between the token and the embedding there is a space " ".
    Between each value in the embedding there is a space " ".
    """
    values = line.strip().split(" ")
    return (values[0], Tensor([float(n) for n in values[1:]]))


def bin_str_to_emb_pair(line: str) -> Tuple[str, Tensor]:
    """Map a bin-embedding string to the key and values.

    Each line is a token and the corresponding embedding.
    Between the token and the embedding there is a ";".
    Between each value in the embedding there is a comma ",".
    """
    key, vector = line.strip().split(";")
    # We stip out '[' and ']'
    return (key, Tensor([float(n) for n in vector[1:-1].split(",")]))


def emb_pairs_to_dict(lines: Iterable[str], f: Callable[[str], Tuple[str, Tensor]]) -> Dict[str, Tensor]:
    """Map a sequence of strings which are embeddings using f to dictionary."""
    embedding_dict: Dict[str, Tensor] = dict()
    for line in tqdm(lines):
        key, values = f(line)
        embedding_dict[key] = values
    return embedding_dict


def map_embedding(
    embedding_dict: Dict[str, Tensor],
    filter_on: Optional[Set[str]] = None,
    special_tokens: Optional[List[Tuple[str, int]]] = None,
) -> Tuple[VocabMap, Tensor]:
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
            embedding_dict[token] = Tensor([0 for _ in range(length_of_embeddings)])
        # Others we treat also as 0
        else:
            embedding_dict[token] = Tensor([0 for _ in range(length_of_embeddings)])

    embeddings = zeros(size=(len(words_to_add) + len(special_tokens), length_of_embeddings))

    vocab_map = VocabMap(words_to_add, special_tokens=special_tokens)
    for symbol, idx in vocab_map.w2i.items():
        embeddings[idx] = embedding_dict[symbol]

    log.info(f"Embedding: final shape={embeddings.shape}")
    return vocab_map, embeddings


def read_morphlex(filepath: str) -> Tuple[VocabMap, Tensor]:
    """Read the MorphLex embeddings. Return the VocabMap and embeddings."""
    with open(filepath) as f:
        it = iter(f)
        embedding_dict = emb_pairs_to_dict(it, bin_str_to_emb_pair)
    m_map, m_embedding = map_embedding(
        embedding_dict=embedding_dict,  # type: ignore
        filter_on=None,
        special_tokens=[(UNK, UNK_ID), (PAD, PAD_ID)],
    )
    return m_map, m_embedding


def read_pretrained_word_embeddings(filepath: str) -> Tuple[VocabMap, Tensor]:
    """Read the pretrained word embeddings."""
    with open(filepath) as f:
        it = iter(f)
        # pop the number of vectors and dimension
        next(it)
        embedding_dict = emb_pairs_to_dict(it, wemb_str_to_emb_pair)
    w_map, w_embedding = map_embedding(
        embedding_dict=embedding_dict,  # type: ignore
        filter_on=None,
        special_tokens=[(UNK, UNK_ID), (PAD, PAD_ID)],
    )
    return w_map, w_embedding
