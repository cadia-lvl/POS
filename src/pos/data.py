"""Data preparation and reading."""
from typing import List, Tuple, Set, Dict, Optional, Iterable, Sequence, Callable, Any
from functools import reduce
from operator import add
import logging
from enum import Enum

from tqdm import tqdm
from torch import (
    Tensor,
    from_numpy,
    zeros_like,
    zeros,
)
from torch.nn.utils.rnn import pad_sequence

from .core import Vocab, VocabMap, Sentence, Dicts, FieldedDataset


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


class BATCH_KEYS(Enum):
    """Keys used on the batch dictionary."""

    TOKENS = "tokens"
    FULL_TAGS = "full_tags"
    TARGET_FULL_TAGS = "target_full_tags"
    LEMMAS = "lemmas"
    TARGET_LEMMAS = "target_lemmas"
    LENGTHS = "lens"


def map_to_index(sentence: Sentence, w2i: Dict[str, int]) -> Tensor:
    """Map a sequence to indices."""
    return Tensor(
        [w2i[token] if token in w2i else w2i[UNK] for token in sentence]
    ).long()


def map_to_chars_and_index(
    sentence: Sentence, w2i: Dict[str, int], add_eos=True, add_sos=True
) -> Tensor:
    """Map a sequence to characters then to indices."""
    SOS_l = [w2i[SOS]] if add_sos else []
    EOS_l = [w2i[EOS]] if add_eos else []
    return pad_sequence(
        [
            Tensor(
                SOS_l
                + [w2i[char] if char in w2i else w2i[UNK] for char in token]
                + EOS_l
            ).long()
            for token in sentence
        ],
        batch_first=True,
        padding_value=w2i[PAD],
    )


def map_to_chars_batch(
    sentences: Sequence[Sentence], w2i: Dict[str, int], add_eos=True, add_sos=True
) -> Tensor:
    """Map a batch of sentences to characters of words. This is convoluted, I know."""
    sents_padded = []
    for sentence in sentences:
        sents_padded.append(
            map_to_chars_and_index(sentence, w2i, add_eos=add_eos, add_sos=add_sos)
        )
    max_words = max((t.shape[0] for t in sents_padded))
    max_chars = max((t.shape[1] for t in sents_padded))
    sents_padded = [
        copy_into_larger_tensor(t, t.new_zeros(size=(max_words, max_chars)))
        for t in sents_padded
    ]
    return (
        pad_sequence(sents_padded, batch_first=True, padding_value=w2i[PAD])
        .reshape(shape=(-1, max_chars))
        .long()
    )


def map_to_index_batch(sentences: Sequence[Sentence], w2i: Dict[str, int]):
    """Map to index, batch."""
    return pad_sequence(
        [map_to_index(sentence=sentence, w2i=w2i) for sentence in sentences],
        batch_first=True,
        padding_value=w2i[PAD],
    )


def copy_into_larger_tensor(tensor: Tensor, like_tensor: Tensor) -> Tensor:
    """Create a larger tensor based on given tensor. Only works for 2-dims."""
    base = zeros_like(like_tensor)
    base[: tensor.shape[0], : tensor.shape[1]] = tensor
    return base


def collate_fn(batch: Sequence[Tuple[Sentence, ...]]) -> Dict[BATCH_KEYS, Any]:
    """Map the inputs to batches."""
    batch_dict = {}
    if len(batch[0]) >= 1:  # we assume we are given the tokens
        batch_dict[BATCH_KEYS.TOKENS] = tuple(element[0] for element in batch)
    if len(batch[0]) >= 2:  # Next, the tags. Usually only for training.
        batch_dict[BATCH_KEYS.FULL_TAGS] = tuple(element[1] for element in batch)
    if len(batch[0]) >= 3:  # lastly, the lemmas. Usually only for training.
        batch_dict[BATCH_KEYS.LEMMAS] = tuple(element[2] for element in batch)
    batch_dict[BATCH_KEYS.LENGTHS] = tuple(
        len(x) for x in batch_dict[BATCH_KEYS.TOKENS]  # type: ignore
    )
    return batch_dict


def read_datasets(
    file_paths: List[str], max_sent_length=0, max_lines=0, fields=None
) -> FieldedDataset:
    """Read tagged datasets from multiple files.

    Args:
        max_sent_length: Sentences longer than "max_sent_length" are thrown away.
        max_lines: Will only keep the first "max_lines" sentences.
        fields: The tagged fields in the dataset
    """
    ds = reduce(
        add,
        (
            FieldedDataset.from_file(training_file, fields)
            for training_file in file_paths
        ),
    )
    if max_sent_length:
        # We want to filter out sentences which are too long (and throw them away, for now)
        ds = FieldedDataset(
            tuple(zip(*[x for x in ds if len(x[0]) <= max_sent_length])), ds.fields
        )
    # DEBUG - read a subset of the data
    if max_lines:
        ds = FieldedDataset(ds[:max_lines], ds.fields)  # type: ignore
    return ds


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


def emb_pairs_to_dict(
    lines: Iterable[str], f: Callable[[str], Tuple[str, Tensor]]
) -> Dict[str, Tensor]:
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

    embeddings = zeros(
        size=(len(words_to_add) + len(special_tokens), length_of_embeddings)
    )

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
    w_embedding = from_numpy(w_embedding).float()
    return w_map, w_embedding


def load_dicts(
    train_ds: FieldedDataset,
    pretrained_word_embeddings_file=None,
    morphlex_embeddings_file=None,
    known_chars_file=None,
):
    """Load all the modules for the model."""
    embeddings: Dict[Dicts, Tensor] = {}
    dictionaries: Dict[Dicts, VocabMap] = {}

    # Pretrained
    if pretrained_word_embeddings_file:
        m_map, m_embedding = read_pretrained_word_embeddings(
            pretrained_word_embeddings_file
        )
        embeddings[Dicts.Pretrained] = m_embedding
        dictionaries[Dicts.Pretrained] = m_map

    dictionaries[Dicts.Tokens] = train_ds.get_vocab_map(special_tokens=VocabMap.UNK_PAD)

    # MorphLex
    if morphlex_embeddings_file:
        # File is provided, use it.
        m_map, m_embedding = read_morphlex(morphlex_embeddings_file)
        embeddings[Dicts.MorphLex] = m_embedding
        dictionaries[Dicts.MorphLex] = m_map

    # Character mappings, if a file is provided, use it. Otherwise, build from dataset.
    if known_chars_file:
        char_vocab = Vocab.from_file(known_chars_file)
    else:
        char_vocab = train_ds.get_char_vocab()
    c_map = VocabMap(char_vocab, special_tokens=VocabMap.UNK_PAD_EOS_SOS,)
    dictionaries[Dicts.Chars] = c_map

    # TAGS (POS)
    dictionaries[Dicts.FullTag] = train_ds.get_tag_vocab_map(
        special_tokens=VocabMap.UNK_PAD
    )
    return embeddings, dictionaries
