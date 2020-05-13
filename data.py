import copy
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Set, Dict, Optional, Union
import logging

from tqdm import tqdm
import numpy as np
import torch
import random

log = logging.getLogger()
# When iterating over Sent-x, whole sentences are yielded
SentTokens = Tuple[str, ...]
SentTags = Tuple[str, ...]
# Either tokens or tags, we don't care
Sent = Union[SentTags, SentTokens]
Vocab = Set[str]
DataPairs = Tuple[List[SentTokens], List[SentTags]]
DataPairs_c = Tuple[Tuple[List[SentTokens], List[SentTags]], List[SentTags]]
Data = Union[DataPairs, DataPairs_c]
# TODO: Remove
TaggedToken = Tuple[str, str]
TaggedSentence = List[TaggedToken]
Corpus = List[TaggedSentence]

x_i_j = Tuple[List[str], str, str]
x_i = List[x_i_j]
X = List[x_i]
y_i_j = str
y_i = List[y_i_j]
Y = List[y_i]

# To pad in batches
PAD = '<pad>'
PAD_ID = 0
# For unkown words in testing
UNK = '<unk>'
UNK_ID = 1
# For EOS and SOS in char BiLSTM
EOS = '</s>'
EOS_ID = 2
SOS = '<s>'
SOS_ID = 3


def read_tsv(input) -> Corpus:
    corpus = []
    with open(input) as f:
        tokens: TaggedSentence = []
        for line in f:
            line = line.strip()
            # We read a blank line - sentence has been read.
            if not line:
                corpus.append(tokens)
                tokens = []
            else:
                word, tag = line.split()
                tokens.append((word, tag))
    # The file does not neccessarily end with a blank line
    if len(tokens) != 0:
        corpus.append(tokens)
    return corpus


def write_tsv(output, corpus: Corpus):
    with open(output, 'w+') as f:
        for sent in corpus:
            for word, tag in sent:
                f.write(f'{word}\t{tag}\n')
            f.write('\n')


def tsv_to_pairs(corpus: Corpus) -> DataPairs:
    tokens, tags = [], []
    for tagged_sentence in corpus:
        sent_tokens, sent_tags = [], []
        for token, tag in tagged_sentence:
            sent_tokens.append(token)
            sent_tags.append(tag)
        tokens.append(tuple(sent_tokens))
        tags.append(tuple(sent_tags))
    return tokens, tags


def get_vocab(sentences: List[Sent]) -> Vocab:
    return set(tok for sent in sentences for tok in sent)


def get_tok_freq(sentences: List[Sent]) -> Counter:
    return Counter((tok for sent in sentences for tok in sent))


def coarsify(sentences: List[SentTags]) -> List[SentTags]:
    return [tuple(tag[0] for tag in tags) for tags in sentences]


def read_known_characters(char_file) -> Vocab:
    with open(char_file) as f:
        return get_vocab([tuple(line.strip().split()) for line in f.readlines()])


@dataclass()
class VocabMap:
    w2i: Dict[str, int]
    i2w: Dict[int, str]

    def __init__(self, vocab: Vocab, special_tokens: Optional[List[Tuple[str, int]]] = None):
        """
        Builds a vocabulary mapping from the provided vocabulary, starting at index=0.
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
        return len(self.w2i)


def read_embedding(emb_file, filter_on: Optional[Set[str]] = None, special_tokens: Optional[List[Tuple[str, int]]] = None) -> Tuple[VocabMap, np.array]:
    """
    Reads an embedding file and returns the read embedding (np.array) and the VocabMap based on the read file.
    filter_on: If provided, will only return mappings for given words (if present in the file). If not provided, will read all the file.
    First element will be all zeroes for UNK.
    Returns: The embeddings as np.array and VocabMap for the embedding.
    """
    if special_tokens is None:
        special_tokens = []
    log.info(f'Embedding reading={emb_file}')
    # This can be huge
    embedding_dict: Dict[str, List[int]] = dict()
    with open(emb_file) as f:
        for line in tqdm(f):
            key, vector = line.strip().split(";")
            # We stip out '[' and ']'
            embedding_dict[key] = [int(n) for n in vector[1:-1].split(',')]
    # find out how long the embeddings are, we assume all have the same length.
    length_of_embeddings = len(list(embedding_dict.values())[0])
    # All special tokens are treated equall as zeros
    for token, _ in special_tokens:
        embedding_dict[token] = [0 for _ in range(length_of_embeddings)]
    # UNK should not be in words_to_add, since the vocab_map will handle adding it.
    words_to_add = set()
    if filter_on is not None:
        log.info(f'Filtering on #symbols={len(filter_on)}')
        for filter_word in filter_on:
            # If the word is present in the file we use it.
            if filter_word in embedding_dict:
                words_to_add.add(filter_word)
    else:
        words_to_add = set(embedding_dict.keys())
    # + special tokens
    embeddings = np.zeros(
        shape=(len(words_to_add) + len(special_tokens), length_of_embeddings))

    vocab_map = VocabMap(words_to_add, special_tokens=special_tokens)
    for symbol, idx in vocab_map.w2i.items():
        embeddings[idx] = embedding_dict[symbol]

    log.info(f'Embedding: final shape={embeddings.shape}')
    return vocab_map, embeddings


class Embeddings:
    # Just here for good-old times
    def __init__(self, vocabmap: VocabMap, embedding: np.array):
        self.vocab = vocabmap.w2i
        self.embeddings = embedding


def unk_analysis(train: Vocab, test: Vocab):
    log.info(f'Train len={len(train)}')
    log.info(f'Test len={len(test)}')
    log.info(f'Test not in train={len(test-train)}')


def _make_x_i_j(token: str) -> x_i_j:
    # ([SOS + characters + EOS], word, morph)
    return ([SOS] + [c for c in token] + [EOS], token, token)  # type: ignore


def _make_x_i(example: SentTokens) -> x_i:
    # We do not put SOS or EOS on words
    return [_make_x_i_j(tok) for tok in example]


def make_x(sentences: List[SentTokens]) -> X:
    return [_make_x_i(sent) for sent in sentences]


def make_y(sentences: List[SentTags]) -> Y:
    return [[tag for tag in sent] for sent in sentences]


def pad_x(x: X) -> X:
    """ Pads characters and sentences in place. Input should be strings since we need the length of a token"""
    longest_token_in_examples = max(
        (max(
            (len(chars) for chars, _, _ in x_i)
        ) for x_i in x)
    )
    longest_sent_in_examples = max(len(x_i) for x_i in x)
    log.debug("Longest token in batch", longest_token_in_examples)
    log.debug("Longest sent in batch", longest_sent_in_examples)
    x_pad = copy.deepcopy(x)
    for x_i_pad in x_pad:
        # Pad sentence-wise
        while len(x_i_pad) < longest_sent_in_examples:
            empty_chars: List[str] = []
            x_i_pad.append((list(empty_chars), PAD, PAD))
        # Pad character-wise
        for chars, _, _ in x_i_pad:
            while len(chars) < longest_token_in_examples:
                chars.append(PAD)
    return x_pad


def pad_y(y: Y) -> Y:
    longest_sent_in_examples = max((len(y_i) for y_i in y))
    y_pad = copy.deepcopy(y)
    for y_i_pad in y_pad:
        # Pad sentence-wise
        while len(y_i_pad) < longest_sent_in_examples:
            y_i_pad.append(PAD)
    return y_pad


def to_idx_x(x: X, c_map: VocabMap, w_map: VocabMap, m_map: VocabMap) -> torch.Tensor:
    x_idx = []
    for x_i in x:
        x_i_idx = []
        for chars, w, m in x_i:
            x_i_idx.append((
                [c_map.w2i[c]
                    if c in c_map.w2i else UNK_ID for c in chars],
                w_map.w2i[w] if w in w_map.w2i else UNK_ID,
                m_map.w2i[m] if m in m_map.w2i else UNK_ID
            ))
        x_idx.append(x_i_idx)
    # Break the list up and return (b, s, f)
    return torch.tensor([[(*chars, w, m) for chars, w, m in x_i_idx] for x_i_idx in x_idx])


def to_idx_y(y: Y, t_map: VocabMap) -> torch.Tensor:
    # UNK is not supported. We should raise an exception if tag is not present in mapping
    return torch.tensor([[t_map.w2i[t] for t in y_i] for y_i in y])


def make_batches(data: Tuple[List[SentTokens], List[SentTags]],
                 batch_size=32,
                 shuffle=True,
                 c_map=None,
                 w_map=None,
                 m_map=None,
                 t_map=None,
                 device=None):
    if shuffle:
        x_y = list(zip(*data))
        random.shuffle(x_y)
        data = tuple(map(list, zip(*x_y)))  # type: ignore

    # Correct format
    x = make_x(data[0])
    y = make_y(data[1])

    # Make batches
    length = len(x)
    for ndx in range(0, length, batch_size):
        # First pad, then map to index
        yield (to_idx_x(pad_x(x[ndx:min(ndx + batch_size, length)]),
                        c_map=c_map,
                        w_map=w_map,
                        m_map=m_map).to(device),
               to_idx_y(pad_y(y[ndx:min(ndx + batch_size, length)]),
                        t_map=t_map).to(device))
