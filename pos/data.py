import copy
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Set, Dict, Optional, Union, Iterable
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
DataSent = List[Sent]
DataPairs = Tuple[List[SentTokens], List[SentTags]]
DataPairs_c = Tuple[Tuple[List[SentTokens], List[SentTags]], List[SentTags]]
Data = Union[DataPairs, DataPairs_c]

TaggedToken = Tuple[str, str]
Token = str
In_Tok = Union[Token, TaggedToken]
In_Sent = List[In_Tok]
In = List[In_Sent]

SentTokTag = List[TaggedToken]
Corpus = List[SentTokTag]


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


def write_tsv(output, data: Tuple[DataSent, ...]):
    with open(output, 'w+') as f:
        for sent_tok_tags in zip(*data):
            for tok_tags in zip(*sent_tok_tags):
                f.write('\t'.join(tok_tags) + '\n')
            f.write('\n')


def read_tsv(input) -> Tuple[DataSent, DataSent, DataSent]:
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
                tokens.append(tuple(sent_tokens))
                tags.append(tuple(sent_tags))
                model_tags.append(tuple(sent_model_tags))
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
        tokens.append(tuple(sent_tokens))
        tags.append(tuple(sent_tags))
        model_tags.append(tuple(sent_model_tags))
    return tokens, tags, model_tags


def get_vocab(sentences: List[Sent]) -> Vocab:
    return set(tok for sent in sentences for tok in sent)


def get_tok_freq(sentences: List[Sent]) -> Counter:
    return Counter((tok for sent in sentences for tok in sent))


def coarsify(sentences: List[SentTags]) -> List[SentTags]:
    return [tuple(tag[0] for tag in tags) for tags in sentences]


def read_vocab(vocab_file) -> Vocab:
    with open(vocab_file) as f:
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
    # All special tokens are treated equally as zeros - just like in DyNet
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


x_i_j = Tuple[List[str], str, str]
# For coarse tags, slightly hacky
x_i_j_c = Tuple[List[str], str, str, str]
x_i = List[Union[x_i_j]]
X = List[x_i]
y_i_j = str
y_i = List[y_i_j]
Y = List[y_i]
@dataclass()
class DataVocabMap:
    c_map: VocabMap  # characters
    w_map: VocabMap  # words "tokens"
    m_map: VocabMap  # morphlex
    t_map: VocabMap  # tags - the target
    # Additional feature maps, words/tokens, morphlex and c_tags, in that order
    f_maps: List[VocabMap]
    c_t_map: VocabMap  # coarse-tags - optional input
    t_freq: Counter

    def __init__(self, tokens: List[SentTokens], tags: Vocab, chars: Vocab, c_tags: Vocab = None, unk_to_tags=False):
        """
        Here we create all the necessary vocabulary mappings for the batch function.
        """
        # We need EOS and SOS for chars
        self.c_map = VocabMap(chars, special_tokens=[
            (UNK, UNK_ID),
            (PAD, PAD_ID),
            (EOS, EOS_ID),
            (SOS, SOS_ID)
        ])
        self.w_map = VocabMap(get_vocab(tokens), special_tokens=[
            (UNK, UNK_ID),
            (PAD, PAD_ID),
        ])
        special_tokens = [
            (PAD, PAD_ID),
        ]
        if unk_to_tags:
            special_tokens.append((UNK, UNK_ID))
        self.t_map = VocabMap(tags, special_tokens=special_tokens)
        log.info(f'Character vocab={len(self.c_map)}')
        log.info(f'Word vocab={len(self.w_map)}')
        log.info(f'Tag vocab={len(self.t_map)}')
        # Add the mappings to a list for idx mapping later
        self.x_maps = [self.w_map]
        if c_tags:
            # The c_tags will be padded, but we do not support UNK
            self.c_t_map = VocabMap(c_tags, special_tokens=[
                (PAD, PAD_ID),
            ])
            log.info(f'Coarse tag vocab={len(self.c_t_map)}')
        self.t_freq = get_tok_freq(tokens)

    def add_morph_map(self, m_map):
        """
        Adds the m_map to the object and the c_t_map if defined (to maintain order). Do not call twice.
        """
        self.m_map = m_map
        self.x_maps.append(self.m_map)
        # We add the c_tag map last
        if hasattr(self, 'c_t_map'):
            self.x_maps.append(self.c_t_map)

    @classmethod
    def make_x(cls, sentences: In) -> X:
        return [
            [
                # ([SOS + characters + EOS], word, morph)
                ([SOS] + [c for c in tok] + [EOS], tok, tok)  # type: ignore
                if type(tok) == str else
                # ([SOS + characters + EOS], word, morph, c_tag)
                ([SOS] + [c for c in tok[0]] + [EOS], tok[0], tok[0], tok[1])
                for tok in sent]
            for sent in sentences]

    @classmethod
    def make_y(cls, sentences: List[SentTags]) -> Y:
        return [[tag for tag in sent] for sent in sentences]

    @classmethod
    def pad_x(cls, x: X) -> X:
        """ Pads characters and sentences in place. Input should be strings since we need the length of a token"""
        longest_token_in_examples = max(
            (max(
                # Avoid tuple unpacking
                (len(x_i_js[0]) for x_i_js in x_is)
            ) for x_is in x)
        )
        longest_sent_in_examples = max(len(x_i) for x_i in x)
        log.debug("Longest token in batch", longest_token_in_examples)
        log.debug("Longest sent in batch", longest_sent_in_examples)
        x_pad = copy.deepcopy(x)
        for x_i_pad in x_pad:
            # chars, word, morph
            if len(x_i_pad[0]) == 3:
                empty_rest = (PAD, PAD)
            # chars, word, morph, c_tag
            else:
                empty_rest = (PAD, PAD, PAD)  # type: ignore
            # Pad sentence-wise
            while len(x_i_pad) < longest_sent_in_examples:
                x_i_pad.append((list(), *empty_rest))
            # Pad character-wise
            for x_i_js_pad in x_i_pad:
                while len(x_i_js_pad[0]) < longest_token_in_examples:
                    x_i_js_pad[0].append(PAD)
        return x_pad

    @classmethod
    def pad_y(cls, y: Y) -> Y:
        longest_sent_in_examples = max((len(y_i) for y_i in y))
        y_pad = copy.deepcopy(y)
        for y_i_pad in y_pad:
            # Pad sentence-wise
            while len(y_i_pad) < longest_sent_in_examples:
                y_i_pad.append(PAD)
        return y_pad

    def to_idx_x(self, x: X) -> torch.Tensor:
        """
        Maps the input to idices, breaks up character list and returns a tensor. Uses f_map for additinal features.
        Input needs to be padded, otherwise the character break up will mess things up.
        """
        x_idx = []
        for x_is in x:
            x_i_idx: List[Tuple[int, ...]] = []
            for x_i_js in x_is:
                chars = x_i_js[0]
                chars_idx = [self.c_map.w2i[c]
                             if c in self.c_map.w2i else UNK_ID for c in chars]
                # for x_maps, 0 = word/token, 1 = morphlex, 2 = c_tags (optional)
                rest = x_i_js[1:]
                rest_idx = [self.x_maps[i].w2i[rest[i]] if rest[i] in self.x_maps[i].w2i else UNK_ID
                            for i in range(len(rest))]
                features = chars_idx + rest_idx
                x_i_idx.append(tuple(features))
            x_idx.append(x_i_idx)
        # return (b, s, f)
        return torch.tensor(x_idx)

    def to_idx_y(self, y: Y) -> torch.Tensor:
        # TODO: Remove UNK when we can express all the tags
        # UNK is temproary supported.
        return torch.tensor([[self.t_map.w2i[t] if t in self.t_map.w2i else UNK_ID for t in y_i] for y_i in y])

    def in_x_y_batches(self,
                       x: In,
                       y: List[SentTags],
                       batch_size=32,
                       shuffle=True,
                       device=None) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        if shuffle:
            x_y = list(zip(x, y))
            random.shuffle(x_y)
            x, y = tuple(map(list, zip(*x_y)))  # type: ignore

        # Correct format
        x_f = self.make_x(x)
        y_f = self.make_y(y)

        # Make batches
        length = len(x_f)
        for ndx in range(0, length, batch_size):
            # First pad, then map to index
            yield (self.to_idx_x(self.pad_x(x_f[ndx:min(ndx + batch_size, length)])).to(device),
                   self.to_idx_y(self.pad_y(y_f[ndx:min(ndx + batch_size, length)])).to(device))

    def in_x_batches(self,
                     x: In,
                     batch_size=32,
                     device=None) -> Iterable[torch.Tensor]:
        x_f = self.make_x(x)
        length = len(x_f)
        for ndx in range(0, length, batch_size):
            # First pad, then map to index
            yield self.to_idx_x(self.pad_x(x_f[ndx:min(ndx + batch_size, length)])).to(device)


def unk_analysis(train: Vocab, test: Vocab):
    log.info(f'Train len={len(train)}')
    log.info(f'Test len={len(test)}')
    log.info(f'Test not in train={len(test-train)}')
