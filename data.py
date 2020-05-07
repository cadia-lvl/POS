from collections import Counter, defaultdict
from itertools import count
import typing
from typing import List, Tuple, Set, Union

from tqdm import tqdm
import numpy as np

TaggedToken = Tuple[str, str]
# When iterating over Sent-x, whole sentences are yielded
SentTokens = Tuple[str, ...]
SentTags = Tuple[str, ...]
# Either tokens or tags, we don't care
Sent = Tuple[str, ...]
Vocab = Set[str]
DataPairs = Tuple[List[SentTokens], List[SentTags]]
TaggedSentence = List[TaggedToken]
Corpus = List[TaggedSentence]


def read_tsv(input) -> Corpus:
    corpus = []
    with open(input) as f:
        tokens: TaggedSentence = []
        for line in f:
            line = line.strip()
            # We read a blank line - sentence has been read.
            if not line:
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
    tokens = tags = []
    for tagged_sentence in corpus:
        sent_tokens = sent_tags = []
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


class VocabMap:
    def __init__(self, vocab: Vocab):
        w2i: typing.DefaultDict[str, int] = defaultdict(count(0).__next__)
        [w2i[v] for v in vocab]
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.items()}

    def size(self):
        return len(self.w2i.keys())


class Embeddings:
    """Reads a file, which defines a mapping from str->vector"""
    def __init__(self, emb_file):
        print(f"Reading embeddings={emb_file}")
        self.vocab = {}
        matrix = []
        with open(emb_file) as f:
            for i, line in enumerate(tqdm(f)):
                key, vector = line.strip().split(";")
                self.vocab[key] = i
                # It is slightly insane to use eval, but let's do it.
                matrix.append(eval(vector))
        self.embeddings = np.array(matrix)
