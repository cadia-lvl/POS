from dataclasses import dataclass
from collections import Counter, defaultdict
from itertools import count
import typing
from typing import List, Tuple, Set, Dict

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

    def __init__(self, vocab: Vocab, build_from: Dict[str, int] = None):
        """
        Builds a vocabulary mapping from the provided vocabulary, starting at index=0
        build_from: If provided will use previously defined mappings and extend them with
        new words in vocab starting at index=len(build_from)
        """
        if build_from is None:
            build_from = dict()
        # Find all missing words
        missing_words = set(vocab) - set(build_from.keys())
        # Copy the contents and set the counter accordingly
        w2i_tmp: typing.DefaultDict[str, int] = defaultdict(count(len(build_from.keys())).__next__, build_from)
        [w2i_tmp[v] for v in missing_words]
        self.w2i = dict(w2i_tmp)
        self.i2w = {i: w for w, i in self.w2i.items()}

    def __len__(self):
        return len(self.w2i)


def read_embedding(emb_file, from_vocab_map: VocabMap, add_all=False) -> Tuple[VocabMap, np.array]:
    """
    Reads an embedding file and constructs an embedding using np.array().
    from_vocab_map: Uses the same ids as given in the mapping
    add_all: If set True will add all remaining tokens to the embedding.
    Returns: The embeddings as np.array and VocabMap for the embedding.
    """
    print(f'Embedding reading={emb_file}. Based on #tokens={len(from_vocab_map)}')
    # This can be huge
    embedding_dict: Dict[str, List[int]] = dict()
    with open(emb_file) as f:
        for line in tqdm(f):
            key, vector = line.strip().split(";")
            # We stip out '[' and ']'
            embedding_dict[key] = [int(n) for n in vector[1:-1].split(',')]
    # The embedding matrix only based on the vocabmap
    length_of_embeddings = len(list(embedding_dict.values())[0])
    embeddings = np.zeros(shape=(len(from_vocab_map), length_of_embeddings))
    # We go through the vocabmap in order
    added_words = set()
    for idx in range(len(from_vocab_map)):
        word_in_vocab_map = from_vocab_map.i2w[idx]
        # If the word is in the embedding dict we want to use that embedding
        if word_in_vocab_map in embedding_dict:
            added_words.add(word_in_vocab_map)
            embeddings[idx] = embedding_dict[word_in_vocab_map]
        else:
            # We leave the embedding as zero
            pass
    # Should we add the rest of the words?
    if add_all:
        missing_words = set(embedding_dict.keys()) - added_words
        embeddings_missing_from_vocab = np.zeros(shape=(len(missing_words), length_of_embeddings))
        print(f'Embedding: adding missing words, len={len(missing_words)}')
        for idx, missing_word in enumerate(missing_words):
            added_words.add(missing_word)
            embeddings_missing_from_vocab[idx] = embedding_dict[missing_word]
        embeddings = np.concatenate([embeddings, embeddings_missing_from_vocab], axis=0)
    print(f'Embedding: final shape={embeddings.shape}')
    return VocabMap(added_words, build_from=from_vocab_map.w2i), embeddings


class Embeddings:
    # Just here for good-old times
    def __init__(self, vocabmap: VocabMap, embedding: np.array):
        self.vocab = vocabmap.w2i
        self.embeddings = embedding
