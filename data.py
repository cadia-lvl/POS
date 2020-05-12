import copy
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Set, Dict, Optional, Union

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
DataPairs_c = Tuple[Tuple[List[SentTokens], List[SentTags]], List[SentTags]]
Data = Union[DataPairs, DataPairs_c]
TaggedSentence = List[TaggedToken]
Corpus = List[TaggedSentence]

TrainW = Tuple[List[str], str, str]
TrainSent = List[TrainW]
TrainWidx = Tuple[List[int], int, int]
TrainSentidx = List[TrainWidx]
# For unkown words in testing
UNK = '<unk>'
UNK_ID = 0
# To pad in batches
PAD = '<pad>'
PAD_ID = 1
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
    print(f'Embedding reading={emb_file}')
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
        print(f'Filtering on #symbols={len(filter_on)}')
        for filter_word in filter_on:
            # If the word is present in the file we use it.
            if filter_word in embedding_dict:
                words_to_add.add(filter_word)
    else:
        words_to_add = set(embedding_dict.keys())
    # + special tokens
    embeddings = np.zeros(shape=(len(words_to_add) + len(special_tokens), length_of_embeddings))

    vocab_map = VocabMap(words_to_add, special_tokens=special_tokens)
    for symbol, idx in vocab_map.w2i.items():
        embeddings[idx] = embedding_dict[symbol]

    print(f'Embedding: final shape={embeddings.shape}')
    return vocab_map, embeddings


class Embeddings:
    # Just here for good-old times
    def __init__(self, vocabmap: VocabMap, embedding: np.array):
        self.vocab = vocabmap.w2i
        self.embeddings = embedding


def unk_analysis(train: Vocab, test: Vocab):
    print(f'Train len={len(train)}')
    print(f'Test len={len(test)}')
    print(f'Test not in train={len(test-train)}')


def create_example_token(token: str) -> TrainW:
    # ([SOS + characters + EOS], word, morph)
    return ([SOS] + [c for c in token] + [EOS], token, token)


def create_example_sent(example: SentTokens) -> List[TrainW]:
    # We do not put SOS or EOS on words
    return [create_example_token(tok) for tok in example]


def create_examples(data: List[SentTokens]) -> List[TrainSent]:
    return [create_example_sent(sent) for sent in data]


def longest_token_in_sent(example: TrainSentidx) -> int:
    return max(len(chars) for chars, _, _ in example)


def pad_examples_(data: List[TrainSentidx], pad_idx: int = PAD_ID):
    """ Pads characters and sentences in place.
    """
    longest_token_in_examples = max(longest_token_in_sent(sent) for sent in data)
    longest_sent_in_examples = max(len(sent) for sent in data)
    # print(longest_token_in_examples)
    # print(longest_sent_in_examples)
    padded = copy.deepcopy(data)
    for train_sent in padded:
        while len(train_sent) < longest_sent_in_examples:
            empty_chars: List[int] = []
            train_sent.append((list(empty_chars), pad_idx, pad_idx))
        for chars, _, _ in train_sent:
            while len(chars) < longest_token_in_examples:
                chars.append(pad_idx)
    return [[(*chars, w, m) for chars, w, m in sent] for sent in padded]


def to_idx(data: List[TrainSent]):
    pass
