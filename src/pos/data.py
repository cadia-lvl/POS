"""Data preparation and reading."""
from enum import Enum
from typing import (
    List,
    Any,
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
from functools import partial, reduce
from operator import add
import logging
from copy import deepcopy
from datetime import datetime
import random

from tqdm import tqdm
import numpy as np
from torch import Tensor, stack, no_grad, from_numpy, device as t_device
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

from .model import (
    copy_into_larger_tensor,
    PretrainedEmbedding,
    FlairTransformerEmbedding,
    ClassingWordEmbedding,
    CharacterAsWordEmbedding,
    Tagger,
    Encoder,
    ABLTagger,
)
from .utils import read_tsv
from .core import (
    Vocab,
    VocabMap,
    Tokens,
    Tags,
    SequenceTaggingDataset,
    TokenizedDataset,
    Modules,
)


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


def map_to_index(sentence: Tokens, w2i: Dict[str, int]) -> Tensor:
    """Map a sequence to indices."""
    return Tensor(
        [w2i[token] if token in w2i else w2i[UNK] for token in sentence]
    ).long()


def get_input_mappings(
    dictionaries: Dict[Modules, Optional[VocabMap]]
) -> Dict[Modules, Callable[[Tokens], Union[Tensor, Tokens]]]:
    """Return the input mappings for the model."""
    mappings: Dict[Modules, Callable[[Tokens], Union[Tensor, Tokens]]] = {}
    for module, value in dictionaries.items():
        if module == Modules.BERT:
            mappings[module] = lambda x: x  #  No-op
        elif module == Modules.WordEmbeddings:
            mappings[module] = partial(map_to_index, w2i=value.w2i)  # type: ignore
        else:
            log.info(f"Skipping module={module} in input mappings.")
    return mappings


def get_target_mappings(
    dictionaries: Dict[Modules, Optional[VocabMap]]
) -> Dict[Modules, Callable[[Tokens], Tensor]]:
    """Return the target mappings for the model."""
    mappings: Dict[Modules, Callable[[Tokens], Tensor]] = {}
    for module, value in dictionaries.items():
        if module == Modules.FullTag:
            mappings[module] = partial(map_to_index, w2i=value.w2i)  # type: ignore
        else:
            log.info(f"Skipping module={module} in target mappings.")
    return mappings


def character_mapping(w2i, batch_x):
    """Old code for character mapping."""
    sents_padded = []
    for sent in batch_x:
        sents_padded.append(
            pad_sequence(
                [
                    Tensor(
                        [w2i[SOS]]
                        + [w2i[char] if char in w2i else w2i[UNK] for char in token]
                        + [w2i[EOS]]
                    )
                    for token in sent
                ],
                batch_first=True,
                padding_value=w2i[PAD],
            )
        )
    max_words = max((t.shape[0] for t in sents_padded))
    max_chars = max((t.shape[1] for t in sents_padded))
    sents_padded = [
        copy_into_larger_tensor(t, t.new_zeros(size=(max_words, max_chars)))
        for t in sents_padded
    ]
    return pad_sequence(sents_padded, batch_first=True, padding_value=w2i[PAD])


def batch_preprocess(
    batch: Sequence[Tuple[Tokens, Tags]],
    x_mappings: Dict[Modules, Callable[[Tokens], Union[Tensor, Tokens]]],
    y_mappings: Dict[Modules, Callable[[Tokens], Union[Tensor, Tokens]]],
    device=None,
) -> Dict[Modules, Union[Tensor, Sequence[Tokens]]]:
    """Batch collate function. It takes care of creating batches and preprocess them.

    Returns:
        A dictionary of Tensors
    """
    if not device:
        device = t_device("cpu")
    tokens, tags = SequenceTaggingDataset(batch).unpack()
    processed = batch_preprocess_input(tokens, x_mappings, device)
    processed.update(batch_preprocess_input(tags, y_mappings, device))
    return processed


def batch_preprocess_input(
    batch: Sequence[Tokens],
    mappings: Dict[Modules, Callable[[Tokens], Union[Tensor, Tokens]]],
    device,
) -> Dict[Modules, Union[Tensor, Sequence[Tokens]]]:
    """Process a batch of inputs."""
    processed: Dict[Modules, Union[Tensor, Sequence[Tokens]]] = {}
    for key, mapping in mappings.items():
        if key == Modules.BERT:
            processed[key] = batch
        else:
            processed[key] = pad_sequence(
                [mapping(x) for x in batch], batch_first=True
            ).to(device)
    # Also add the lengths
    processed[Modules.Lengths] = Tensor([len(x) for x in batch]).long().to(device)
    return processed


def pad_dict_tensors(batch_of_dicts: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Pad the tensors in a dict."""
    # Convert List[Dict] to Dict[List]
    result = {}
    a_dict = {key: [d[key] for d in batch_of_dicts] for key in batch_of_dicts[0]}
    for key, value in a_dict.items():
        result[key] = pad_sequence(value, batch_first=True)
    return result


def get_encoder(modules: Dict[Modules, Module]) -> Encoder:
    """Return an Encoder based on the modules available."""
    return Encoder(
        transformer_embedding=modules.get(Modules.BERT, None),
        morphlex_embedding=modules.get(Modules.MorphLex, None),
        pretrained_word_embedding=modules.get(Modules.Pretrained, None),
        word_embedding=modules.get(Modules.WordEmbeddings, None),
        chars_as_word_embedding=modules.get(Modules.CharsAsWord, None),
    )


def get_tagger(
    encoder_output_dim: int, dictionaries: Dict[Modules, VocabMap]
) -> Tagger:
    """Return a Tagger."""
    return Tagger(
        input_dim=encoder_output_dim, output_dim=len(dictionaries[Modules.FullTag].w2i),
    )


def run_batch(
    model: Module, batch: Dict[Modules, Tensor], criterion=None, optimizer=None,
) -> Tuple[Tensor, float]:
    """Run a batch through the model.
    
    If criterion is given, it will be applied and returned (as float).
    If optimizer is given, it will be used to update parameters in conjunction with the criterion.
    """
    if optimizer is not None:
        optimizer.zero_grad()
    model_out = model(batch)
    # (b, seq, tag_features)
    loss = 0.0
    if criterion is not None and Modules.FullTag in batch:
        t_loss = criterion(
            model_out.view(-1, model_out.shape[-1]), batch[Modules.FullTag].view(-1)
        )
        if optimizer is not None:
            t_loss.backward()
            clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        loss = t_loss.item()
    return model_out, loss


def tag_batch(
    model: Module,
    batch: Dict[Modules, Tensor],
    tag_map: VocabMap,
    criterion=None,
    optimizer=None,
) -> Tuple[Iterable[Sequence[str]], float]:
    """Tag (apply POS) on a given data set."""
    preds, loss = run_batch(model, batch, criterion, optimizer)
    idxs = preds.argmax(dim=2).tolist()

    tags = [
        tuple(
            tag_map.i2w[tag_idx]
            for token_count, tag_idx in enumerate(sent)
            # All sentences are padded (at the right end) to be of equal length.
            # We do not want to return tags for the paddings.
            # We check the information about lengths and paddings.
            if token_count < batch[Modules.Lengths][sent_idx]
        )
        for sent_idx, sent in enumerate(idxs)
    ]
    return tags, loss


def tag_data_loader(
    model: Module, data_loader: DataLoader, tag_map: VocabMap, criterion=None,
) -> Tuple[List[Sequence[str]], float]:
    """Tag (apply POS) on a given data set. Sets the model to evaluation mode."""
    tags: List[Sequence[str]] = []
    loss = 0.0
    model.eval()
    with no_grad():
        start = datetime.now()
        for batch in data_loader:
            b_tags, b_loss = tag_batch(model, batch, tag_map, criterion)
            loss += b_loss
            tags.extend(b_tags)
        end = datetime.now()
    log.info(f"Tagged {sum((1 for sent in tags for token in sent))} tokens")
    log.info(f"Tagging took={end-start} seconds")
    return tags, loss


def read_datasets(
    file_paths: List[str], max_sent_length=0, max_lines=0
) -> SequenceTaggingDataset:
    """Read tagged datasets from multiple files.
    
    Args:
        max_sent_length: Sentences longer than "max_sent_length" are thrown away.
        max_lines: Will only keep the first "max_lines" sentences.
    """
    ds = reduce(
        add,
        (
            SequenceTaggingDataset.from_file(training_file)
            for training_file in file_paths
        ),
        SequenceTaggingDataset(tuple()),
    )
    if max_sent_length:
        # We want to filter out sentences which are too long (and throw them away, for now)
        ds = SequenceTaggingDataset(
            [(x, y) for x, y in ds if len(x) <= max_sent_length]
        )
    # DEBUG - read a subset of the data
    if max_lines:
        ds = ds[:max_lines]
    return ds


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
    lines: Iterable[str], f: Callable[[str], Tuple[str, List[float]]]
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
                words_to_add.add(filter_word)
    else:
        words_to_add.update(embedding_dict.keys())
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
    m_embedding = from_numpy(m_embedding).float()
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


def vocab_map_from_dataset(dataset: TokenizedDataset) -> VocabMap:
    """Create a VocabMap given a Dataset."""
    return VocabMap(
        Vocab.from_symbols((x for x in dataset)),
        special_tokens=[(PAD, PAD_ID), (UNK, UNK_ID)],
    )


def load_modules(
    train_ds,
    pretrained_word_embeddings_file=None,
    bert_encoder=None,
    word_embedding_dim=0,
    morphlex_embeddings_file=None,
    known_chars_file=None,
    **kwargs,
):
    """Load all the modules for the model."""
    modules: Dict[Modules, Module] = {}
    dictionaries: Dict[Modules, VocabMap] = {}

    # Pretrained
    if pretrained_word_embeddings_file:
        m_map, m_embedding = read_pretrained_word_embeddings(
            pretrained_word_embeddings_file
        )
        modules[Modules.Pretrained] = PretrainedEmbedding(m_embedding)
        dictionaries[Modules.Pretrained] = m_map

    # pretrained BERT like model, we use it.
    if bert_encoder:
        modules[Modules.BERT] = FlairTransformerEmbedding(bert_encoder, **kwargs)
        dictionaries[Modules.BERT] = None

    # This breaks the pattern a bit...
    w_map = vocab_map_from_dataset((x for x, y in train_ds))
    dictionaries[Modules.WordEmbeddings] = w_map
    if word_embedding_dim:
        # The word embedding dimension is not -1 we train word-embeddings from scratch.
        modules[Modules.WordEmbeddings] = ClassingWordEmbedding(
            len(w_map), word_embedding_dim
        )

    # MorphLex
    if morphlex_embeddings_file:
        # File is provided, use it.
        m_map, m_embedding = read_morphlex(morphlex_embeddings_file)
        modules[Modules.MorphLex] = PretrainedEmbedding(
            m_embedding, freeze=True
        )  # Currently hard-coded to "freeze"
        dictionaries[Modules.MorphLex] = m_map

    # Character embeddings.
    if known_chars_file:
        char_vocab = Vocab.from_file(known_chars_file)
        c_map = VocabMap(
            char_vocab,
            special_tokens=[
                (UNK, UNK_ID),
                (PAD, PAD_ID),
                (EOS, EOS_ID),
                (SOS, SOS_ID),
            ],
        )
        modules[Modules.CharsAsWord] = CharacterAsWordEmbedding(len(c_map))
        dictionaries[Modules.CharsAsWord] = c_map

    # TAGS (POS)
    t_map = vocab_map_from_dataset((y for _, y in train_ds))
    dictionaries[Modules.FullTag] = t_map
    return modules, dictionaries
