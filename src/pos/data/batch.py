"""Batch processing."""

from typing import Any, Dict, Sequence, Tuple

from pos import core
from pos.constants import BATCH_KEYS, EOS, PAD, PAD_ID, SOS, UNK
from pos.core import Dicts, FieldedDataset, Fields, Sentence, Vocab, VocabMap
from torch import Tensor, zeros_like
from torch.nn.utils.rnn import pad_sequence

from .pretrained import read_morphlex, read_pretrained_word_embeddings


def map_to_index(sentence: Sentence, w2i: Dict[str, int]) -> Tensor:
    """Map a sequence to indices."""
    return Tensor([w2i[token] if token in w2i else w2i[UNK] for token in sentence]).long().to(core.device)


def map_to_chars_and_index(sentence: Sentence, w2i: Dict[str, int], add_eos=True, add_sos=True) -> Tensor:
    """Map a sequence to characters then to indices."""
    SOS_l = [w2i[SOS]] if add_sos else []
    EOS_l = [w2i[EOS]] if add_eos else []
    # (tokens, chars)
    return pad_sequence(
        [
            Tensor(SOS_l + [w2i[char] if char in w2i else w2i[UNK] for char in token] + EOS_l).long()
            for token in sentence
        ],
        batch_first=True,
        padding_value=w2i[PAD],
    ).to(core.device)


def map_to_chars_batch(sentences: Sequence[Sentence], w2i: Dict[str, int], add_eos=True, add_sos=True) -> Tensor:
    """Map a batch of sentences to characters of words. This is convoluted, I know."""
    sents_padded = []
    for sentence in sentences:
        sents_padded.append(map_to_chars_and_index(sentence, w2i, add_eos=add_eos, add_sos=add_sos))
    max_words = max((t.shape[0] for t in sents_padded))
    max_chars = max((t.shape[1] for t in sents_padded))
    sents_padded = [copy_into_larger_tensor(t, t.new_zeros(size=(max_words, max_chars))) for t in sents_padded]
    # (b * tokens, chars)
    return (
        pad_sequence(sents_padded, batch_first=True, padding_value=w2i[PAD])
        .reshape(shape=(-1, max_chars))
        .long()
        .to(core.device)
    )


def map_to_index_batch(sentences: Sequence[Sentence], w2i: Dict[str, int]) -> Tensor:
    """Map to index, batch."""
    return pad_sequence(
        [map_to_index(sentence=sentence, w2i=w2i) for sentence in sentences],
        batch_first=True,
        padding_value=w2i[PAD],
    ).to(core.device)


def copy_into_larger_tensor(tensor: Tensor, like_tensor: Tensor) -> Tensor:
    """Create a larger tensor based on given tensor. Only works for 2-dims."""
    base = zeros_like(like_tensor)
    base[: tensor.shape[0], : tensor.shape[1]] = tensor
    return base


def load_dicts(
    train_ds: FieldedDataset,
    ignore_e_x=False,
    pretrained_word_embeddings_file=None,
    morphlex_embeddings_file=None,
    known_chars_file=None,
):
    """Load all the modules for the model."""
    embeddings: Dict[Dicts, Tensor] = {}
    dictionaries: Dict[Dicts, VocabMap] = {}

    # Pretrained
    if pretrained_word_embeddings_file:
        m_map, m_embedding = read_pretrained_word_embeddings(pretrained_word_embeddings_file)
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
    c_map = VocabMap(
        char_vocab,
        special_tokens=VocabMap.UNK_PAD_EOS_SOS,
    )
    dictionaries[Dicts.Chars] = c_map

    # TAGS (POS)
    special_tokens = VocabMap.UNK_PAD
    tags = train_ds.get_vocab(field=Fields.GoldTags)
    if ignore_e_x:
        special_tokens.append(("e", PAD_ID))
        tags.remove("e")
        special_tokens.append(("x", PAD_ID))
        tags.remove("x")
    dictionaries[Dicts.FullTag] = VocabMap(tags, special_tokens=VocabMap.UNK_PAD)
    return embeddings, dictionaries
