"""Dataset manipulation."""


from functools import reduce
from operator import add
from re import sub
from typing import List, Tuple, cast
import logging

from transformers.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from pos.data.tokenizer import load_tokenizer

from ..core import FieldedDataset, Fields, Sentence, Sentences

log = logging.getLogger(__name__)


def read_datasets(
    file_paths: List[str],
    fields=None,
) -> FieldedDataset:
    """Read tagged datasets from multiple files.

    Args:
        max_sent_length: Sentences longer than "max_sent_length" are thrown away.
        fields: The tagged fields in the dataset
    """
    return reduce(
        add,
        (
            FieldedDataset.from_file(training_file, fields)
            for training_file in file_paths
        ),
    )


def get_cut_index(
    tokens: Tuple[str, ...], tokenizer: PreTrainedTokenizer, max_sequence_length
) -> int:
    token_ids = tokenizer(
        list(tokens),
        add_special_tokens=False,
        is_split_into_words=True,
    )["input_ids"]
    token_ids = cast(List[int], token_ids)
    if len(token_ids) > max_sequence_length:
        log.debug(f"Subwords too long: {len(token_ids)}")
        sub_tokens = [tokenizer._convert_id_to_token(idx) for idx in token_ids]
        # Create the word masks. 1 means start of word, 0 means continuation.
        word_masks = [
            0 if sub_token.startswith("##") else 1 for sub_token in sub_tokens
        ]
        # We run backwards until we find a start of a word
        index = max_sequence_length
        word_mask = word_masks[index]
        while not word_mask:
            index -= 1
            word_mask = word_masks[index]
            if index < 0:
                raise RuntimeError(
                    f"Unable to split sentence based on subwords: {tokens}={sub_tokens}"
                )
        # We have found a start of word at "index"
        cut_index = index
        num_tokens = sum(word_masks[:cut_index])
        return num_tokens
    else:
        return len(tokens)


def chunk_dataset(
    ds: FieldedDataset, tokenizer: PreTrainedTokenizer, max_sequence_length
) -> FieldedDataset:
    """Split up sentences which are too long."""
    log.info("Splitting sentences in order to fit BERT-like model")
    chunks: List[List[Sentence]] = [list() for _ in range(len(ds.fields))]
    for field_sentences in ds:
        tokens = field_sentences[ds.fields.index(Fields.Tokens)]
        start_index = 0
        while tokens:
            cut_index = get_cut_index(
                tokens, tokenizer, max_sequence_length=max_sequence_length
            )
            cut_fields = tuple(
                field_sentence[start_index : cut_index + start_index]
                for field_sentence in field_sentences
            )
            for a_list, field in zip(chunks, cut_fields):
                a_list.append(field)
            tokens = tokens[cut_index:]
            start_index = cut_index
    return FieldedDataset(tuple(tuple(a_list) for a_list in chunks), ds.fields)


def dechunk_dataset(
    original_ds: FieldedDataset, chunked_ds: FieldedDataset
) -> FieldedDataset:
    """Reverse the chunking from the original dataset."""
    log.info("Reversing the splitting of sentences in order to fit BERT-like model")
    dechunks: List[List[Sentence]] = [list() for _ in range(len(chunked_ds.fields))]
    index = 0
    append = True
    for field_sentences in chunked_ds:
        if append:
            for a_list, field in zip(dechunks, field_sentences):
                a_list.append(field)
        else:
            for a_list, field in zip(dechunks, field_sentences):
                a_list[index] = tuple(a_list[index] + field)
        # The indexing is different in FieldedDataset
        if len(dechunks[0][index]) == len(original_ds[index][0]):
            index += 1
            append = True
        else:
            append = False
    return FieldedDataset(
        tuple(tuple(a_list) for a_list in dechunks), chunked_ds.fields
    )
