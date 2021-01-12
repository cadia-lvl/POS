"""Dataset manipulation."""


from functools import reduce
from itertools import chain
from operator import add
from re import sub
from typing import List, Tuple, cast
import logging

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pos.data.tokenizer import load_tokenizer, get_initial_token_mask
from pos.core import FieldedDataset, Fields, Sentence, Sentences

log = logging.getLogger(__name__)


def read_datasets(
    file_paths: List[str],
    fields=None,
) -> FieldedDataset:
    """Read tagged datasets from multiple files.

    Args:
        file_paths: The paths to the datasets.
        fields: The tagged fields in the dataset
    """
    return reduce(
        add,
        (
            FieldedDataset.from_file(training_file, fields)
            for training_file in file_paths
        ),
    )


def get_adjusted_lengths(
    sentences: Sentences,
    tokenizer: PreTrainedTokenizer,
    max_sequence_length,
) -> Tuple[int]:
    """Return adjusted lengths based on a tokenizer and model max length."""
    encodings = [
        tokenizer.encode_plus(
            sentence, is_split_into_words=True, return_offsets_mapping=True
        )
        for sentence in sentences
    ]
    # Create end-token masks: [CLS] Hauk ur er [SEP] -> [dropped, 0, 1, 1, dropped]
    # By getting  initial token masks and shifting them:
    # [CLS] Hauk ur er [SEP] -> [0, 1, 0, 1, 0] ->
    # -> drop [mid shifted to left] + [1] drop
    # -> [_, 0, 1, 1, _]
    end_token_masks = [
        get_initial_token_mask(encoded["offset_mapping"])[2:-1] + [1]
        for encoded in encodings
    ]
    # We need to account for SEP and CLS when finding the cuts
    max_sequence_length -= 2
    # And some extra, because of errors
    max_sequence_length -= 4
    lengths = []
    for end_token_mask in end_token_masks:
        while len(end_token_mask) != 0:
            prefix, end_token_mask = (
                end_token_mask[:max_sequence_length],
                end_token_mask[max_sequence_length:],
            )
            length = sum(prefix)
            lengths.append(length)

    return tuple(int(length) for length in lengths)


def chunk_dataset(
    ds: FieldedDataset, tokenizer: PreTrainedTokenizer, max_sequence_length
) -> FieldedDataset:
    """Split up sentences which are too long."""
    log.info("Splitting sentences in order to fit BERT-like model")
    tokens = ds.get_field()
    lengths = get_adjusted_lengths(
        tokens, tokenizer, max_sequence_length=max_sequence_length
    )
    return ds.adjust_lengths(lengths, shorten=True)


def dechunk_dataset(
    original_ds: FieldedDataset, chunked_ds: FieldedDataset
) -> FieldedDataset:
    """Reverse the chunking from the original dataset."""
    log.info("Reversing the splitting of sentences in order to fit BERT-like model")
    original_lengths = original_ds.get_lengths()
    return chunked_ds.adjust_lengths(original_lengths, shorten=False)
