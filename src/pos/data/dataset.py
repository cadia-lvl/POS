"""Dataset manipulation."""


from functools import reduce
from itertools import chain
from operator import add
from re import sub
from typing import List, Tuple, cast
import logging

from transformers.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pos.data.tokenizer import get_partial, load_tokenizer
from pos.core import FieldedDataset, Fields, Sentence, Sentences

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


def get_adjusted_lengths(
    sentences: Sentences,
    tokenizer: PreTrainedTokenizer,
    max_sequence_length,
    account_for_specials=True,
) -> Tuple[int]:
    num_specials = 0
    if account_for_specials:
        num_specials = 2  # Just [SEP] and [CLS]
        max_sequence_length = max_sequence_length - num_specials
    token_ids = tokenizer(
        list(" ".join(sentence) for sentence in sentences), add_special_tokens=False
    )["input_ids"]
    sub_tokens = [tokenizer.convert_ids_to_tokens(sentence) for sentence in token_ids]  # type: ignore
    # Create end-token masks: Hauk ur -> 0, 1
    is_partial_func = get_partial(tokenizer)
    end_token_masks = [list() for _ in range(len(sub_tokens))]
    for sentence, end_token_mask in zip(sub_tokens, end_token_masks):
        s_iter = iter(sentence)
        # Skip first token, we place them after reading next token.
        next(s_iter)
        for sub_token in s_iter:
            is_partial = is_partial_func(sub_token)
            if is_partial:
                end_token_mask.append(0)
            else:
                end_token_mask.append(1)
        # All sentences end with a full word
        end_token_mask.append(1)

    lengths = []
    for idx, end_token_mask in enumerate(end_token_masks):
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
