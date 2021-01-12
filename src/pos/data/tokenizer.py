"""Tokenizer."""

from typing import List, Tuple
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)


def load_tokenizer(directory: str) -> PreTrainedTokenizerFast:
    """Load a subword tokenizer from a directory."""
    return AutoTokenizer.from_pretrained(directory)  # type: ignore


def is_initial(offset_mapping: Tuple[int, int]) -> bool:
    """Return True if offset mapping represents start of token."""
    char_start, char_end = offset_mapping
    # start of a new token
    if char_start == 0:
        # ends at same place = special added token -> not initial
        if char_end == 0:
            return False
        # does not end in same place -> Initial
        return True
    # represents characters inside a token
    return False


def get_initial_token_mask(offsets_mapping: List[Tuple[int, int]]):
    """Return the inital token masks for subtokenized tokens. Special tokens are not considered inital."""
    return [
        1 if is_initial(offset_mapping) else 0 for offset_mapping in offsets_mapping
    ]
