"""Tokenizer."""

from typing import List, Tuple

from transformers import PreTrainedTokenizerFast


def tok_space_added(tokenizer: PreTrainedTokenizerFast) -> bool:
    """Return True if tokenizer is set to add_prefix_space."""
    if tokenizer.__getattribute__("add_prefix_space") is not None:
        return tokenizer.__getattribute__("add_prefix_space")
    return False


def get_initial_token_mask(offsets_mapping: List[Tuple[int, int]]):
    """Return the inital token masks for subword tokens. Special tokens are not considered inital."""
    initial_token_masks = []
    last_end = 0
    for start, end in offsets_mapping:
        if end == start == 0:
            # Special token
            initial_token_masks.append(0)
        elif start == 0 != end:
            initial_token_masks.append(1)
        elif last_end == start:
            initial_token_masks.append(0)
        elif start == end:
            initial_token_masks.append(0)
        else:
            initial_token_masks.append(1)
        last_end = end
    return initial_token_masks
