"""Tokenizer."""

from typing import List, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(directory: str) -> PreTrainedTokenizerFast:
    """Load a subword tokenizer from a directory."""
    return AutoTokenizer.from_pretrained(directory)  # type: ignore


def tok_space_added(tokenizer: PreTrainedTokenizerFast) -> bool:
    """Return True if tokenizer is set to add_prefix_space."""
    if hasattr(tokenizer, "add_prefix_space"):
        return tokenizer.__getattribute__("add_prefix_space")
    return False


def get_initial_token_mask(
    offsets_mapping: List[Tuple[int, int]], contains_bos_eos=True
):
    """Return the inital token masks for subword tokens. Special tokens are not considered inital."""
    initial_token_masks = []
    if contains_bos_eos:
        offsets_mapping = offsets_mapping[1:-1]
        initial_token_masks.append(0)
    last_start, last_end = None, None
    for start, end in offsets_mapping:
        if last_start is None:
            # First token
            initial_token_masks.append(1)
        elif last_end == start:
            # Continuation of previous token
            initial_token_masks.append(0)
        elif start == end == 0:
            # When padding to maximum length, most offsets are (0,0)
            initial_token_masks.append(0)
        elif last_end == end and last_start == start:
            # RoBERTa special, sometimes we get two subword tokens from a single character.
            initial_token_masks.append(0)
        else:
            # Otherwise it's initial
            initial_token_masks.append(1)
        last_end = end
        last_start = start
    if contains_bos_eos:
        initial_token_masks.append(0)
    return initial_token_masks
