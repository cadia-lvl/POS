"""Tokenizer."""

from typing import List, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(directory: str) -> PreTrainedTokenizerFast:
    """Load a subword tokenizer from a directory."""
    return AutoTokenizer.from_pretrained(directory)  # type: ignore


def is_initial(offset_mapping: Tuple[int, int], space_added: bool) -> bool:
    """Return True if offset mapping represents start of token."""
    char_start, char_end = offset_mapping
    # start of a new token
    if char_start == 0 or (space_added and char_start == 1):
        # ends at zero = special added token -> not initial
        if char_end == 0:
            return False
        # Ends at something else than 0 -> initial
        return True
    # represents characters inside a token -> not initial
    return False


def tok_space_added(tokenizer: PreTrainedTokenizerFast) -> bool:
    """Return True if tokenizer is set to add_prefix_space."""
    if tokenizer.__getattribute__("add_prefix_space") is not None:
        return tokenizer.__getattribute__("add_prefix_space")
    return False


def get_initial_token_mask(offsets_mapping: List[Tuple[int, int]], space_added: bool):
    """Return the inital token masks for subtokenized tokens. Special tokens are not considered inital."""
    return [1 if is_initial(offset_mapping, space_added=space_added) else 0 for offset_mapping in offsets_mapping]
