"""Tokenizer."""

from typing import Callable, Set
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    RobertaTokenizer,
    BertTokenizer,
    ElectraTokenizer,
)


def load_tokenizer(directory: str) -> PreTrainedTokenizer:
    """Load a subword tokenizer from a directory."""
    return AutoTokenizer.from_pretrained(directory)  # type: ignore


def is_partial(sub_token: str, special_str: str, denotes_partial: bool) -> bool:
    """Return True if subtoken is partial."""
    found = special_str in sub_token
    if found and denotes_partial:
        return True
    elif found and not denotes_partial:
        return False
    elif not found and denotes_partial:
        return False
    else:  # not found and not denotes_partial:
        return True


def get_partial(tokenizer: PreTrainedTokenizer) -> Callable[[str], bool]:
    """Return the special character used by the tokenizer and wether it denotes a partial token."""
    special_str = ""
    if tokenizer.__class__ == RobertaTokenizer:
        special_str, denotes_partial = "Ġ", False
    elif tokenizer.__class__ in {BertTokenizer, ElectraTokenizer}:
        special_str, denotes_partial = "##", True
    else:
        raise ValueError(f"Unkown tokenizer={tokenizer.__class__}")

    def is_partial_closure(sub_token: str):
        return is_partial(
            sub_token=sub_token,
            special_str=special_str,
            denotes_partial=denotes_partial,
        )

    return is_partial_closure
