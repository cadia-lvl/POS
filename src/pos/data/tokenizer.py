"""Tokenizer."""

from re import sub
from typing import Callable, List, Set
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    RobertaTokenizer,
    BertTokenizer,
    ElectraTokenizer,
    ElectraTokenizerFast,
)


def load_tokenizer(directory: str) -> PreTrainedTokenizer:
    """Load a subword tokenizer from a directory."""
    return AutoTokenizer.from_pretrained(directory)  # type: ignore


def is_initial(
    sub_token: str, special_str: str, denotes_initial: bool, special_tokens: Set[str]
) -> bool:
    """Return True if subtoken is partial."""
    special_found = sub_token.startswith(special_str)
    if special_found and denotes_initial:
        return True
    elif special_found and not denotes_initial:
        return False
    elif not special_found and denotes_initial:
        return False
    else:  # not found and not denotes_initial -
        # Could also be a special token - that is not an initial token
        if sub_token in special_tokens:
            return False
        return True


def get_is_initial(tokenizer: PreTrainedTokenizer) -> Callable[[str], bool]:
    """Return the special character used by the tokenizer and wether it denotes a partial token."""
    special_str = ""
    if tokenizer.__class__ == RobertaTokenizer:
        special_str, denotes_initial = "Ġ", True
    elif tokenizer.__class__ in {BertTokenizer, ElectraTokenizer, ElectraTokenizerFast}:
        # '##' denotes a partial string
        special_str, denotes_initial = "##", False
    else:
        raise ValueError(f"Unkown tokenizer={tokenizer.__class__}")
    special_tokens = set(tokenizer.all_special_tokens)

    def is_inital_closure(sub_token: str):
        return is_initial(
            sub_token=sub_token,
            special_str=special_str,
            denotes_initial=denotes_initial,
            special_tokens=special_tokens,
        )

    return is_inital_closure


def get_initial_token_mask(
    tokenizer: PreTrainedTokenizer, sub_tokenized_sentence: List[int]
):
    """Return the inital token masks for subtokenized tokens. Special tokens are not considered inital."""
    is_initial = get_is_initial(tokenizer)
    return [1 if is_initial(sub_token) else 0 for sub_token in sub_tokenized_sentence]
