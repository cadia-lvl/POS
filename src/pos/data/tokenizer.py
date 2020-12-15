"""Tokenizer."""

from transformers import AutoTokenizer, PreTrainedTokenizer


def load_tokenizer(directory: str) -> PreTrainedTokenizer:
    """Load a subword tokenizer from a directory."""
    return AutoTokenizer.from_pretrained(directory)  # type: ignore
