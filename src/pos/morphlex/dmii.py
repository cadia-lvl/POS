"""The database of morphological icelandic inflections wrapper."""
from typing import Tuple
from reynir.bindb import BIN_Db

from .interface import MorphologicalLexicon, Tag, Lemma


class DMII(MorphologicalLexicon):
    """The lookup_word implementation."""

    def __init__(self) -> None:
        """Initialize the BIN DB."""
        super().__init__()
        self.db = BIN_Db()

    def lookup_word(self, word: str) -> Tuple[Tuple[Tag, Lemma], ...]:
        """Return the tag as '' and corresponding lemma."""
        # returns ('the query string', [(stofn='lemma', ordfl='word_category', fl='category' utg='id', ordmynd='word_form', inflection_str)])
        _, results = self.db.lookup_word(word)
        return tuple(
            (result.ordfl + result.fl + result.beyging, result.stofn)
            for result in results
        )
