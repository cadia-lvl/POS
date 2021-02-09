"""An interface for Morphological lexicons."""
import abc
from typing import Sequence, Tuple

Lemma = str
Tag = str


class MorphologicalLexicon(metaclass=abc.ABCMeta):
    """An abstract base class for morphological lexicons."""

    @abc.abstractmethod
    def lookup_word(self, word, *args, **kwargs) -> Sequence[Tuple[Tag, Lemma]]:
        """Return a set of possible tags and lemmas given a word. Return an empty sequence when not found."""
