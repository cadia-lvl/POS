"""An Icelandic POS tagger."""
import logging

from .api import Lemmatizer, Tagger
from .core import FieldedDataset, Fields

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
