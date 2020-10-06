"""An Icelandic POS tagger."""
import logging

from .types import SimpleDataset, Symbols
from .api import Tagger

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
