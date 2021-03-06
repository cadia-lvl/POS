"""An Icelandic POS tagger."""
import logging

from .core import FieldedDataset, Fields
from .api import Tagger

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
