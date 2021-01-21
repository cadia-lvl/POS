"""Constants used in data processing."""
from enum import Enum


# To pad in batches
PAD = "<pad>"
PAD_ID = 0
# For unkown words in testing
UNK = "<unk>"
UNK_ID = 1
# For EOS and SOS in char BiLSTM
EOS = "</s>"
EOS_ID = 2
SOS = "<s>"
SOS_ID = 3


class BATCH_KEYS(Enum):
    """Keys used on the batch dictionary."""

    TOKENS = "tokens"
    FULL_TAGS = "full_tags"
    TARGET_FULL_TAGS = "target_full_tags"
    LEMMAS = "lemmas"
    TARGET_LEMMAS = "target_lemmas"
    LENGTHS = "lens"
    TOKEN_CHARS_LENS = "token_char_lens"