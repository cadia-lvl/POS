"""Constants used in data processing."""

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


class BATCH_KEYS:
    """Keys used on the batch dictionary."""

    TOKENS = "tokens"
    TOKEN_IDS = "token_ids"
    FULL_TAGS = "full_tags"
    FULL_TAGS_IDS = "target_full_tags"
    LEMMAS = "lemmas"
    LEMMA_CHAR_IDS = "target_lemmas"
    LENGTHS = "lens"
    TOKEN_CHARS_LENS = "token_char_lens"
    CHAR_IDS = "char_ids"
    PRETRAINED_TOKEN_IDS = "pretrained_token_ids"
    SUBWORDS = "subwords"


class Modules:
    """To hold the module names."""

    Pretrained = "pretrained"
    Trained = "trained"
    MorphLex = "morphlex"
    CharactersToTokens = "chars_to_tok"
    Characters = "chars"
    BiLSTM = "bilstm"
    BERT = "bert"
    Tagger = "tagger"
    Lemmatizer = "character_decoder"
    TagEmbedding = "tag_embedding"
