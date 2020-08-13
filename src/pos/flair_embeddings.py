"""Flair word/character embeddings."""
from torch import stack
from flair.data import Sentence
from flair.embeddings import (
    TransformerWordEmbeddings,
    FlairEmbeddings,
    StackedEmbeddings,
)

from .types import Symbols

TRANSFORMER = None


def electra_embedding(tokens: Symbols):
    """Create Electra embeddings from tokens."""
    # init embedding
    global TRANSFORMER
    if TRANSFORMER is None:
        TRANSFORMER = TransformerWordEmbeddings(
            "electra_model/", layers="all", use_scalar_mix=True
        )

    # create a sentence
    length = len(tokens)
    max_seq_len = 128  # This is the maximum sequence length for the model. It uses subwords- so some sequences will be too long.
    embs = []
    for ndx in range(0, length, max_seq_len):
        batch = tokens[ndx : min(ndx + max_seq_len, length)]
        sentence = Sentence(" ".join(batch))
        # embed words in sentence
        TRANSFORMER.embed(sentence)
        embs.extend([token.embedding for token in sentence])
    return stack(embs)  # pylint: disable=not-callable


def train_flair_embeddings(
    corpus_path: str,
    save_to: str,
    forward: bool,
    hidden_size=1024,
    sequence_length=250,
    batch_size=100,
    epochs=10,
):
    """Train Flair embeddings. See https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md."""
    from flair.data import Dictionary
    from flair.models import LanguageModel
    from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

    # are you training a forward or backward LM?
    is_forward_lm = forward

    # load the default character dictionary
    dictionary: Dictionary = Dictionary.load("chars")

    # get your corpus, process forward and at the character level
    corpus = TextCorpus(corpus_path, dictionary, is_forward_lm, character_level=True)

    # instantiate your language model, set hidden size and number of layers
    language_model = LanguageModel(
        dictionary, is_forward_lm, hidden_size=hidden_size, nlayers=1
    )

    # train your language model
    trainer = LanguageModelTrainer(language_model, corpus)

    trainer.train(
        save_to,
        sequence_length=sequence_length,
        mini_batch_size=batch_size,
        max_epochs=epochs,
    )


def flair_embedding():
    stacked_embeddings = StackedEmbeddings(
        [FlairEmbeddings("X-forward"), FlairEmbeddings("X-backward"),]
    )
    sentence = Sentence("Þetta er tilraun .")
    stacked_embeddings.embed(sentence)
    for token in sentence:
        print(token.embedding)
