"""Flair word/character embeddings."""
from torch import stack
from flair.data import Sentence
from flair.embeddings import (
    TransformerWordEmbeddings,
    FlairEmbeddings,
    StackedEmbeddings,
)

from .types import Symbols

transformer = None


def electra_embedding(tokens: Symbols):
    """Create Electra embeddings from tokens."""
    # init embedding
    global transformer
    if transformer is None:
        transformer = TransformerWordEmbeddings("electra_model/", layers="-1")

    # create a sentence
    sentence = Sentence(" ".join(tokens))

    # embed words in sentence
    transformer.embed(sentence)
    return stack(
        [token.embedding for token in sentence]
    )  # pylint: disable=not-callable


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
