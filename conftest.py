"""Fixtures for tests."""
from pytest import fixture
from typing import Dict

from torch.utils.data.dataloader import DataLoader
from pos.cli import MORPHLEX_VOCAB_PATH, PRETRAINED_VOCAB_PATH

from pos.core import Fields, Vocab, VocabMap, Dicts, FieldedDataset
from pos.data import collate_fn, load_dicts
from pos.evaluate import Experiment
from pos.model import ABLTagger, Encoder, ClassingWordEmbedding, Tagger, Modules


def pytest_addoption(parser):
    """Add extra command-line options to pytest."""
    parser.addoption("--tagger", action="store")
    parser.addoption("--dictionaries", action="store")
    parser.addoption("--electra_model", action="store")


@fixture()
def dictionaries(request):
    """Exposes the command-line option to a test case."""
    return request.config.getoption("--dictionaries")


@fixture()
def electra_model(request):
    """Exposes the command-line option to a test case."""
    return request.config.getoption("--electra_model")


@fixture()
def tagger(request):
    """Exposes the command-line option to a test case."""
    return request.config.getoption("--tagger")


@fixture()
def test_tsv_file():
    """Return the filepath of the test tsv file."""
    return "./tests/test.tsv"


@fixture()
def tagged_test_tsv_file():
    """Return the filepath of the test tsv file."""
    return "./tests/test_pred.tsv"


@fixture()
def test_tsv_lemma_file():
    """Return the filepath of the test tsv file."""
    return "./tests/test_lemma.tsv"


@fixture
def ds(test_tsv_file):
    """Return a sequence tagged dataset."""
    return FieldedDataset.from_file(
        test_tsv_file, fields=(Fields.Tokens, Fields.GoldTags)
    )


@fixture
def ds_lemma(test_tsv_lemma_file):
    """Return a sequence tagged dataset."""
    return FieldedDataset.from_file(
        test_tsv_lemma_file, fields=(Fields.Tokens, Fields.GoldTags, Fields.GoldLemmas)
    )


@fixture
def vocab_maps(ds_lemma) -> Dict[Dicts, VocabMap]:
    """Return the dictionaries for the dataset."""
    return load_dicts(ds_lemma)[1]


@fixture()
def data_loader(ds_lemma):
    """Return a data loader over the unit testing data."""
    return DataLoader(ds_lemma, batch_size=3, collate_fn=collate_fn)  # type: ignore


@fixture
def encoder(vocab_maps) -> Encoder:
    """Return an Encoder."""
    wembs = ClassingWordEmbedding(vocab_map=vocab_maps[Dicts.Tokens], embedding_dim=3)
    return Encoder({Modules.Trained: wembs}, main_lstm_layers=1)


@fixture
def tagger_module(vocab_maps, encoder) -> Tagger:
    """Return a Tagger."""
    return Tagger(vocab_map=vocab_maps[Dicts.FullTag], input_dim=encoder.output_dim)


@fixture
def abl_tagger(encoder, tagger_module) -> ABLTagger:
    return ABLTagger(encoder=encoder, decoders={Modules.Tagger: tagger_module})


@fixture
def tagger_evaluator(ds_lemma):
    return Experiment.all_accuracy_closure(
        ds_lemma,
        train_vocab=ds_lemma.get_vocab(),
        morphlex_vocab=Vocab.from_file(MORPHLEX_VOCAB_PATH),
        pretrained_vocab=Vocab.from_file(PRETRAINED_VOCAB_PATH),
    )


@fixture
def kwargs():
    return {
        "tagger": True,
        "lemmatizer": False,
        "tagger_weight": 1,
        "scheduler": "multiply",
        "learning_rate": 5e-5,
        "word_embedding_lr": 0.2,
        "optimizer": "adam",
        "label_smoothing": 0.1,
        "output_dir": "debug/",
        "epochs": 20,
    }
