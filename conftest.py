"""Fixtures for tests."""
from pytest import fixture
from typing import Dict

import torch

from pos.core import SequenceTaggingDataset, VocabMap, Dicts, FieldedDataset
from pos.data import collate_fn, load_dicts
from pos.model import Encoder, ClassingWordEmbedding, Tagger


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
    return FieldedDataset.from_file(test_tsv_file, fields=["tokens", "tags"])


@fixture
def ds_lemma(test_tsv_lemma_file):
    """Return a sequence tagged dataset."""
    return FieldedDataset.from_file(
        test_tsv_lemma_file, fields=["tokens", "tags", "lemmas"]
    )


@fixture
def vocab_maps(ds_lemma) -> Dict[str, VocabMap]:
    """Return the dictionaries for the dataset."""
    return load_dicts(ds_lemma)[1]


@fixture()
def data_loader(ds_lemma):
    """Return a data loader over the unit testing data."""
    return torch.utils.data.DataLoader(ds_lemma, batch_size=3, collate_fn=collate_fn)


@fixture
def encoder(vocab_maps) -> Encoder:
    """Return an Encoder."""
    wembs = ClassingWordEmbedding(vocab_map=vocab_maps[Dicts.Tokens], embedding_dim=3)
    return Encoder([wembs], main_lstm_layers=1)


@fixture
def tagger_module(vocab_maps, encoder) -> Tagger:
    """Return a Tagger."""
    return Tagger(
        vocab_map=vocab_maps[Dicts.FullTag], input_dim=encoder.output_dim, output_dim=5
    )
