"""Fixtures for tests."""
from pytest import fixture
from functools import partial

from pos.core import SequenceTaggingDataset, VocabMap, TokenizedDataset
from pos.data import (
    load_dicts,
    get_input_mappings,
    get_target_mappings,
    batch_preprocess,
)
import torch


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


@fixture
def ds(test_tsv_file):
    """Return a sequence tagged dataset."""
    return SequenceTaggingDataset.from_file(test_tsv_file)


@fixture()
def data_loader(ds):
    """Return a data loader over the unit testing data."""
    _, dicts = load_dicts(ds)
    input_mappings = get_input_mappings(dicts)
    target_mappings = get_target_mappings(dicts)
    collate_fn = partial(
        batch_preprocess, x_mappings=input_mappings, y_mappings=target_mappings
    )
    return torch.utils.data.DataLoader(ds, batch_size=3, collate_fn=collate_fn)

