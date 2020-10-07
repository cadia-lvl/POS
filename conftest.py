"""Fixtures for tests."""
from pytest import fixture

print("aaa")


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
