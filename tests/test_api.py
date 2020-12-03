"""Test api.py functionality."""
import pytest

import pos
from pos.core import FieldedDataset


def test_tagger(tagger, dictionaries):
    """Test all methods of the Tagger."""
    if not tagger or not dictionaries:
        pytest.skip("No --tagger or --dictionaries given")
    # Initialize the tagger
    tagger = pos.Tagger(
        model_file=tagger,
        device="cpu",
    )
    # Tag a single sentence
    tags = tagger.tag_sent(("Þetta", "er", "setning", "."))
    assert tags == ("fahen", "sfg3en", "nven", "pl")

    # Tag a correctly formatted file.
    dataset = FieldedDataset.from_file("tests/test.tsv", fields=("Tokens",))
    tags = tagger.tag_bulk(dataset=dataset)
    print(tags)
    assert tags == (("au",), ("fahen", "sfg3en", "nhen"), ("au", "aa"))
