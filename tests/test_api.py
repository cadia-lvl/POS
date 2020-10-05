"""Test api.py functionality."""
import pytest

import pos
from pos.types import Dataset

MODEL_LOCATION = "tagger.pt"
DICT_LOCATION = "dictionaries.pickle"


@pytest.mark.model
def test_tagger():
    """Test all methods of the Tagger."""
    # Initialize the tagger
    tagger = pos.Tagger(
        model_file=MODEL_LOCATION, dictionaries_file=DICT_LOCATION, device="cpu",
    )
    # Tag a single sentence
    tags = tagger.tag_sent(["Þetta", "er", "setning", "."])
    assert tags == ("fahen", "sfg3en", "nven", "pl")

    # Tag a correctly formatted file.
    dataset = Dataset.from_file("tests/test.tsv").unpack()[0]
    tags = tagger.tag_bulk(dataset=dataset)
    print(tags)
    assert tags == (("au",), ("fahen", "sfg3en", "nhen"), ("au", "aa"))
