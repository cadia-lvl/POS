"""Test api.py functionality."""
import pos
from pos.core import FieldedDataset


def test_tagger(pretrained_tagger):
    """Test all methods of the Tagger."""
    # Initialize the tagger
    tagger = pos.Tagger(
        model_file=pretrained_tagger,
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
