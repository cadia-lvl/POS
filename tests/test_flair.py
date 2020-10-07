from os.path import isfile

import pytest
from pos import flair_embeddings
from pos.types import Symbols


def test_electra(electra_model):
    if not electra_model:
        pytest.skip("No --electra_model given")
    test = Symbols(("Þetta", "er", "próf", "."))
    embedding = flair_embeddings.electra_embedding(test)
    print(embedding)
    assert embedding.size() == (4, 256)
