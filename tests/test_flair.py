from pos import flair_embeddings
from pos.types import Symbols


def test_electra():
    test = Symbols(("Þetta", "er", "próf", "."))
    embedding = flair_embeddings.electra_embedding(test)
    print(embedding)
    assert embedding.size() == (4, 256)
