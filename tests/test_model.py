"""To test parts of the model."""
import pytest
from torch import Tensor

from pos import model


def test_classic_wemb():
    emb_dim = 3
    wemb = model.ClassingWordEmbedding(2, emb_dim)  # 2 tokens, 3 dim for embedding
    padded_emb = Tensor([0.0] * emb_dim)
    t = Tensor([0, 1, 0]).long()  # pad, tok, pad
    emb = wemb(t)
    assert emb.requires_grad == True
    assert all(emb[0].eq(padded_emb))
    assert not all(emb[1].eq(padded_emb))
    assert all(emb[2].eq(padded_emb))


def test_pretrained_wemb():
    emb_dim = 3
    # We place 1.0 where the padding is supposed to be, to check how it works
    pretrained_weights = Tensor([[1.0] * emb_dim, [0.0] * emb_dim, [-1.0] * emb_dim])
    wemb = model.PretrainedEmbedding(pretrained_weights, freeze=True, padding_idx=0)
    padded_emb = Tensor([0.0] * emb_dim)
    t = Tensor([0, 1, 2]).long()  # tok0, tok1, tok2
    emb = wemb(t)
    assert emb.requires_grad == False
    assert sum(emb[0]).item() == 3.0
    assert sum(emb[1]).item() == 0.0
    assert sum(emb[2]).item() == -3.0

    wemb = model.PretrainedEmbedding(pretrained_weights, freeze=False, padding_idx=0)
    emb = wemb(t)
    assert emb.requires_grad == True


def test_chars_as_words():
    assert True  # We will not do this now.


def test_transformer_embedding_electra_small(electra_model):
    if not electra_model:
        pytest.skip("No --electra_model given")
    transformer_emb = model.load_transformer_embeddings(electra_model)
    # The TransformerEmbedding expects the input to be a Sentence, not vectors.
    from flair.data import Sentence

    s = Sentence("Þetta er vinnuvélaverkamannaskúr")
    transformer_emb.embed(s)
    # (Electra-small) This will be 3 tokens since we only use the head-subword token.
    assert len(s) == 3
    assert all(token.embedding.shape == (256,) for token in s)


def test_combination(data_loader):
    num_tokens = 8
    wemb_dim = 3
    num_tags = 6
    batch_size = 3
    max_sent_len = 3
    wemb = model.ClassingWordEmbedding(num_tokens, wemb_dim)
    encoder = model.Encoder(word_embedding=wemb)
    tagger = model.Tagger(encoder.output_dim, num_tags)
    abl_tagger = model.ABLTagger(encoder=encoder, tagger=tagger)
    for batch in data_loader:
        assert abl_tagger(batch).shape == (batch_size, max_sent_len, num_tags)
    assert True
