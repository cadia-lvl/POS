"""To test parts of the model."""
import pytest
from torch import Tensor, zeros

from pos.model import (
    ClassingWordEmbedding,
    PretrainedEmbedding,
    FlairTransformerEmbedding,
    Tagger,
    Encoder,
    ABLTagger,
    GRUDecoder,
)
from pos.core import Dicts
from pos.data import BATCH_KEYS


def test_classic_wemb(vocab_maps, data_loader):
    emb_dim = 3
    wemb = ClassingWordEmbedding(
        vocab_map=vocab_maps[Dicts.Tokens], embedding_dim=emb_dim
    )
    for batch in data_loader:
        embs = wemb(batch[BATCH_KEYS.TOKENS])
        assert embs.shape == (3, 3, emb_dim)
        assert embs.requires_grad == True


def test_pretrained_wemb(vocab_maps, data_loader):
    emb_dim = 3
    pretrained_weights = zeros(size=(9, emb_dim))
    wemb = PretrainedEmbedding(
        vocab_map=vocab_maps[Dicts.Tokens],
        embeddings=pretrained_weights,
        freeze=True,
    )
    for batch in data_loader:
        embs = wemb(batch[BATCH_KEYS.TOKENS])
        assert embs.shape == (3, 3, emb_dim)
        assert embs.requires_grad == False


def test_chars_as_words():
    assert True  # We will not do this now.


def test_transformer_embedding_electra_small(electra_model, data_loader):
    if not electra_model:
        pytest.skip("No --electra_model given")
    wemb = FlairTransformerEmbedding(electra_model)
    # The TransformerEmbedding expects the input to be a Sentence, not vectors.
    for batch in data_loader:
        embs = wemb(batch[BATCH_KEYS.TOKENS])
        assert embs.shape == (3, 3, 256)
        assert embs.requires_grad == True


def test_encoder(encoder, data_loader):
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        assert embs.shape == (3, 3, encoder.output_dim)
        assert embs.requires_grad == True


def test_tagger(encoder, data_loader, tagger_module):
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        tag_embs = tagger_module(embs, batch)
        assert tag_embs.shape == (3, 3, tagger_module.output_dim)
        assert embs.requires_grad == True


def test_gru_decoder(vocab_maps, data_loader, encoder: Encoder):
    hidden_dim = 3
    output_dim = 4
    context_dim = encoder.output_dim
    emb_dim = 5
    gru = GRUDecoder(
        vocab_map=vocab_maps[Dicts.Chars],
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        output_dim=output_dim,
        emb_dim=emb_dim,
    )
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        tok_embs = gru(embs, batch)
        assert tok_embs.shape == (
            9,
            8,
            4,
        )  # 9 tokens, 8 chars at most, each char uses 4 emb_dim
        assert tok_embs.requires_grad == True


def test_full_run(data_loader, vocab_maps, electra_model):
    if not electra_model:
        pytest.skip("No --electra_model given")
    emb = FlairTransformerEmbedding(electra_model)
    encoder = Encoder(embeddings=[emb])
    tagger = Tagger(
        vocab_map=vocab_maps[Dicts.FullTag], input_dim=encoder.output_dim, output_dim=5
    )
    abl_tagger = ABLTagger(encoder=encoder, decoders=[tagger])
    for batch in data_loader:
        abl_tagger(batch)
