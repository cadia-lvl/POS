"""To test parts of the model."""
from torch import zeros
import torch

from pos.model import (
    ClassingWordEmbedding,
    MultiplicativeAttention,
    Modules,
    PretrainedEmbedding,
    TransformerEmbedding,
    Tagger,
    Encoder,
    ABLTagger,
    CharacterDecoder,
)
from pos.core import Dicts
from pos.data import BATCH_KEYS, collate_fn


def test_classic_wemb(vocab_maps, data_loader):
    emb_dim = 3
    wemb = ClassingWordEmbedding(
        vocab_map=vocab_maps[Dicts.Tokens], embedding_dim=emb_dim
    )
    for batch in data_loader:
        embs = wemb(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
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
        embs = wemb(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        assert embs.shape == (3, 3, emb_dim)
        assert embs.requires_grad == False


def test_chars_as_words():
    assert True  # We will not do this now.


def test_transformer_embedding_electra_small(electra_model, data_loader):
    wemb = TransformerEmbedding(electra_model)
    # The TransformerEmbedding expects the input to be a Sentence, not vectors.
    for batch in data_loader:
        embs = wemb(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        assert embs.shape == (3, 3, 256)
        assert embs.requires_grad == True


def test_encoder(encoder: Encoder, data_loader):
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        assert embs[Modules.BiLSTM].shape == (3, 3, encoder.output_dim)
        assert embs[Modules.BiLSTM].requires_grad == True


def test_tagger(encoder, data_loader, vocab_maps):
    tagger_module = Tagger(
        vocab_map=vocab_maps[Dicts.FullTag], input_dim=encoder.output_dim
    )
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        tag_embs = tagger_module(embs, batch)
        assert tag_embs.shape == (3, 3, tagger_module.output_dim)
        assert embs[Modules.BiLSTM].requires_grad == True


def test_gru_decoder(vocab_maps, data_loader, encoder: Encoder):
    hidden_dim = encoder.output_dim
    emb_dim = 5
    char_decoder = CharacterDecoder(
        vocab_map=vocab_maps[Dicts.Chars],
        hidden_dim=hidden_dim,
        context_dim=hidden_dim,
        num_layers=2,
        emb_dim=emb_dim,
    )
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        tok_embs = char_decoder(embs, batch)
        assert tok_embs.shape == (
            9,
            8,  # 8 + 20 - 1 if eval.
            len(vocab_maps[Dicts.Chars]),
        )  # 9 tokens, 8 chars at most, each char emb
        assert tok_embs.requires_grad == True


def test_full_run(data_loader, vocab_maps, electra_model):
    emb = TransformerEmbedding(electra_model)
    encoder = Encoder(embeddings={Modules.BERT: emb})
    tagger = Tagger(vocab_map=vocab_maps[Dicts.FullTag], input_dim=encoder.output_dim)
    abl_tagger = ABLTagger(encoder=encoder, decoders={Modules.Tagger: tagger})
    for batch in data_loader:
        abl_tagger(batch)


def test_attention():
    hidden_decoder = torch.rand(size=(4, 2))
    hiddens_encoder = torch.rand(size=(4, 3, 2))
    attention = MultiplicativeAttention(encoder_dim=2, decoder_dim=2)
    result = attention(hidden_decoder, hiddens_encoder)