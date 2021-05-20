"""To test parts of the model."""
import torch
from pos.core import Dicts
from pos.data.constants import Modules
from pos.model import (
    CharacterAsWordEmbedding,
    CharacterDecoder,
    ClassicWordEmbedding,
    EncodersDecoders,
    MultiplicativeAttention,
    PretrainedEmbedding,
    Tagger,
    TransformerEmbedding,
)
from pos.model.embeddings import CharacterEmbedding
from torch import zeros


def test_classic_wemb(vocab_maps, data_loader):
    emb_dim = 3
    wemb = ClassicWordEmbedding(Modules.Trained, vocab_map=vocab_maps[Dicts.Tokens], embedding_dim=emb_dim)
    for batch in data_loader:
        batch = wemb.preprocess(batch)
        embs = wemb(batch)[Modules.Trained]
        assert embs.shape == (3, 3, emb_dim)
        assert embs.requires_grad


def test_pretrained_wemb(vocab_maps, data_loader):
    emb_dim = 3
    pretrained_weights = zeros(size=(9, emb_dim))
    wemb = PretrainedEmbedding(
        Modules.Pretrained,
        vocab_map=vocab_maps[Dicts.Tokens],
        embeddings=pretrained_weights,
        freeze=True,
    )
    for batch in data_loader:
        batch = wemb.preprocess(batch)
        embs = wemb(batch)[Modules.Pretrained]
        assert embs.shape == (3, 3, emb_dim)
        assert not embs.requires_grad


def test_chars_as_words(vocab_maps, data_loader):
    character_embedding = CharacterEmbedding(Modules.Characters, vocab_maps[Dicts.Chars], embedding_dim=10)
    wemb = CharacterAsWordEmbedding(Modules.CharactersToTokens, character_embedding=character_embedding)
    # The TransformerEmbedding expects the input to be a Sentence, not vectors.
    for batch in data_loader:
        batch = wemb.preprocess(batch)
        embs = wemb(batch)[Modules.CharactersToTokens][0]  # Only take the chars
        assert embs.shape == (3 * 3, 9, 64 * 2)
        assert embs.requires_grad


def test_transformer_embedding_electra_small(electra_model, data_loader):
    wemb = TransformerEmbedding(Modules.BERT, electra_model)
    for batch in data_loader:
        batch = wemb.preprocess(batch)
        embs = wemb(batch)[Modules.BERT]
        assert embs.shape == (3, 3, wemb.hidden_dim)
        assert embs.requires_grad


def test_tagger(tagger_module, data_loader):
    for batch in data_loader:
        batch = tagger_module.encoder.preprocess(batch)
        batch = tagger_module.encoder(batch)
        tag_embs = tagger_module(batch)[Modules.Tagger]
        assert tag_embs.shape == (3, 3, tagger_module.output_dim)


def test_full_run(data_loader, vocab_maps, electra_model):
    emb = TransformerEmbedding(Modules.BERT, electra_model)
    tagger = Tagger(Modules.Tagger, vocab_map=vocab_maps[Dicts.FullTag], encoder=emb, encoder_key=Modules.BERT)
    character_embedding = CharacterEmbedding(Modules.Characters, vocab_maps[Dicts.Chars], embedding_dim=10)
    wemb = CharacterAsWordEmbedding(Modules.CharactersToTokens, character_embedding=character_embedding)
    tag_emb = ClassicWordEmbedding(Modules.TagEmbedding, vocab_map=vocab_maps[Dicts.FullTag], embedding_dim=4)
    char_decoder = CharacterDecoder(
        Modules.Lemmatizer,
        tag_encoder=tag_emb,
        characters_to_tokens_encoder=wemb,
        vocab_map=vocab_maps[Dicts.Chars],
        hidden_dim=70,
        num_layers=2,
    )
    abl_tagger = EncodersDecoders(
        encoders={emb.key: emb, tag_emb.key: tag_emb, wemb.key: wemb},
        decoders={Modules.Tagger: tagger, Modules.Lemmatizer: char_decoder},
    )
    for batch in data_loader:
        preds = abl_tagger(batch)
        assert preds[Modules.Tagger].shape == (3, 3, len(vocab_maps[Dicts.FullTag]))
        assert preds[Modules.Lemmatizer].shape == (
            9,
            8,
            len(vocab_maps[Dicts.Chars]),
        )  # num words, max word length in chars


def test_attention():
    hidden_decoder = torch.rand(size=(4, 2))
    hiddens_encoder = torch.rand(size=(4, 3, 2))
    attention = MultiplicativeAttention(encoder_dim=2, decoder_dim=2)
    result = attention(hidden_decoder, hiddens_encoder)
    assert result is not None
