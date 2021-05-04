"""To test parts of the model."""
import torch
from pos.core import Dicts
from pos.data import BATCH_KEYS, collate_fn
from pos.model import (
    ABLTagger,
    CharacterAsWordEmbedding,
    CharacterDecoder,
    ClassingWordEmbedding,
    Encoder,
    Modules,
    MultiplicativeAttention,
    PretrainedEmbedding,
    Tagger,
    TransformerEmbedding,
)
from pos.model.embeddings import CharacterEmbedding
from torch import zeros


def test_classic_wemb(vocab_maps, data_loader):
    emb_dim = 3
    wemb = ClassingWordEmbedding(vocab_map=vocab_maps[Dicts.Tokens], embedding_dim=emb_dim)
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


def test_chars_as_words(vocab_maps, data_loader):
    character_embedding = CharacterEmbedding(vocab_maps[Dicts.Chars], embedding_dim=10)
    wemb = CharacterAsWordEmbedding(character_embedding=character_embedding)
    # The TransformerEmbedding expects the input to be a Sentence, not vectors.
    for batch in data_loader:
        embs = wemb(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])[0]  # Only take the chars
        assert embs.shape == (3 * 3, 9, 64 * 2)
        assert embs.requires_grad == True


def test_transformer_embedding_electra_small(electra_model, data_loader):
    wemb = TransformerEmbedding(electra_model)
    # The TransformerEmbedding expects the input to be a Sentence, not vectors.
    for batch in data_loader:
        embs = wemb(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        embs = embs["hidden_states"][-1]
        assert embs.shape == (3, 128, 256)
        assert embs.requires_grad == True


def test_transformer_embedding_electra_small_only_last(electra_model, data_loader):
    wemb = TransformerEmbedding(electra_model)
    # The TransformerEmbedding expects the input to be a Sentence, not vectors.
    for batch in data_loader:
        embs = wemb(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        embs = embs["hidden_states"][-1]
        assert embs.shape == (3, 128, 256)
        assert embs.requires_grad == True


def test_encoder(encoder: Encoder, data_loader):
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        assert embs[Modules.BiLSTM].shape == (3, 3, encoder.output_dim)
        assert embs[Modules.BiLSTM].requires_grad == True


def test_tagger(encoder, data_loader, vocab_maps):
    tagger_module = Tagger(vocab_map=vocab_maps[Dicts.FullTag], input_dim=encoder.output_dim)
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        tag_embs = tagger_module(embs, batch)
        assert tag_embs.shape == (3, 3, tagger_module.output_dim)
        assert embs[Modules.BiLSTM].requires_grad == True


def test_full_run(data_loader, vocab_maps, electra_model):
    emb = TransformerEmbedding(electra_model)
    encoder = Encoder(embeddings={Modules.BERT: emb})
    tagger = Tagger(vocab_map=vocab_maps[Dicts.FullTag], input_dim=encoder.output_dim)
    character_embedding = CharacterEmbedding(vocab_maps[Dicts.Chars], embedding_dim=10)
    emb_dim = 5
    char_decoder = CharacterDecoder(
        character_embedding=character_embedding,
        vocab_map=vocab_maps[Dicts.Chars],
        hidden_dim=70,
        context_dim=tagger.output_dim,
        num_layers=2,
    )
    abl_tagger = ABLTagger(encoder=encoder, tagger=tagger, character_decoder=char_decoder)
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
