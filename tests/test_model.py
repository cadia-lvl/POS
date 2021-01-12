"""To test parts of the model."""
from torch import zeros
import torch

from pos.model import (
    ClassingWordEmbedding,
    DotAttention,
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


# def test_transformer_embedding_electra_small_preprocess(electra_model):
# wemb = TransformerEmbedding(electra_model)
# # fmt: off
# batch =  (('Glæsilegt', 'nýtt', 'heilsárshús', 'á', 'góðri', 'kjarri', 'vaxinni', 'lóð', '.'), ('DALVÍKUR', 'Hlýri', '75', '75', '75', '37', '2,775', 'Steinbítur', '73', '47', '62', '611', '37,957', 'Ufsi', '7', '7', '7', '142', '994', 'Und.', 'ýsa', '55', '55', '55', '75', '4,125', 'Und.', 'þorskur', '75', '66', '71', '2,438', '173,741', 'Ýsa', '136', '58', '122', '1,328', '161,697', 'Þorskur', '124', '112', '116', '6,690', '776,939', 'Samtals', '102', '11,321', '1,158,227', 'FISKMARKAÐUR', 'DJÚPAVOGS', 'Sandkoli', '60', '60', '60', '132', '7,920', 'Skarkoli', '171', '169', '170', '512', '87,124', 'Skrápflúra', '40', '40', '40', '121', '4,840', 'Steinbítur', '78', '76', '77', '6,786', '519,500', 'Ýsa', '123', '123', '123', '128', '15,744', 'Þorskur', '177', '177', '177', '109', '19,293', 'Samtals', '7,788', '654,421', 'FISKMARKAÐUR', 'FLATEYRAR', 'Gullkarfi', '6', '6', '6', '3', '18', 'Hlýri', '61', '61', '61', '144', '8,784', 'Lúða', '235', '189', '217', '95', '20,574', 'Skarkoli', '167', '167', '167', '51', '8,517', 'Skötuselur', '186', '186', '186', '7', '1,302', 'Steinbítur', '59', '42', '59', '3,458', '203,989', 'Ufsi', '23', '23', '23', '83', '1,909', 'Und.', 'ýsa', '38', '38', '38', '1,117', '42,446', 'Ýsa', '165', '71', '102', '2,397', '244,368', 'Samtals', '72', '7,355', '531,907', 'FISKMARKAÐUR', 'HÚSAVÍKUR', 'Hlýri', '79', '79', '79', '25', '1,975', 'Steinbítur', '75', '75', '75', '4', '300', 'Ufsi', '5', '5', '5', '10', '50', 'Ýsa', '144', '138', '140', '904', '126,828', 'Samtals', '137', '943', '129,153', 'FISKMARKAÐUR', 'SUÐUREYRAR', 'Gullkarfi', '38', '38', '9', '342', 'Lúða', '207', '207', '207', '5', '1,035', 'Skarkoli', '172', '172', '172', '51', '8,772', 'Steinbítur', '42', '42', '42', '10', '420', 'Ufsi', '24', '24', '24', '206', '4,944', 'Und.', 'þorskur', '70', '70', '70', '886', '62,020', 'Þorskur', '139', '84', '109', '9,352', '1,014,855', 'Samtals', '104', '10,519', '1,092,388', 'FISKMARKAÐUR', 'TÁLKNAFJARÐAR', 'Gellur', '478', '478', '478', '40', '19,120', 'Lúða', '189', '180', '184', '27', '4,968', 'Skarkoli', '172', '167', '171', '349', '59,828', 'Ýsa', '117', '62', '82', '986', '81,158', 'Samtals', '118', '1,402', '165,074', 'FISKMARKAÐUR', 'VESTFJARÐA', 'Gullkarfi', '10', '10', '10', '8', '80', 'Hlýri', '73', '73', '73', '13', '949', 'Skarkoli', '152', '152', '152', '4', '608', 'Steinbítur', '58', '58', '156', '9,048', 'Und.', 'ýsa', '42', '42', '42', '60', '2,520'), ('ég', 'sperrti', 'aðeins', 'fjaðrirnar', ',', 'sveiflaði', 'augnhárununum', 'og', 'leyfði', 'munúðarfullu', 'brosi', 'að', 'leika', 'um', 'varirnar', 'meðan', 'ég', 'gekk', 'í', 'gegnum', 'salinn', '.'), ('Íbúar', 'höfuðborgarsvæðisins', 'geta', 'keypt', 'þessa', 'afurð', 'í', 'Blómavali', ',', 'Grafarvogi', 'og', 'Skútuvogi', 'og', 'í', 'Álfsnesi', 'og', 'sjá', 'starfsmenn', 'urðunarstaðarins', 'um', 'að', 'moka', 'því', 'á', 'kerrur', '.'), ('«', 'Hvernig', 'fór', 'hann', 'að', 'þessu', '?', '»'), ('Uppl.', 'í', 's.', '895', '8763', 'www.trevinnustofan@visir.is', '.'), ('what', 'a', 'waste', 'skiljiði', '.'), ('Í', 'ár', 'hyggst', 'Júlli', 'í', 'samstarfi', 'við', 'verslunina', 'Elektro', 'taka', 'myndir', 'í', 'bænum', 'með', 'reglulegu', 'millibili', 'og', 'birta', 'á', 'vefnum', ',', 'til', 'þess', 'að', 'sýna', 'þróunina', 'í', 'útiskreytingum', 'eftir', 'því', 'sem', 'á', 'desember', 'líður', '.'))
# # fmt: on
# wemb.preprocess(batch)


def test_encoder(encoder: Encoder, data_loader):
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        assert embs[Modules.BiLSTM].shape == (3, 3, encoder.output_dim)
        assert embs[Modules.BiLSTM].requires_grad == True


def test_tagger(encoder, data_loader, tagger_module):
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
        emb_dim=emb_dim,
    )
    for batch in data_loader:
        embs = encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        tok_embs = char_decoder(embs, batch)
        assert tok_embs.shape == (
            9,
            8,
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
    attention = DotAttention()
    result = attention(hidden_decoder, hiddens_encoder)