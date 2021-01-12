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
#    wemb = TransformerEmbedding(electra_model)
#    # fmt: off
#    #batch =  (('Tertudeig', ':', '200', 'g', 'hveiti', ',', '100', 'g', 'sykur', ',', '100', 'g', 'saltað', 'smjör', 'við', 'stofuhita', ',', '2', 'eggjarauður', '.'), ('Það', 'var', 'eins', 'og', 'hann', 'drægi', 'augnaráð', 'hennar', 'til', 'sín', 'með', 'einhverjum', 'töframætti', 'uns', 'það', 'staðnæmdist', 'við', 'brún', 'augu', 'hans', '.'), ('6,528', 'Langlúra', '50', '50', '50', '249', '12,450', 'Lúða', '210', '133', '206', '187', '38,444', 'Skarkoli', '170', '22', '134', '33', '4,426', 'Skata', '8', '8', '8', '119', '952', 'Skötuselur', '177', '83', '169', '1,124', '190,223', 'Steinbítur', '80', '80', '80', '1,256', '100,480', 'Ufsi', '30', '12', '28', '6,720', '189,412', 'Ýsa', '79', '79', '1,451', '114,629', 'Þykkvalúra', '207', '82', '198', '188', '37,291', 'Samtals', '61', '11,527', '698,288', 'FMS', 'SANDGERÐI', '/', 'NJARÐVÍK', 'Gullkarfi', '101', '101', '101', '1,200', '121,200', 'Keila', '42', '15', '39', '641', '24,852', 'Langa', '77', '55', '67', '544', '36,688', 'Lúða', '488', '187', '315', '70', '22,032', 'Skarkoli', '192', '178', '190', '825', '156,902', 'Skrápflúra', '7', '7', '7', '220', '1,540', 'Skötuselur', '126', '126', '126', '374', '47,124', 'Steinbítur', '89', '52', '74', '2,449', '182,026', 'Ufsi', '34', '29', '30', '6,743', '203,762', 'Und.', 'þorskur', '113', '92', '99', '2,406', '237,678', 'Ýsa', '136', '39', '112', '4,349', '486,820', 'Þorskur', '195', '148', '150', '6,425', '964,775', 'Þykkvalúra', '228', '197', '223', '1,127', '251,725', 'Samtals', '27,373', '2,737,124', 'FMS', 'ÍSAFIRÐI', 'Gellur', '478', '478', '478', '15', '7,170', 'Hlýri', '63', '63', '63', '60', '3,780', 'Keila', '60', '60', '60', '25', '1,500', 'Lúða', '218', '207', '213', '22', '4,686', 'Skarkoli', '139', '139', '139', '47', '6,533', 'Steinbítur', '77', '71', '75', '1,543', '115,560', 'Und.', 'ýsa', '36', '36', '36', '266', '9,576', 'Und.', 'þorskur', '63', '61', '62', '471', '29,295', 'Ýsa', '181', '53', '121', '3,571', '431,427', 'Þorskur', '161', '81', '110', '4,827', '528,779', 'Samtals', '105', '10,847', '1,138,306', 'FISKMARKAÐUR', 'ÍSLANDS', 'Blálanga', '39', '39', '39', '118', '4,602', 'Gellur', '512', '504', '507', '40', '20,270', 'Grálúða', '197', '197', '197', '264', '52,008', 'Gullkarfi', '97', '32', '89', '4,203', 'Hlýri', '89', '68', '82', '3,906', '318,663', 'Keila', '69', '6', '35', '742', '25,893', 'Langa', '73', '61', '1,445', '88,166', 'Lúða', '456', '106', '297', '444', '131,853', 'Lýsa', '32', '15', '31', '161', '4,965'), ('Sífellt', 'meira', 'úrval', 'er', 'af', 'þjóðlegu', 'jólaskrauti', 'sem', 'er', 'vel', 'því', 'íslenskar', 'jólahefðir', 'eru', 'rótgrónar', 'og', 'greyptar', 'inn', 'í', 'þjóðarsálina', '.'), ('Jólaskapið', 'er', 'þannig', 'nátengt', 'staðnum', 'og', 'Júlíus', 'bætir', 'því', 'við', 'að', 'hann', 'hafi', 'saknað', 'Dalvíkur', 'þegar', 'hann', ',', 'eitt', 'árið', ',', 'eyddi', 'aðventunni', 'fyrir', 'sunnan', '.'), ('Eftir', 'nokkrar', 'mínútur', 'breyttist', 'hún', 'þó', 'aðeins', 'og', 'sagði', 'að', 'það', 'væri', 'sennilega', 'ekki', 'sniðugt', 'að', 'hún', 'væri', 'að', 'reyna', 'við', 'mig', ',', 'þar', 'sem', 'ég', 'væri', '«', 'alltof', 'ungur', '»', '.'), ('Miriam', 'Stoppard', '(', '1992', ')', 'nefnir', 'margar', 'ástæður', 'þess', 'að', 'kona', 'hafi', 'ekki', 'áhuga', 'á', 'kynlífi', '.'), ('Á', 'leiðinni', 'niður', 'man', 'hann', 'eftir', 'að', 'hafa', 'ætlað', 'að', 'segja', 'föður', 'sínum', 'frá', 'nafninu', 'á', 'götunni', 'sem', 'hann', 'býr', 'við', ';', 'hann', 'þykist', 'vita', 'að', 'það', 'muni', 'kæta', 'sósíalistann', 'Jón', 'Magnússon', '.'))
#    # fmt: on
#    wemb.preprocess(batch)


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