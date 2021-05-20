"""Fixtures for tests."""
from typing import Dict

import pytest
from pytest import fixture
from torch.utils.data.dataloader import DataLoader

import pos
from pos import evaluate
from pos.cli import MORPHLEX_VOCAB_PATH, PRETRAINED_VOCAB_PATH
from pos.core import Dicts, FieldedDataset, Fields, Vocab, VocabMap
from pos.data import collate_fn, load_dicts
from pos.data.constants import Modules
from pos.model import CharacterDecoder, ClassicWordEmbedding, EncodersDecoders, Tagger
from pos.model.embeddings import CharacterAsWordEmbedding, CharacterEmbedding
from pos.model.interface import Decoder


def pytest_addoption(parser):
    """Add extra command-line options to pytest."""
    parser.addoption("--tagger", action="store")
    parser.addoption("--electra_model", action="store")


@fixture()
def electra_model(request):
    """Exposes the command-line option to a test case."""
    electra_model_path = request.config.getoption("--electra_model")
    if not electra_model_path:
        pytest.skip("No --electra_model given")
    else:
        return electra_model_path


@fixture(scope="session")
def pretrained_tagger(request):
    """Exposes the command-line option to a test case."""
    pretrained_tagger_path = request.config.getoption("--tagger")
    if not pretrained_tagger_path:
        pytest.skip("No --tagger given")
    else:
        return pos.Tagger(
            model_file=pretrained_tagger_path,
            device="cpu",
        )


@fixture()
def test_tsv_untagged_file():
    """Return the filepath of the test tsv file."""
    return "./tests/test_untagged.tsv"


@fixture()
def test_tsv_file():
    """Return the filepath of the test tsv file."""
    return "./tests/test.tsv"


@fixture()
def tagged_test_tsv_file():
    """Return the filepath of the test tsv file."""
    return "./tests/test_pred.tsv"


@fixture()
def test_tsv_lemma_file():
    """Return the filepath of the test tsv file."""
    return "./tests/test_lemma.tsv"


@fixture
def ds(test_tsv_file):
    """Return a sequence tagged dataset."""
    return FieldedDataset.from_file(test_tsv_file, fields=(Fields.Tokens, Fields.GoldTags))


@fixture
def ds_lemma(test_tsv_lemma_file):
    """Return a sequence tagged dataset."""
    return FieldedDataset.from_file(test_tsv_lemma_file, fields=(Fields.Tokens, Fields.GoldTags, Fields.GoldLemmas))


@fixture
def vocab_maps(ds_lemma) -> Dict[Dicts, VocabMap]:
    """Return the dictionaries for the dataset."""
    return load_dicts(ds_lemma)[1]


@fixture
def kwargs():
    """Return a default set of arguments."""
    return {
        "tagger": True,
        "batch_size": 3,
        "lemmatizer": True,
        "lemmatizer_hidden_dim": 50,
        "word_embedding_dim": 3,
        "tagger_weight": 1,
        "lemmatizer_weight": 1,
        "char_emb_dim": 20,
        "char_lstm_layers": 1,
        "main_lstm_layers": 1,
        "main_lstm_dim": 128,
        "scheduler": "multiply",
        "learning_rate": 5e-5,
        "word_embedding_lr": 0.2,
        "optimizer": "adam",
        "label_smoothing": 0.1,
        "output_dir": "debug/",
        "epochs": 20,
    }


@fixture()
def data_loader(ds_lemma, kwargs):
    """Return a data loader over the unit testing data."""
    return DataLoader(ds_lemma, batch_size=kwargs["batch_size"], collate_fn=collate_fn)  # type: ignore


@fixture
def classic_emb(vocab_maps, kwargs) -> ClassicWordEmbedding:
    """Return an Encoder."""
    return ClassicWordEmbedding(
        Modules.Trained, vocab_map=vocab_maps[Dicts.Tokens], embedding_dim=kwargs["word_embedding_dim"]
    )


@fixture
def tagger_module(vocab_maps, classic_emb) -> Tagger:
    """Return a Tagger."""
    return Tagger(Modules.Tagger, vocab_map=vocab_maps[Dicts.FullTag], encoder=classic_emb, encoder_key=Modules.Trained)


@fixture
def char_emb_module(vocab_maps, classic_emb) -> CharacterEmbedding:
    """Return a Tagger."""
    character_embedding = CharacterEmbedding(Modules.Characters, vocab_maps[Dicts.Chars], embedding_dim=10)
    return character_embedding


@fixture
def char_as_word_emb_module(char_emb_module) -> CharacterAsWordEmbedding:
    """Return a Tagger."""
    wemb = CharacterAsWordEmbedding(Modules.CharactersToTokens, character_embedding=char_emb_module)
    return wemb


@fixture
def tag_emb_module(vocab_maps) -> ClassicWordEmbedding:
    """Return a Tagger."""
    tag_emb = ClassicWordEmbedding(Modules.TagEmbedding, vocab_map=vocab_maps[Dicts.FullTag], embedding_dim=4)
    return tag_emb


@fixture
def lemmatizer_module(tag_emb_module, char_as_word_emb_module, vocab_maps) -> CharacterDecoder:
    """Return a Tagger."""
    char_decoder = CharacterDecoder(
        Modules.Lemmatizer,
        tag_encoder=tag_emb_module,
        characters_to_tokens_encoder=char_as_word_emb_module,
        vocab_map=vocab_maps[Dicts.Chars],
        hidden_dim=70,
        num_layers=2,
    )
    return char_decoder


@fixture
def decoders(tagger_module, lemmatizer_module) -> Dict[str, Decoder]:
    """Return the decoders."""
    return {Modules.Lemmatizer: lemmatizer_module, Modules.Tagger: tagger_module}


@fixture
def encoders(char_as_word_emb_module, tag_emb_module, classic_emb) -> Dict[str, Decoder]:
    """Return the decoders."""
    return {
        char_as_word_emb_module.key: char_as_word_emb_module,
        tag_emb_module.key: tag_emb_module,
        classic_emb.key: classic_emb,
    }


@fixture
def abl_tagger(encoders, decoders) -> EncodersDecoders:
    """Return a default ABLTagger."""
    return EncodersDecoders(encoders=encoders, decoders=decoders)


@fixture
def tagger_evaluator(ds_lemma):
    """Return a tagger evaluator."""
    return evaluate.TaggingEvaluation(
        test_dataset=ds_lemma,
        train_vocab=ds_lemma.get_vocab(),
        external_vocabs=evaluate.ExternalVocabularies(
            morphlex_tokens=Vocab.from_file(MORPHLEX_VOCAB_PATH),
            pretrained_tokens=Vocab.from_file(PRETRAINED_VOCAB_PATH),
        ),
    ).tagging_accuracy


@fixture
def lemma_evaluator(ds_lemma):
    """Return a lemma evaluator."""
    return evaluate.LemmatizationEvaluation(
        test_dataset=ds_lemma,
        train_vocab=ds_lemma.get_vocab(),
        train_lemmas=Vocab.from_symbols(ds_lemma.get_field(Fields.GoldLemmas)),
    ).lemma_accuracy
