#!/usr/bin/env python
"""The main entrypoint to training and running a POS-tagger."""
import json
import logging
import pathlib
import pickle
import random
import re
from collections import Counter
from pprint import pprint
from typing import Dict

import click
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from pos import bin_to_ifd, core, evaluate
from pos.api import Tagger as api_tagger
from pos.core import Dicts, FieldedDataset, Fields, Vocab, VocabMap, set_device, set_seed
from pos.data import (
    bin_str_to_emb_pair,
    chunk_dataset,
    collate_fn,
    dechunk_dataset,
    emb_pairs_to_dict,
    load_dicts,
    read_datasets,
    read_morphlex,
    read_pretrained_word_embeddings,
    wemb_str_to_emb_pair,
)
from pos.data.constants import PAD, Modules
from pos.model import (
    CharacterAsWordEmbedding,
    CharacterDecoder,
    ClassicWordEmbedding,
    Decoder,
    Encoder,
    EncodersDecoders,
    Tagger,
    TransformerEmbedding,
)
from pos.model.embeddings import CharacterEmbedding
from pos.train import (
    MODULE_TO_FIELD,
    get_criterion,
    get_optimizer,
    get_scheduler,
    print_model,
    run_epochs,
    tag_data_loader,
)

MORPHLEX_VOCAB_PATH = "./data/extra/morphlex_vocab.txt"
PRETRAINED_VOCAB_PATH = "./data/extra/pretrained_vocab.txt"

log = logging.getLogger(__name__)


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):  # pylint: disable=redefined-outer-name
    """Entrypoint to the program. --debug flag from command line is caught here."""
    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        log.info("Logging set to DEBUG")
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=log_level)


@cli.command()
@click.argument("filepaths", nargs=-1)
@click.argument("output", type=click.File("w"))
@click.option(
    "--type",
    type=click.Choice(["tags", "tokens", "lemmas", "morphlex", "pretrained"], case_sensitive=False),
    default="tags",
)
def collect_vocabularies(filepaths, output, type):
    """Read all input files and extract relevant vocabulary."""
    result = []
    if type == "tags":
        ds = read_datasets(filepaths)
        result = list(Vocab.from_symbols(ds.get_field(Fields.GoldTags)))
    elif type == "tokens":
        ds = read_datasets(filepaths)
        result = list(Vocab.from_symbols(ds.get_field(Fields.Tokens)))
    elif type == "lemmas":
        ds = read_datasets(filepaths)
        result = list(Vocab.from_symbols(ds.get_field(Fields.GoldLemmas)))
    elif type == "morphlex":
        vocab_map, _ = read_morphlex(filepaths[0])
        result = list(vocab_map.w2i.keys())
    elif type == "pretrained":
        vocab_map, _ = read_pretrained_word_embeddings(filepaths[0])
        result = list(vocab_map.w2i.keys())
    for element in sorted(result):
        output.write(f"{element}\n")


@cli.command()
@click.argument("filepaths", nargs=-1)
@click.argument("embedding", default=None)
@click.argument("output", type=click.File("w+"))
@click.argument("emb_format", type=str)
def filter_embedding(filepaths, embedding, output, emb_format):
    """Filter an 'embedding' file based on the words which occur in the 'inputs' files.

    Result is written to the 'output' file with the same format as the embedding file.

    filepaths: Files to use for filtering (supports multiple files = globbing).
    Files should be .tsv, with two columns, the token and tag.
    embedding: The embedding file to filter.
    output: The file to write the result to.
    format: The format of the embedding file being read, 'bin' or 'wemb'.
    """
    log.info(f"Reading files={filepaths}")
    ds = read_datasets(filepaths)
    tokens = Vocab.from_symbols(ds.get_field())

    log.info(f"Number of tokens={len(tokens)}")
    with open(embedding) as f:
        if emb_format == "bin":
            emb_dict = emb_pairs_to_dict(f, bin_str_to_emb_pair)
            for token, value in emb_dict.items():
                if token in tokens:  # pylint: disable=unsupported-membership-test
                    output.write(f"{token};[{','.join((str(x) for x in value))}]\n")
        elif emb_format == "wemb":
            emb_dict = emb_pairs_to_dict(f, wemb_str_to_emb_pair)
            output.write("x x\n")
            for token, value in emb_dict.items():
                if token in tokens:  # pylint: disable=unsupported-membership-test
                    output.write(f"{token} {' '.join((str(x) for x in value))}\n")


@cli.command()
@click.argument("sh_snid")
@click.argument("output")
def prepare_bin_lemma_data_bart(sh_snid, output):
    """Prepare the BÍN data, extract form, pos (translated) and lemma."""
    bin_data = []
    with open(sh_snid) as f:
        for line in f:
            lemma, auðkenni, kyn_orðflokkur, hluti, orðmynd, mörk = line.strip().split(";")
            mim_mark = bin_to_ifd.parse_bin_str(
                orðmynd=orðmynd,
                lemma=lemma,
                kyn_orðflokkur=kyn_orðflokkur,
                mörk=mörk,
                samtengingar="c",
                afturbeygð_fn="fp",
            )
            # fjölyrtar segðir
            if mim_mark is None:
                continue
            bin_data.append((orðmynd, mim_mark, lemma))
    random.shuffle(bin_data)
    with open(output, "w") as f_mynd, open(f"{output}.lemma", "w") as f_lemma:
        for idx, values in enumerate(bin_data):
            orðmynd, mim_mark, lemma = values
            f_mynd.write(" ".join(mim_mark + "|" + orðmynd) + "\n")
            f_lemma.write(" ".join(lemma) + "\n")


@cli.command()
@click.argument("sh_snid")
@click.argument("output")
def prepare_bin_lemma_data(sh_snid, output):
    """Prepare the BÍN data, extract form, pos (translated) and lemma."""
    bin_data = []
    with open(sh_snid) as f:
        for line in f:
            lemma, auðkenni, kyn_orðflokkur, hluti, orðmynd, mörk = line.strip().split(";")
            mim_mark = bin_to_ifd.parse_bin_str(
                orðmynd=orðmynd,
                lemma=lemma,
                kyn_orðflokkur=kyn_orðflokkur,
                mörk=mörk,
                samtengingar="c",
                afturbeygð_fn="fp",
            )
            # fjölyrtar segðir
            if mim_mark is None:
                continue
            bin_data.append((orðmynd, mim_mark, lemma))
    random.shuffle(bin_data)
    with open(output, "w") as f:
        for idx, values in enumerate(bin_data):
            f.write("\t".join(values) + "\n")
            # we "prebatch"
            if idx % 128 == 127:
                f.write("\n")


# fmt: off
@cli.command()
@click.argument("training_files", nargs=-1)
@click.argument("test_file")
@click.argument("output_dir")
@click.option("--gpu/--no_gpu", default=False)
@click.option("--save_model/--no_save_model", default=True)
@click.option("--save_vocab/--no_save_vocab", default=True)
@click.option("--lemmatizer_accept_char_rnn_last/--no_lemmatizer_accept_char_rnn_last", default=False, help="Should the Character RNN last hidden state be input to Lemmatizer")
@click.option("--lemmatizer_hidden_dim", default=128, help="The hidden dim of the decoder RNN.")
@click.option("--lemmatizer_char_dim", default=64, help="The character embedding dim.")
@click.option("--lemmatizer_num_layers", default=1, help="The number of layers in Lemmatizer RNN.")
@click.option("--lemmatizer_char_attention/--no_lemmatizer_char_attention", default=True, help="Attend over characters?")
@click.option("--tag_embedding_dim", default=128, help="The tag embedding dimension")
@click.option("--known_chars_file", default="./data/extra/characters_training.txt", help="A file which contains the characters the model should know. File should be a single line, the line is split() to retrieve characters.",)
@click.option("--known_tags_file", default="./data/extra/all_tags.txt", help="A file which contains the pos the model should know. File should be a single line, the line is split() to retrieve elements.",)
@click.option("--char_lstm_layers", default=1, help="The number of layers in character LSTM embedding. Should not be set to 0.")
@click.option("--char_lstm_dim", default=128, help="The size of the hidden dim in character RNN.")
@click.option("--char_emb_dim", default=64, help="The embedding size for characters.")
@click.option("--emb_dropouts", default=0.0, help="The dropout to use for Embeddings.")
@click.option("--from_trained", default=None, help="The path to a model.pt which will be used as a starting point.")
@click.option("--label_smoothing", default=0.1)
@click.option("--learning_rate", default=5e-5)
@click.option("--epochs", default=20)
@click.option("--batch_size", default=16)
@click.option("--optimizer", default="adam", type=click.Choice(["sgd", "adam"], case_sensitive=False), help="The optimizer to use.")
@click.option("--scheduler", default="multiply", type=click.Choice(["none", "multiply", "plateau"], case_sensitive=False), help="The learning rate scheduler to use.")
# fmt: on
def train_lemmatizer(**kwargs):
    """Train a lemmatizer on intpus and write out results to a test file.

    training_files: Files to use for training (supports multiple files = globbing).
    All training files should be .tsv, with two/three columns, the token, tag, lemma.
    test_file: Same format as training_files. Used to evaluate the model.
    output_dir: The directory to write out model and results.
    """

    log.info("Training Lemmatizer")
    log.info(kwargs)
    set_seed()
    set_device(gpu_flag=kwargs["gpu"])

    # Read train and test data
    train_ds = read_datasets(kwargs["training_files"], fields=(Fields.Tokens, Fields.GoldTags, Fields.GoldLemmas))
    test_ds = read_datasets([kwargs["test_file"]], fields=(Fields.Tokens, Fields.GoldTags, Fields.GoldLemmas))
    # Dicts
    dictionaries = build_dictionaries(kwargs)
    if not kwargs["from_trained"]:
        lemmatizer = build_model(kwargs, dictionaries)
    else:
        log.info(f"Loading model={kwargs['from_trained']}")
        lemmatizer = load_lemmatizer_model(kwargs["from_trained"])
    lemmatizer.to(core.device)
    train_dl = DataLoader(
        train_ds, collate_fn=collate_fn, shuffle=True, batch_size=kwargs["batch_size"]  # type: ignore
    )
    test_dl = DataLoader(
        test_ds, collate_fn=collate_fn, shuffle=False, batch_size=kwargs["batch_size"] * 10  # type: ignore
    )
    criterion = get_criterion(lemmatizer.decoders, label_smoothing=kwargs["label_smoothing"])  # type: ignore
    optimizer = get_optimizer(lemmatizer.named_parameters(), kwargs["optimizer"], kwargs["learning_rate"])
    scheduler = get_scheduler(optimizer, kwargs["scheduler"])
    evaluators = {}
    evaluators[Modules.Lemmatizer] = evaluate.LemmatizationEvaluation(
        test_dataset=test_ds,
        train_vocab=train_ds.get_vocab(),
        train_lemmas=Vocab.from_symbols(train_ds.get_field(Fields.GoldLemmas)),
    ).lemma_accuracy

    # Write all configuration to disk
    output_dir = pathlib.Path(kwargs["output_dir"])
    write_hyperparameters(output_dir / "hyperparamters.json", (kwargs))

    # Start the training
    run_epochs(
        model=lemmatizer,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        evaluators=evaluators,
        train_data_loader=train_dl,
        test_data_loader=test_dl,
        epochs=kwargs["epochs"],
        output_dir=output_dir,
    )
    _, values = tag_data_loader(model=lemmatizer, data_loader=test_dl)
    log.info("Writing predictions, dictionaries and model")
    for module_name, value in values.items():
        test_ds = test_ds.add_field(value, MODULE_TO_FIELD[module_name])

    test_ds.to_tsv_file(str(output_dir / "predictions.tsv"))

    with output_dir.joinpath("known_toks.txt").open("w+") as f:
        for token in Vocab.from_symbols(train_ds.get_field(Fields.Tokens)):
            f.write(f"{token}\n")
    if Fields.GoldLemmas in train_ds.fields:
        with output_dir.joinpath("known_lemmas.txt").open("w+") as f:
            for lemma in Vocab.from_symbols(train_ds.get_field(Fields.GoldLemmas)):
                f.write(f"{lemma}\n")
    if kwargs["save_vocab"]:
        save_location = output_dir.joinpath("dictionaries.pickle")
        with save_location.open("wb+") as f:
            pickle.dump(dictionaries, f)
    if kwargs["save_model"]:
        save_location = output_dir.joinpath("lemmatizer.pt")
        torch.save(lemmatizer.state_dict(), str(save_location))
    log.info("Done!")


def load_lemmatizer_model(path):
    lemmatizer = torch.load(path, map_location=torch.device("cpu"))
    return lemmatizer


def build_dictionaries(kwargs):
    dictionaries: Dict[Dicts, VocabMap] = {}
    char_vocab = Vocab.from_file(kwargs["known_chars_file"])
    c_map = VocabMap(char_vocab, special_tokens=VocabMap.UNK_PAD_EOS_SOS)
    dictionaries[Dicts.Chars] = c_map
    tag_vocab = bin_to_ifd.öll_mörk(strip=True)
    tag_vocab = {tag for tag in tag_vocab if tag is not None}
    t_map = VocabMap(tag_vocab, special_tokens=VocabMap.UNK_PAD)
    dictionaries[Dicts.FullTag] = t_map
    return dictionaries


def build_model(kwargs, dicts) -> EncodersDecoders:
    embs: Dict[str, Encoder] = {}
    if kwargs["bert_encoder"]:
        emb = TransformerEmbedding(Modules.BERT, kwargs["bert_encoder"], dropout=kwargs["emb_dropouts"])
        embs[Modules.BERT] = emb

    # if kwargs["morphlex_embeddings_file"]:
    #     embs[Modules.MorphLex] = PretrainedEmbedding(
    #         vocab_map=dicts[Dicts.MorphLex],
    #         embeddings=embeddings[Dicts.MorphLex],
    #         freeze=kwargs["morphlex_freeze"],
    #         dropout=kwargs["emb_dropouts"],
    #     )
    # if kwargs["pretrained_word_embeddings_file"]:
    #     embs[Modules.Pretrained] = PretrainedEmbedding(
    #         vocab_map=dicts[Dicts.Pretrained],
    #         embeddings=embeddings[Dicts.Pretrained],
    #         freeze=True,
    #         dropout=kwargs["emb_dropouts"],
    #     )
    if kwargs["word_embedding_dim"]:
        embs[Modules.Trained] = ClassicWordEmbedding(
            Modules.Trained,
            dicts[Dicts.Tokens],
            kwargs["word_embedding_dim"],
            dropout=kwargs["emb_dropouts"],
        )
    character_embedding = CharacterEmbedding(
        Modules.Characters,
        dicts[Dicts.Chars],
        embedding_dim=kwargs["char_emb_dim"],
        dropout=kwargs["emb_dropouts"],
    )
    char_as_word = CharacterAsWordEmbedding(
        Modules.CharactersToTokens,
        character_embedding=character_embedding,
        char_lstm_layers=kwargs["char_lstm_layers"],
        char_lstm_dim=kwargs["char_lstm_dim"],
        dropout=kwargs["emb_dropouts"],
    )
    if kwargs["char_lstm_layers"]:
        embs[Modules.CharactersToTokens] = char_as_word
    decoders: Dict[str, Decoder] = {}
    if kwargs["tagger"]:
        log.info("Training Tagger")
        decoders[Modules.Tagger] = Tagger(
            key=Modules.Tagger,
            vocab_map=dicts[Dicts.FullTag],
            # TODO: Fix so that we define the Encoder the Tagger accepts
            encoder=embs[Modules.BERT],
            encoder_key=Modules.BERT,
            weight=kwargs["tagger_weight"],
        )
    if kwargs["lemmatizer"]:
        tag_embedding = ClassicWordEmbedding(
            key=Modules.TagEmbedding,
            vocab_map=dicts[Dicts.FullTag],
            embedding_dim=kwargs["tag_embedding_dim"],
            padding_idx=dicts[Dicts.FullTag].w2i[PAD],
            dropout=kwargs["emb_dropouts"],
        )
        log.info("Training Lemmatizer")
        char_decoder = CharacterDecoder(
            key=Modules.Lemmatizer,
            tag_encoder=tag_embedding,
            characters_to_tokens_encoder=char_as_word,
            vocab_map=dicts[Dicts.Chars],
            hidden_dim=kwargs["lemmatizer_hidden_dim"],
            char_rnn_input_dim=0 if not kwargs["lemmatizer_accept_char_rnn_last"] else char_as_word.output_dim,
            attention_dim=char_as_word.output_dim,
            char_attention=kwargs["lemmatizer_char_attention"],
            num_layers=kwargs["lemmatizer_num_layers"],
            dropout=kwargs["emb_dropouts"],
        )
        decoders[Modules.Lemmatizer] = char_decoder
    model = EncodersDecoders(encoders=embs, decoders=decoders).to(core.device)
    return model


# fmt: off
@cli.command()
@click.argument("training_files", nargs=-1)
@click.argument("test_file")
@click.argument("output_dir", default="./out")
@click.option("--gpu/--no_gpu", default=False)
@click.option("--save_model/--no_save_model", default=False)
@click.option("--save_vocab/--no_save_vocab", default=False)
@click.option("--tagger/--no_tagger", is_flag=True, default=False, help="Train tagger")
@click.option("--tagger_weight", default=1.0, help="Value to multiply tagging loss")
@click.option("--tagger_embedding", default="bilstm", help="The embedding to feed to the Tagger, see pos.model.Modules.")
@click.option("--tagger_ignore_e_x/--no_tagger_ignore_e_x", is_flag=True, default=True)
@click.option("--lemmatizer/--no_lemmatizer", is_flag=True, default=False, help="Train lemmatizer")
@click.option("--lemmatizer_weight", default=0.1, help="Value to multiply lemmatizer loss")
@click.option("--lemmatizer_accept_char_rnn_last/--no_lemmatizer_accept_char_rnn_last", default=False, help="Should the Character RNN last hidden state be input to Lemmatizer")
@click.option("--lemmatizer_hidden_dim", default=128, help="The hidden dim of the decoder RNN.")
@click.option("--lemmatizer_char_dim", default=64, help="The character embedding dim.")
@click.option("--lemmatizer_num_layers", default=1, help="The number of layers in Lemmatizer RNN.")
@click.option("--lemmatizer_char_attention/--no_lemmatizer_char_attention", default=True, help="Attend over characters?")
@click.option("--known_chars_file", default=None, help="A file which contains the characters the model should know. File should be a single line, the line is split() to retrieve characters.",)
@click.option("--char_lstm_layers", default=0, help="The number of layers in character LSTM embedding. Set to 0 to disable.")
@click.option("--char_lstm_dim", default=128, help="The size of the hidden dim in character RNN.")
@click.option("--char_emb_dim", default=64, help="The embedding size for characters.")
@click.option("--morphlex_embeddings_file", default=None, help="A file which contains the morphological embeddings.")
@click.option("--morphlex_freeze/--no_morphlex_freeze", is_flag=True, default=True)
@click.option("--pretrained_word_embeddings_file", default=None, help="A file which contains pretrained word embeddings. See implementation for supported formats.")
@click.option("--word_embedding_dim", default=0, help="The word/token embedding dimension. Set to 0 to disable word embeddings.")
@click.option("--bert_encoder", default=None, help="A folder which contains a pretrained BERT-like model. Set to None to disable.")
@click.option("--main_lstm_layers", default=1, help="The number of bilstm layers to use in the encoder.")
@click.option("--main_lstm_dim", default=128, help="The dimension of the lstm to use in the encoder.")
@click.option("--emb_dropouts", default=0.0, help="The dropout to use for Embeddings.")
@click.option("--label_smoothing", default=0.1)
@click.option("--learning_rate", default=5e-5)
@click.option("--epochs", default=20)
@click.option("--batch_size", default=16)
@click.option("--optimizer", default="adam", type=click.Choice(["sgd", "adam"], case_sensitive=False), help="The optimizer to use.")
@click.option("--scheduler", default="multiply", type=click.Choice(["none", "multiply", "plateau"], case_sensitive=False), help="The learning rate scheduler to use.")
# fmt: on
def train_and_tag(**kwargs):
    """Train a POS tagger and/or lemmatizer on intpus and write out the tagged test file.

    training_files: Files to use for training (supports multiple files = globbing).
    All training files should be .tsv, with two/three columns, the token, tag, lemma.
    test_file: Same format as training_files. Used to evaluate the model.
    output_dir: The directory to write out model and results.
    """
    pprint(kwargs)
    set_seed()
    set_device(gpu_flag=kwargs["gpu"])

    # Read train and test data
    unchunked_train_ds = read_datasets(kwargs["training_files"])
    unchunked_test_ds = read_datasets([kwargs["test_file"]])
    # Set configuration values and create mappers
    embeddings, dicts = load_dicts(
        train_ds=unchunked_train_ds,
        ignore_e_x=kwargs["tagger_ignore_e_x"],
        pretrained_word_embeddings_file=kwargs["pretrained_word_embeddings_file"],
        morphlex_embeddings_file=kwargs["morphlex_embeddings_file"],
        known_chars_file=kwargs["known_chars_file"],
    )
    model = build_model(kwargs=kwargs, dicts=dicts)
    if Modules.BERT in model.encoders.keys():
        tok: PreTrainedTokenizerFast = model.encoders[Modules.BERT].tokenizer  # type: ignore
        max_length = model.encoders[Modules.BERT].max_length
        train_ds = chunk_dataset(unchunked_train_ds, tok, max_length)
        test_ds = chunk_dataset(unchunked_test_ds, tok, max_length)
    else:
        train_ds = unchunked_train_ds
        test_ds = unchunked_test_ds

    # Train a model
    print_model(model)

    train_dl = DataLoader(
        train_ds,
        collate_fn=collate_fn,  # type: ignore
        shuffle=True,
        batch_size=kwargs["batch_size"],
    )
    test_dl = DataLoader(
        test_ds,
        collate_fn=collate_fn,  # type: ignore
        shuffle=False,
        batch_size=kwargs["batch_size"] * 10,
    )
    criterion = get_criterion(decoders=model.decoders, label_smoothing=kwargs["label_smoothing"])  # type: ignore
    optimizer = get_optimizer(model.parameters(), kwargs["optimizer"], kwargs["learning_rate"])
    scheduler = get_scheduler(optimizer, kwargs["scheduler"])
    evaluators = {}
    if Modules.Tagger in model.decoders:
        evaluators[Modules.Tagger] = evaluate.TaggingEvaluation(
            test_dataset=test_ds,
            train_vocab=train_ds.get_vocab(),
            external_vocabs=evaluate.ExternalVocabularies(
                morphlex_tokens=Vocab.from_file(MORPHLEX_VOCAB_PATH),
                pretrained_tokens=Vocab.from_file(PRETRAINED_VOCAB_PATH),
            ),
        ).tagging_accuracy
    if Modules.Lemmatizer in model.decoders:
        evaluators[Modules.Lemmatizer] = evaluate.LemmatizationEvaluation(
            test_dataset=test_ds,
            train_vocab=train_ds.get_vocab(),
            train_lemmas=Vocab.from_symbols(train_ds.get_field(Fields.GoldLemmas)),
        ).lemma_accuracy

    # Write all configuration to disk
    output_dir = pathlib.Path(kwargs["output_dir"])
    write_hyperparameters(output_dir / "hyperparamters.json", (kwargs))

    # Start the training
    run_epochs(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        evaluators=evaluators,
        train_data_loader=train_dl,
        test_data_loader=test_dl,
        epochs=kwargs["epochs"],
        output_dir=output_dir,
    )
    _, values = tag_data_loader(
        model=model,
        data_loader=test_dl,
    )
    log.info("Writing predictions, dictionaries and model")
    for module_name, value in values.items():
        test_ds = test_ds.add_field(value, MODULE_TO_FIELD[module_name])
    # Dechunk - if we chunked
    if len(test_ds) != len(unchunked_test_ds):
        test_ds = dechunk_dataset(unchunked_test_ds, test_ds)

    test_ds.to_tsv_file(str(output_dir / "predictions.tsv"))

    with output_dir.joinpath("known_toks.txt").open("w+") as f:
        for token in Vocab.from_symbols(train_ds.get_field(Fields.Tokens)):
            f.write(f"{token}\n")
    if Fields.GoldLemmas in train_ds.fields:
        with output_dir.joinpath("known_lemmas.txt").open("w+") as f:
            for lemma in Vocab.from_symbols(train_ds.get_field(Fields.GoldLemmas)):
                f.write(f"{lemma}\n")
    if kwargs["bert_encoder"]:
        write_bert_config(pathlib.Path(kwargs["bert_encoder"]), output_dir)
    if kwargs["save_vocab"]:
        save_location = output_dir.joinpath("dictionaries.pickle")
        with save_location.open("wb+") as f:
            pickle.dump(dicts, f)
    if kwargs["save_model"]:
        save_location = output_dir.joinpath("tagger.pt")
        torch.save(model.state_dict(), str(save_location))
    log.info("Done!")


def write_bert_config(in_dir: pathlib.Path, out_dir: pathlib.Path):
    import shutil

    for file_name in ("config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"):
        shutil.copy(in_dir.joinpath(file_name), out_dir.joinpath(file_name))


def write_hyperparameters(path, hyperparameters):
    """Write hyperparameters to disk."""
    with path.open(mode="w") as f:
        json.dump(hyperparameters, f, indent=4)


@cli.command()
@click.argument("model_file")
@click.argument("data_in", type=str)
@click.argument("output", type=str)
@click.option("--device", default="cpu", help="The device to use, 'cpu' or 'cuda:0' for GPU.")
@click.option(
    "--batch_size",
    default=16,
    help="The number of sentences to process at once. Works best to have this high for a GPU.",
)
def tag(model_file, data_in, output, device, batch_size):
    """Tag tokens in a file.

    Args:
        model_file: A filepath to a trained model.
        data_in: A filepath of a file formatted as: token per line, sentences separated with newlines (empty line).
        output: A filepath. Output is formatted like the input, but after each token there is a tab and then the tag.
        device: cpu or gpu:0
        contains_tags: A flag. Set it if the data_in already contains tags.
    """
    tagger = api_tagger(model_file=model_file, device=device)
    log.info("Reading dataset")
    ds = FieldedDataset.from_file(data_in)
    predicted_tags = tagger.tag_bulk(dataset=ds, batch_size=batch_size)
    ds = ds.add_field(predicted_tags, Fields.Tags)
    log.info("Writing results")
    ds.to_tsv_file(output)
    log.info("Done!")


# fmt: off
@cli.command()
@click.argument("predictions")
@click.argument("fields")
@click.option("--morphlex_vocab", help="The location of the morphlex vocabulary.", default=MORPHLEX_VOCAB_PATH)
@click.option("--pretrained_vocab", help="The location of the pretrained vocabulary.", default=PRETRAINED_VOCAB_PATH)
@click.option("--train_tokens", help="The location of the training tokens used to train the model.", default=None)
@click.option("--train_lemmas", help="The location of the training lemmas used to train the model.", default=None)
@click.option("--criteria", type=click.Choice(["accuracy", "profile", "confusion"], case_sensitive=False), help="Which criteria to evaluate.", default="accuracy")
@click.option("--feature", type=click.Choice(["tags", "lemmas"], case_sensitive=False), help="Which feature to evaluate.", default="tags")
@click.option("--up_to", help="For --criteria profile, the number of errors to report", default=30)
# fmt: on
def evaluate_predictions(
    predictions, fields, morphlex_vocab, pretrained_vocab, train_tokens, train_lemmas, criteria, feature, up_to
):
    """Evaluate predictions.

    Evaluate a single prediction file.

    Args:
        predictions: The tagged test file.
        fields: The fields present in the test file. Separated with ',', f.ex. 'tokens,gold_tags,tags'.
        morphlex_vocab: The location of the morphlex vocab.
        pretrained_vocab: The location of the pretrained vocab.
        train_tokens: The location of the tokens used in training.
        train_lemmas: The location of the lemmas used in training.
        criteria: The evaluation criteria
        feature: Lemmas or tags?
        up_to: The number of errors for profile.
    """
    click.echo(f"Evaluating: {predictions}")
    ds = FieldedDataset.from_file(predictions, fields=tuple(fields.split(",")))
    if criteria == "accuracy":
        result = evaluate.get_accuracy_from_files(
            feature,
            ds,
            train_tokens=train_tokens,
            train_lemmas=train_lemmas,
            morphlex_vocab=morphlex_vocab,
            pretrained_vocab=pretrained_vocab,
        )
        click.echo(evaluate.format_result(result))
    elif criteria == "profile":
        result = evaluate.get_profile_from_files(
            feature,
            ds,
            train_tokens=train_tokens,
            train_lemmas=train_lemmas,
            morphlex_vocab=morphlex_vocab,
            pretrained_vocab=pretrained_vocab,
        )
        click.echo(evaluate.format_profile(result, up_to=up_to))
    else:  # confusion
        train_lemmas = Vocab.from_file(train_lemmas)
        morphlex_vocab = Vocab.from_file(morphlex_vocab)
        pretrained_vocab = Vocab.from_file(pretrained_vocab)
        evaluation = evaluate.TaggingLemmatizationEvaluation(
            test_dataset=ds,
            train_vocab=train_tokens,
            external_vocabs=evaluate.ExternalVocabularies(morphlex_vocab, pretrained_vocab),
            train_lemmas=train_lemmas,
        )
        click.echo(evaluation.lemma_tag_confusion_matrix())


# fmt: off
@cli.command()
@click.argument("directories", nargs=-1)
@click.argument("fields")
@click.option("--morphlex_vocab", help="The location of the morphlex vocabulary.", default=MORPHLEX_VOCAB_PATH)
@click.option("--pretrained_vocab", help="The location of the pretrained vocabulary.", default=PRETRAINED_VOCAB_PATH)
@click.option("--criteria", type=click.Choice(["accuracy", "profile"], case_sensitive=False), help="Which criteria to evaluate.", default="accuracy")
@click.option("--feature", type=click.Choice(["tags", "lemmas"], case_sensitive=False), help="Which feature to evaluate.", default="tags")
@click.option("--up_to", help="For --criteria profile, the number of errors to report", default=30)
@click.option("--skip_gold_ex/--no_skip_gold_ex", is_flag=True, default=True, help="When evaluating accurcy, should we ignore 'e' and 'x' gold tags?")
# fmt: on
def evaluate_experiments(
    directories,
    fields,
    pretrained_vocab,
    morphlex_vocab,
    criteria,
    feature,
    up_to,
    skip_gold_ex,
):
    """Evaluate the model predictions in the directory. If the directory contains other directories, it will recurse into it."""
    directories = [pathlib.Path(directory) for directory in directories]
    fields = fields.split(",")
    accuracy_results = []
    profile = Counter()
    for directory in directories:
        ds = FieldedDataset.from_file(str(directory / "predictions.tsv"), fields=fields)
        train_tokens = str(directory / "known_toks.txt")
        train_lemmas = str(directory / "known_lemmas.txt")
        if criteria == "accuracy":
            accuracy_results.append(
                evaluate.get_accuracy_from_files(
                    feature,
                    ds,
                    train_tokens=train_tokens,
                    train_lemmas=train_lemmas,
                    morphlex_vocab=morphlex_vocab,
                    pretrained_vocab=pretrained_vocab,
                    skip_gold_ex=skip_gold_ex,
                )
            )
        elif criteria == "profile":
            profile += evaluate.get_profile_from_files(
                feature,
                ds,
                train_tokens=train_tokens,
                train_lemmas=train_lemmas,
                morphlex_vocab=morphlex_vocab,
                pretrained_vocab=pretrained_vocab,
                skip_gold_ex=skip_gold_ex,
            )
    if criteria == "accuracy":
        click.echo(evaluate.format_results(evaluate.all_accuracy_average(accuracy_results)))
    elif criteria == "profile":
        click.echo(f"Total errors: {sum(profile.values())}")
        pred_x_e_pattern = re.compile("^[ex] >")
        click.echo(
            f"Errors caused by model predicting 'x' and 'e': {sum(value for key, value in profile.items() if pred_x_e_pattern.search(key) is not None)}"
        )
        click.echo(evaluate.format_profile(profile, up_to=up_to))
