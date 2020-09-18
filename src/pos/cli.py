#!/usr/bin/env python
"""The main entrypoint to training and running a POS-tagger."""
import pickle
import random
import logging
import json
import pathlib
from typing import Dict
from functools import reduce, partial
from operator import add

import click
import torch
import numpy as np

from . import data, train, api
from .types import write_tsv

DEBUG = False
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
log = logging.getLogger()


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    """Entrypoint to the program. --debug flag from command line is caught here."""
    global DEBUG
    DEBUG = debug


@cli.command()
@click.argument("filepaths", nargs=-1)
@click.argument("output", type=click.File("w+"))
def gather_tags(filepaths, output):
    """Read all input tsv files and extract all tags in files."""
    ds = reduce(add, (data.Dataset.from_file(filepath) for filepath in filepaths), (),)
    tags = data.Vocab.from_symbols(y for x, y in ds)
    for tag_ in sorted(list(tags)):
        output.write(f"{tag_}\n")


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
    ds = reduce(add, (data.Dataset.from_file(filepath) for filepath in filepaths), (),)
    tokens = data.Vocab.from_symbols(ds.unpack_dataset()[0])

    log.info(f"Number of tokens={len(tokens)}")
    with open(embedding) as f:
        if emb_format == "bin":
            emb_dict = data.emb_pairs_to_dict(f, data.bin_str_to_emb_pair)
            for token, value in emb_dict.items():
                if token in tokens:  # pylint: disable=unsupported-membership-test
                    output.write(f"{token};[{','.join((str(x) for x in value))}]\n")
        elif emb_format == "wemb":
            emb_dict = data.emb_pairs_to_dict(f, data.wemb_str_to_emb_pair)
            output.write("x x\n")
            for token, value in emb_dict.items():
                if token in tokens:  # pylint: disable=unsupported-membership-test
                    output.write(f"{token} {' '.join((str(x) for x in value))}\n")


@cli.command()
@click.argument("training_files", nargs=-1)
@click.argument("test_file")
@click.argument("output_dir", default="./out")
@click.option("--gpu/--no_gpu", default=False)
@click.option("--save_model/--no_save_model", default=False)
@click.option("--save_vocab/--no_save_vocab", default=False)
@click.option(
    "--known_chars_file",
    help="A file which contains the characters the model should know. Omit, to disable character embeddings. "
    + "File should be a single line, the line is split() to retrieve characters.",
)
@click.option("--char_lstm_layers", default=1)
@click.option(
    "--morphlex_embeddings_file",
    default=None,
    help="A file which contains the morphological embeddings.",
)
@click.option("--morphlex_freeze", is_flag=True, default=False)
@click.option(
    "--morphlex_extra_dim",
    default=-1,
    help="The dimension to map morphlex embeddings to. -1 to disable.",
)
@click.option(
    "--pretrained_word_embeddings_file",
    default=None,
    help="A file which contains pretrained word embeddings. See implementation for supported formats.",
)
@click.option(
    "--word_embedding_dim",
    default=-1,
    help="The word/token embedding dimension. Set to -1 to disable word embeddings.",
)
@click.option(
    "--word_embedding_lr", default=0.2, help="The word/token embedding learning rate."
)
@click.option(
    "--pretrained_model_folder",
    default=None,
    help="A folder which contains a pretrained BERT-like model.",
)
@click.option("--main_lstm_layers", default=1)
@click.option(
    "--final_layer",
    default="dense",
    type=click.Choice(["dense", "none", "attention"], case_sensitive=False),
    help="The type of final layer to use.",
)
@click.option(
    "--final_layer_attention_heads",
    default=1,
    help="The number of attention heads to use.",
)
@click.option("--final_dim", default=32)
@click.option("--label_smoothing", default=0.0)
@click.option("--learning_rate", default=0.20)
@click.option("--epochs", default=20)
@click.option("--batch_size", default=32)
@click.option(
    "--optimizer",
    default="sgd",
    type=click.Choice(["sgd", "adam"], case_sensitive=False),
    help="The optimizer to use.",
)
@click.option(
    "--scheduler",
    default="multiply",
    type=click.Choice(["multiply", "plateau"], case_sensitive=False),
    help="The learning rate scheduler to use.",
)
def train_and_tag(
    training_files,
    test_file,
    output_dir,
    known_chars_file,
    morphlex_embeddings_file,
    morphlex_freeze,
    morphlex_extra_dim,
    pretrained_word_embeddings_file,
    word_embedding_lr,
    word_embedding_dim,
    pretrained_model_folder,
    main_lstm_layers,
    char_lstm_layers,
    label_smoothing,
    final_layer,
    final_layer_attention_heads,
    final_dim,
    batch_size,
    save_model,
    save_vocab,
    gpu,
    optimizer,
    learning_rate,
    epochs,
    scheduler,
):
    """Train a POS tagger on intpus and write out the tagged the test files.

    training_files: Files to use for training (supports multiple files = globbing).
    All training files should be .tsv, with two columns, the token and tag.
    test_file: Same format as training_files. Used to evaluate the model.
    output_dir: The directory to write out model and results.
    """
    # Set the seed on all platforms
    SEED = 42
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True  # type: ignore

    if torch.cuda.is_available() and gpu:
        device = torch.device("cuda")
        # Torch will use the allocated GPUs from environment variable CUDA_VISIBLE_DEVICES
        # --gres=gpu:titanx:2
        log.info(f"Using {torch.cuda.device_count()} GPUs")
    else:
        device = torch.device("cpu")
        threads = 1
        # Set the number of threads to use for CPU
        torch.set_num_threads(threads)
        log.info(f"Using {threads} CPU threads")

    # Run parameters
    parameters = {
        "training_files": training_files,
        "test_file": test_file,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "word_embedding_lr": word_embedding_lr,
        "scheduler": scheduler,
        "label_smoothing": label_smoothing,
        "optimizer": optimizer,
    }

    # Read train and test data
    train_ds = data.Dataset(
        reduce(
            add,
            (data.Dataset.from_file(training_file) for training_file in training_files),
            (),
        )
    )
    test_ds = data.Dataset.from_file(test_file)

    # DEBUG - read a subset of the data
    if DEBUG:
        device = torch.device("cpu")
        threads = 1
        # Set the number of threads to use for CPU
        torch.set_num_threads(threads)
        log.info(f"Using {threads} CPU threads")
        train_ds = data.Dataset(
            train_ds[:batch_size]  # pylint: disable=unsubscriptable-object
        )
        test_ds = data.Dataset(
            test_ds[:batch_size]  # pylint: disable=unsubscriptable-object
        )

    # Set configuration values and create mappers
    dictionaries: Dict[str, data.VocabMap] = {}
    extras: Dict[str, np.array] = {}

    # WORD EMBEDDINGS
    # By default we do not use word-embeddings.
    w_emb = "none"
    if pretrained_word_embeddings_file is not None:
        # If a file is provided, we use it.
        w_emb = "pretrained"
        with open(pretrained_word_embeddings_file) as f:
            it = iter(f)
            # pop the number of vectors and dimension
            next(it)
            embedding_dict = data.emb_pairs_to_dict(it, data.wemb_str_to_emb_pair)
            w_map, w_embedding = data.map_embedding(
                embedding_dict=embedding_dict,
                filter_on=None,
                special_tokens=[(data.UNK, data.UNK_ID), (data.PAD, data.PAD_ID)],
            )
            dictionaries["w_map"] = w_map
            extras["word_embeddings"] = torch.from_numpy(w_embedding).float().to(device)
    # We are given a pretrained BERT like model, we use it.
    elif pretrained_model_folder is not None:
        w_emb = "electra"
    elif word_embedding_dim != -1:
        # No file is given and the dimension is not -1 we train from scratch.
        w_emb = "standard"
        dictionaries["w_map"] = data.VocabMap(
            data.Vocab.from_symbols((x for x, y in train_ds)),
            special_tokens=[(data.PAD, data.PAD_ID), (data.UNK, data.UNK_ID)],
        )

    # MORPHLEX EMBEDDINGS
    # By default we do not use morphlex embeddings.
    m_emb = "none"
    if morphlex_embeddings_file is not None:
        # File is provided, use it.
        m_emb = "standard"
        with open(morphlex_embeddings_file) as f:
            it = iter(f)
            embedding_dict = data.emb_pairs_to_dict(it, data.bin_str_to_emb_pair)
            m_map, m_embedding = data.map_embedding(
                embedding_dict=embedding_dict,
                filter_on=None,
                special_tokens=[(data.UNK, data.UNK_ID), (data.PAD, data.PAD_ID)],
            )
            dictionaries["m_map"] = m_map
            extras["morphlex_embeddings"] = (
                torch.from_numpy(m_embedding).float().to(device)
            )
        if morphlex_extra_dim != -1:
            m_emb = "extra"

    # CHARACTER EMBEDDINGS
    # By default we do not use character embeddings.
    c_emb = "none"
    if known_chars_file is not None:
        # A file is given, use it.
        c_emb = "standard"
        char_vocab = data.Vocab.from_file(known_chars_file)
        dictionaries["c_map"] = data.VocabMap(
            char_vocab,
            special_tokens=[
                (data.UNK, data.UNK_ID),
                (data.PAD, data.PAD_ID),
                (data.EOS, data.EOS_ID),
                (data.SOS, data.SOS_ID),
            ],
        )

    # TAGS (POS)
    dictionaries["t_map"] = data.VocabMap(
        data.Vocab.from_symbols((y for x, y in train_ds)),
        special_tokens=[(data.PAD, data.PAD_ID), (data.UNK, data.UNK_ID),],
    )

    model_parameters = {
        "w_emb": w_emb,
        "c_emb": c_emb,
        "m_emb": m_emb,
        "char_dim": len(dictionaries["c_map"]) if "c_map" in dictionaries else 0,
        "token_dim": len(dictionaries["w_map"]) if w_emb == "standard" else 0,
        "tags_dim": len(dictionaries["t_map"]),
        "emb_char_dim": 20,  # The characters are mapped to this dim
        "char_lstm_dim": 64,  # The character LSTM will output with this dim
        "char_lstm_layers": char_lstm_layers,  # The character LSTM will output with this dim
        "emb_token_dim": word_embedding_dim,  # The tokens are mapped to this dim
        "main_lstm_dim": 64,  # The main LSTM dim will output with this dim
        "main_lstm_layers": main_lstm_layers,  # The main LSTM dim will output with this dim
        "final_layer": final_layer,
        "final_layer_attention_heads": final_layer_attention_heads,
        "final_dim": final_dim,  # The main LSTM time-steps will be mapped to this dim
        "morphlex_extra_dim": morphlex_extra_dim,
        "lstm_dropouts": 0.1,
        "input_dropouts": 0.0,
        "noise": 0.1,  # Noise to main_in, to main_bilstm
        "morphlex_freeze": morphlex_freeze,
    }

    # Write all configuration to disk
    output_dir = pathlib.Path(output_dir)
    with output_dir.joinpath("hyperparamters.json").open(mode="+w") as f:
        json.dump({**parameters, **model_parameters}, f, indent=4)

    # Train a model
    data_loader = partial(
        data.data_loader,
        device=device,
        w_emb=w_emb,
        c_emb=c_emb,
        m_emb=m_emb,
        dictionaries=dictionaries,
    )
    tagger = train.run_training(
        run_parameters=parameters,
        model_parameters=model_parameters,
        model_extras=extras,
        data_loader=data_loader,
        dictionaries=dictionaries,
        device=device,
        train_ds=train_ds,
        test_ds=test_ds,
        output_dir=output_dir,
    )
    test_tags_tagged = tagger.tag_sents(
        data_loader=data_loader(
            dataset=test_ds, shuffle=False, batch_size=batch_size * 10,
        ),
        dictionaries=dictionaries,
        criterion=None,
    )
    log.info("Writing predictions, dictionaries and model")
    with (output_dir / "predictions.tsv").open("w") as f:
        write_tsv(f, (*test_ds.unpack(), test_tags_tagged))
    with (output_dir / "known_toks.txt").open("w+") as f:
        for token in data.Vocab.from_symbols(  # pylint: disable=not-an-iterable
            x for x, y in train_ds
        ):
            f.write(f"{token}\n")
    if save_vocab:
        save_location = output_dir.joinpath("dictionaries.pickle")
        with save_location.open("wb+") as f:
            pickle.dump(dictionaries, f)
    if save_model:
        save_location = output_dir.joinpath("tagger.pt")
        torch.save(tagger, str(save_location))
    log.info("Done!")


@cli.command()
@click.argument("model_file")
@click.argument("dictionaries_file")
@click.argument("data_in", type=str)
@click.argument("output", type=str)
@click.option(
    "--device", default="cpu", help="The device to use, 'cpu' or 'cuda:0' for GPU."
)
@click.option(
    "--contains_tags",
    is_flag=True,
    default=False,
    help="Does input data contain tags? Useful when predicting tags on a dataset which is already tagged (gold tags). All are written out: token\tgold\tpredicted",
)
def tag(model_file, dictionaries_file, data_in, output, device, contains_tags):
    """Tag tokens in a file.

    Args:
        model_file: A filepath to a trained model.
        dictionaries_file: A filepath to dictionaries (vocabulary mappings) for preprocessing.
        data_in: A filepath of a file formatted as: token per line, sentences separated with newlines (empty line).
        output: A filepath. Output is formatted like the input, but after each token there is a tab and then the tag.
    """
    tagger = api.Tagger(
        model_file=model_file, dictionaries_file=dictionaries_file, device=device
    )
    log.info("Reading dataset")
    if contains_tags:
        ds_with_tags = data.Dataset.from_file(data_in)
        ds, gold = ds_with_tags.unpack()
    else:
        ds = data.SimpleDataset.from_file(data_in)
    predicted_tags = tagger.tag_bulk(dataset=ds, batch_size=16)
    log.info("Writing results")
    with open(output, "w+") as f:
        if contains_tags:
            write_tsv(f=f, data=(ds, gold, predicted_tags))
        else:
            write_tsv(f=f, data=(ds, predicted_tags))
    log.info("Done!")
