#!/usr/bin/env python
"""The main entrypoint to training and running a POS-tagger."""
import pickle
import random
import logging
import json
import pathlib
from typing import Dict, List
from functools import reduce, partial
from operator import add

import click
import flair
import torch
from torch.utils.data import DataLoader
import numpy as np

from .data import (
    read_morphlex,
    vocab_map_from_dataset,
    tag_data_loader,
    MAPPING_KEY_TAGS,
    EOS,
    EOS_ID,
    SOS,
    SOS_ID,
    PAD,
    PAD_ID,
    UNK,
    UNK_ID,
    SequenceTaggingDataset,
    TokenizedDataset,
)
from .core import (
    Vocab,
    VocabMap,
    Dataset,
    SimpleDataset,
    TaggedSentence,
)
from .model import ABLTagger, PretrainedWordEmbeddings, load_transformer_embeddings
from .train import (
    print_tagger,
    get_criterion,
    get_parameter_groups,
    get_optimizer,
    get_scheduler,
    run_epochs,
)
from .api import Tagger
from .evaluate import Experiment
from .utils import write_tsv

DEBUG = False
log = logging.getLogger(__name__)


@click.group()
@click.option("--debug/--no-debug", default=False)
@click.option("--log/--no-log", default=False)
def cli(debug, log):
    """Entrypoint to the program. --debug flag from command line is caught here."""
    log_level = logging.INFO
    if debug or log:
        log_level = logging.DEBUG
    logging.basicConfig(format="%(asctime)s - %(message)s", level=log_level)
    global DEBUG
    DEBUG = debug


@cli.command()
@click.argument("filepaths", nargs=-1)
@click.argument("output", type=click.File("w"))
def gather_tags(filepaths, output):
    """Read all input tsv files and extract all tags in files."""
    ds = read_dataset(filepaths)
    tags = Vocab.from_symbols(y for x, y in ds)
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
    ds = read_dataset(filepaths)
    tokens = Vocab.from_symbols(ds.unpack_dataset()[0])

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
    "--bert_encoder",
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
def train_and_tag(**kwargs):
    """Train a POS tagger on intpus and write out the tagged the test files.

    training_files: Files to use for training (supports multiple files = globbing).
    All training files should be .tsv, with two columns, the token and tag.
    test_file: Same format as training_files. Used to evaluate the model.
    output_dir: The directory to write out model and results.
    """
    print(kwargs)
    set_seed()
    device = set_device(gpu_flag=kwargs["gpu"])

    # Read train and test data
    train_ds = read_dataset(kwargs["training_files"], max_length=128)
    test_ds = read_dataset([kwargs["test_file"]], max_length=128)

    # Set configuration values and create mappers
    dictionaries: Dict[str, VocabMap] = {}
    extras: Dict[str, np.array] = {}

    model_parameters = {
        "w_emb": w_emb,
        "c_emb": c_emb,
        "m_emb": m_emb,
        "char_dim": len(dictionaries["c_map"]) if "c_map" in dictionaries else 0,
        "token_dim": len(dictionaries["w_map"]) if w_emb == "standard" else 0,
        "tags_dim": len(dictionaries["t_map"]),
        "emb_char_dim": 20,  # The characters are mapped to this dim
        "char_lstm_dim": 64,  # The character LSTM will output with this dim
        "char_lstm_layers": kwargs[
            "char_lstm_layers"
        ],  # The character LSTM will output with this dim
        "emb_token_dim": kwargs[
            "word_embedding_dim"
        ],  # The tokens are mapped to this dim
        "main_lstm_dim": 64,  # The main LSTM dim will output with this dim
        "main_lstm_layers": kwargs[
            "main_lstm_layers"
        ],  # The main LSTM dim will output with this dim
        "final_layer": kwargs["final_layer"],
        "final_layer_attention_heads": kwargs["final_layer_attention_heads"],
        "final_dim": kwargs[
            "final_dim"
        ],  # The main LSTM time-steps will be mapped to this dim
        "morphlex_extra_dim": kwargs["morphlex_extra_dim"],
        "transformer_embedding": transformer_embedding,
        "lstm_dropouts": 0.1,
        "input_dropouts": 0.0,
        "noise": 0.1,  # Noise to main_in, to main_bilstm
        "morphlex_freeze": kwargs["morphlex_freeze"],
    }

    # Run parameters
    parameters = {
        "training_files": kwargs["training_files"],
        "test_file": kwargs["test_file"],
        "epochs": kwargs["epochs"],
        "batch_size": kwargs["batch_size"],
        "learning_rate": kwargs["learning_rate"],
        "word_embedding_lr": kwargs["word_embedding_lr"],
        "scheduler": kwargs["scheduler"],
        "label_smoothing": kwargs["label_smoothing"],
        "optimizer": kwargs["optimizer"],
    }

    # Write all configuration to disk
    output_dir = pathlib.Path(kwargs["output_dir"])
    write_hyperparameters(
        output_dir / "hyperparamters.json", dict(parameters).update(model_parameters)
    )
    # Train a model

    tagger = ABLTagger(**{**model_parameters, **extras}).to(device)
    print_tagger(tagger)

    criterion = get_criterion(**kwargs)
    parameter_groups = get_parameter_groups(tagger.named_parameters(), **kwargs)
    optimizer = get_optimizer(parameter_groups, **kwargs)
    scheduler = get_scheduler(optimizer, **kwargs)
    evaluator = partial(
        Experiment.from_predictions, test_ds=test_ds, dicts=dictionaries
    )

    run_epochs(
        model=tagger,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        evaluator=evaluator,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        dictionaries=dictionaries,
        epochs=kwargs["epochs"],
        output_dir=output_dir,
    )
    test_tags_tagged, _ = tag_data_loader(
        model=tagger,
        data_loader=test_data_loader,
        tag_map=dictionaries[MAPPING_KEY_TAGS],
    )
    log.info("Writing predictions, dictionaries and model")
    with (output_dir / "predictions.tsv").open("w") as f:
        write_tsv(f, (*test_ds.unpack(), test_tags_tagged))
    with (output_dir / "known_toks.txt").open("w+") as f:
        for token in Vocab.from_symbols(x for x, y in train_ds):
            f.write(f"{token}\n")
    if kwargs["save_vocab"]:
        save_location = output_dir.joinpath("dictionaries.pickle")
        with save_location.open("wb+") as f:
            pickle.dump(dictionaries, f)
    if kwargs["save_model"]:
        save_location = output_dir.joinpath("tagger.pt")
        torch.save(tagger, str(save_location))
    log.info("Done!")


def read_dataset(file_paths: List[str], max_length=-1) -> SequenceTaggingDataset:
    """Read tagged datasets from multiple files."""
    ds = SequenceTaggingDataset(
        reduce(
            add, (Dataset.from_file(training_file) for training_file in file_paths), (),
        )
    )
    if max_length != -1:
        # We want to filter out sentences which are too long (and throw them away, for now)
        ds = Dataset(TaggedSentence((x, y)) for x, y in ds if len(x) <= max_length)
    # DEBUG - read a subset of the data
    if DEBUG:
        debug_size = 100
        ds = Dataset(ds[:debug_size])
    return ds


def set_seed(seed=42):
    """Set the seed on all platforms. 0 for need specific seeding."""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore


def write_hyperparameters(path, hyperparameters):
    """Write hyperparameters to disk."""
    with path.open(mode="w") as f:
        json.dump(hyperparameters, f, indent=4)


def set_device(gpu_flag=False):
    """Set the torch device."""
    if DEBUG:
        # We do not use GPU when debugging
        gpu_flag = False
    if torch.cuda.is_available() and gpu_flag:
        device = torch.device("cuda")
        # Torch will use the allocated GPUs from environment variable CUDA_VISIBLE_DEVICES
        # --gres=gpu:titanx:2
        flair.device = device
        log.info(f"Using {torch.cuda.device_count()} GPUs")
    else:
        device = torch.device("cpu")
        threads = 1
        # Set the number of threads to use for CPU
        torch.set_num_threads(threads)
        flair.device = device
        log.info(f"Using {threads} CPU threads")
    return device


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
    tagger = Tagger(
        model_file=model_file, dictionaries_file=dictionaries_file, device=device
    )
    log.info("Reading dataset")
    if contains_tags:
        ds_with_tags = SequenceTaggingDataset.from_file(data_in)
        ds, gold = ds_with_tags.unpack()
    else:
        ds = TokenizedDataset.from_file(data_in)
    predicted_tags = tagger.tag_bulk(dataset=ds, batch_size=16)
    log.info("Writing results")
    with open(output, "w+") as f:
        if contains_tags:
            write_tsv(f=f, data=(ds, gold, predicted_tags))
        else:
            write_tsv(f=f, data=(ds, predicted_tags))
    log.info("Done!")
