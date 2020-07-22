#!/usr/bin/env python
"""The main entrypoint to training and running a POS-tagger."""
import pickle
import random
import logging
import json
import pathlib
from functools import partial

import click
import torch
import numpy as np

from pos import data
from pos import train

DEBUG = False


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    """Entrypoint to the program. --debug flag from command line is caught here."""
    global DEBUG
    DEBUG = debug


@cli.command()
@click.argument("inputs", nargs=-1)
@click.argument("output", type=click.File("w+"))
@click.option("--coarse", is_flag=True, help='Maps the tags "coarse".')
def gather_tags(inputs, output, coarse):
    """Read all input tsv files and extract all tags in files."""
    input_data = data.read_datasets(inputs)
    tags = data.get_vocab(y for x, y in input_data)
    for tag in sorted(list(tags)):
        output.write(f"{tag}\n")


@cli.command()
@click.argument("inputs", nargs=-1)
def fix_known_toks(inputs):
    """Add a known_toks.txt file in the directories provided in 'inputs'.

    Reads the hyperparameters.json, grabs the training files, parses and writes the vocabulary out.
    """
    for input in inputs:
        log.info(f"Fixing dir={input}")
        with open(f"{input}/hyperparamters.json", "r") as f:
            hyperparameters = json.load(f)
        ds = data.read_datasets(hyperparameters["training_files"])
        with open(f"{input}/known_toks.txt", "w+") as f:
            for token in data.get_vocab(x for x, y in ds):
                f.write(f"{token}\n")
    log.info("Done!")


@cli.command()
@click.argument("inputs", nargs=-1)
@click.argument("embedding", default=None)
@click.argument("output", type=click.File("w+"))
@click.argument("format", type=str)
def filter_embedding(inputs, embedding, output, format):
    """Filter an 'embedding' file based on the words which occur in the 'inputs' files.

    Result is written to the 'output' file with the same format as the embedding file.

    inputs: Files to use for filtering (supports multiple files = globbing).
    Files should be .tsv, with two columns, the token and tag.
    embedding: The embedding file to filter.
    output: The file to write the result to.
    format: The format of the embedding file being read, 'bin' or 'wemb'.
    """
    tokens = set()
    log.info(f"Reading files={inputs}")
    for input in inputs:
        toks, _, _ = data.read_tsv(input)
        tokens.update(*toks)
    log.info(f"Number of tokens={len(tokens)}")
    with open(embedding) as f:
        if format == "bin":
            emb_dict = data.read_bin_embedding(f)
            for token, value in emb_dict.items():
                if token in tokens:
                    output.write(f"{token};[{','.join((str(x) for x in value))}]\n")
        elif format == "wemb":
            emb_dict = data.read_word_embedding(f)
            output.write("x x\n")
            for token, value in emb_dict.items():
                if token in tokens:
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
@click.option(
    "--morphlex_embeddings_file",
    default=None,
    help="A file which contains the morphological embeddings.",
)
@click.option(
    "--pretrained_word_embeddings_file",
    default=None,
    help="A file which contains pretrained word embeddings. See implementation for supported formats.",
)
@click.option("--epochs", default=20)
@click.option("--batch_size", default=32)
@click.option("--char_lstm_layers", default=1)
@click.option("--main_lstm_layers", default=1)
@click.option("--final_dim", default=32)
@click.option("--label_smoothing", default=0.0)
@click.option("--learning_rate", default=0.20)
@click.option("--morphlex_freeze", is_flag=True, default=False)
@click.option(
    "--morphlex_extra_dim",
    default=-1,
    help="The dimension to map morphlex embeddings to. -1 to disable.",
)
@click.option(
    "--word_embedding_dim",
    default=128,
    help="The word/token embedding dimension. Set to -1 to disable word embeddings.",
)
@click.option(
    "--word_embedding_lr", default=0.002, help="The word/token embedding learning rate."
)
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
    main_lstm_layers,
    char_lstm_layers,
    label_smoothing,
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
    # Tracking experiments and visualization
    from torch.utils import tensorboard

    # Set configuration values
    w_emb = "none"
    if word_embedding_dim != -1:
        w_emb = "standard"
    elif pretrained_word_embeddings_file is not None:
        w_emb = "pretrained"

    m_emb = "none"
    if morphlex_embeddings_file is not None:
        m_emb = "standard"
        if morphlex_extra_dim != -1:
            m_emb = "extra"

    c_emb = "none"
    char_vocab = None
    if known_chars_file is not None:
        c_emb = "standard"
        char_vocab = data.read_vocab(known_chars_file)

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

    # Read train and test data
    train_ds = data.read_datasets(training_files)
    test_ds = data.read_datasets([test_file])
    dictionaries, extras = data.create_mappers(
        train_ds,
        w_emb=w_emb,
        c_emb=c_emb,
        m_emb=m_emb,
        pretrained_word_embeddings_file=pretrained_word_embeddings_file,
        morphlex_embeddings_file=morphlex_embeddings_file,
        known_chars=char_vocab,
    )
    if DEBUG:
        device = torch.device("cpu")
        threads = 1
        # Set the number of threads to use for CPU
        torch.set_num_threads(threads)
        log.info(f"Using {threads} CPU threads")
        train_ds = train_ds[:batch_size]
        test_ds = test_ds[:batch_size]

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
        "hidden_dim": final_dim,  # The main LSTM time-steps will be mapped to this dim
        "morphlex_extra_dim": morphlex_extra_dim,
        "lstm_dropouts": 0.1,
        "input_dropouts": 0.0,
        "noise": 0.1,  # Noise to main_in, to main_bilstm
        "morphlex_freeze": morphlex_freeze,
    }
    extras = {
        "morph_lex_embeddings": torch.from_numpy(extras["morph_lex_embeddings"])
        .float()
        .to(device)
        if m_emb == "standard" or m_emb == "extra"
        else None,
        "word_embeddings": torch.from_numpy(extras["word_embeddings"])
        .float()
        .to(device)
        if w_emb == "pretrained"
        else None,
    }
    # Write all configuration to disk
    output_dir = pathlib.Path(output_dir)
    with output_dir.joinpath("hyperparamters.json").open(mode="+w") as f:
        json.dump({**parameters, **model_parameters}, f, indent=4)

    from pos.model import ABLTagger

    tagger = ABLTagger(**model_parameters, **extras).to(device)
    log.info(tagger)

    for name, tensor in tagger.state_dict().items():
        log.info(f"{name}: {torch.numel(tensor)}")
    log.info(
        f"Trainable parameters={sum(p.numel() for p in tagger.parameters() if p.requires_grad)}"
    )
    log.info(
        f"Not trainable parameters={sum(p.numel() for p in tagger.parameters() if not p.requires_grad)}"
    )
    if label_smoothing == 0.0:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=data.PAD_ID, reduction="sum")
    else:
        criterion = partial(
            train.smooth_ce_loss, pad_idx=data.PAD_ID, smoothing=label_smoothing
        )

    reduced_lr_names = ["token_embedding.weight"]
    params = [
        {
            "params": list(
                param
                for name, param in filter(
                    lambda kv: kv[0] not in reduced_lr_names, tagger.named_parameters()
                )
            )
        },
        {
            "params": list(
                param
                for name, param in filter(
                    lambda kv: kv[0] in reduced_lr_names, tagger.named_parameters()
                )
            ),
            "lr": parameters["word_embedding_lr"],
        },
    ]
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=parameters["learning_rate"])
        log.info("Using SGD")
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=parameters["learning_rate"])
        log.info("Using Adam")
    else:
        raise ValueError("Unknown optimizer")
    if scheduler == "multiply":
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambda epoch: 0.95
        )
    elif scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
            threshold=100.0,
            threshold_mode="abs",
            cooldown=0,
        )
    else:
        raise ValueError("Unknown scheduler")

    train.run_epochs(
        model=tagger,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=partial(
            data.data_loader,
            dataset=train_ds,
            device=device,
            shuffle=True,
            w_emb=w_emb,
            c_emb=c_emb,
            m_emb=m_emb,
            dictionaries=dictionaries,
            batch_size=batch_size,
        ),
        test_loader=partial(
            data.data_loader,
            dataset=test_ds,
            device=device,
            shuffle=True,
            w_emb=w_emb,
            c_emb=c_emb,
            m_emb=m_emb,
            dictionaries=dictionaries,
            batch_size=batch_size * 10,
        ),
        epochs=epochs,
        writer=tensorboard.SummaryWriter(str(output_dir)),
    )
    test_tags_tagged = train.tag_sents(
        model=tagger,
        data_loader=data.data_loader(
            dataset=test_ds,
            device=device,
            shuffle=False,
            w_emb=w_emb,
            c_emb=c_emb,
            m_emb=m_emb,
            dictionaries=dictionaries,
            batch_size=batch_size * 10,
        ),
        dictionaries=dictionaries,
    )
    log.info("Writing predictions, dictionaries and model")
    data.write_tsv(
        str(output_dir.joinpath("predictions.tsv")),
        (*data.unpack_dataset(test_ds), test_tags_tagged),
    )
    with (output_dir / "known_toks.txt").open("w+") as f:
        for token in data.get_vocab(x for x, y in train_ds):
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
@click.argument("dictionaries_files")
@click.argument("input", type=click.File("r"))
@click.argument("output", type=click.File("w+"))
@click.option(
    "--device", default="cpu", help="The device to use, 'cpu' or 'cuda:0' for GPU."
)
def tag(model_file, dictionaries_files, input, output, device):
    """Tag tokens in a file.

    Args:
        model_file: A filepath to a trained model.
        dictionaries_file: A filepath to dictionaries (vocabulary mappings) for preprocessing.
        input: A file or stdin (-), formatted as: token per line, sentences separated with newlines (empty line).
        output: A file or stdout (-). Output is formatted like the input, but after each token there is a tab and then the tag.
    """
    log.info(f"Using device={device}")
    device = torch.device(device)
    log.info("Reading model")
    model = torch.load(model_file, map_location=device)
    log.info("Reading dictionaries")
    with open(dictionaries_files, "rb") as f:
        dictionaries = pickle.load(f)
    log.info("Reading dataset")
    datasent = data.read_datasent(input)
    log.info("Predicting tags")
    predicted_tags = train.tag_sents(
        model,
        data.data_loader(
            datasent,
            device,
            dictionaries,
            shuffle=False,
            w_emb="pretrained",
            c_emb="standard",
            m_emb="standard",
            batch_size=16,
        ),
        dictionaries=dictionaries,
    )

    log.info("Writing results")
    for tokens, tags in zip(datasent, predicted_tags):
        for token, tag in zip(tokens, tags):
            output.write("\t".join([token, tag]) + "\n")
        output.write("\n")
    log.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    log = logging.getLogger()
    cli()
