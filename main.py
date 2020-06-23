#!/usr/bin/env python
import pickle
import random
import logging
import json
import pprint
import pathlib

import click
import torch
import numpy as np

from pos import data
from pos import evaluate
from pos import train

DEBUG = False


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    global DEBUG
    DEBUG = debug


@cli.command()
@click.argument("inputs", nargs=-1)
@click.argument("output", type=click.File("w+"))
@click.option("--coarse", is_flag=True, help='Maps the tags "coarse".')
def gather_tags(inputs, output, coarse):
    input_data = []
    for input in inputs:
        input_data.extend(data.read_tsv(input))
    _, tags = data.tsv_to_pairs(input_data)
    if coarse:
        tags = data.coarsify(tags)
    tags = data.get_vocab(tags)
    for tag in sorted(list(tags)):
        output.write(f"{tag}\n")


@cli.command()
@click.argument("inputs", nargs=-1)
@click.argument("embedding", default=None)
@click.argument("output", type=click.File("w+"))
@click.argument("format", type=str)
def filter_embedding(inputs, embedding, output, format):
    """
    This will filter an 'embedding' file based on the words which occur in the 'inputs' files.
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
@click.argument("input")
@click.option(
    "--report_type",
    type=click.Choice(["accuracy", "accuracy-breakdown", "errors"]),
    help="Type of reporting to do",
)
@click.option("--count", default=10, help="The number of outputs for top-k")
@click.option(
    "--vocab",
    default=None,
    help="The pickle file containing the DataVocabMap of the model.",
)
def report(input, report_type, count, vocab):
    examples = evaluate.analyse_examples(evaluate.flatten_data(data.read_tsv(input)))
    if report_type == "accuracy":
        log.info(evaluate.calculate_accuracy(examples))
    elif report_type == "accuracy-breakdown":
        with open(vocab, "rb") as f:
            data_vocab_map: data.DataVocabMap = pickle.load(f)
        test_vocab = evaluate.get_vocab(examples)
        train_vocab = set(data_vocab_map.w_map.w2i.keys())
        morphlex_vocab = set(data_vocab_map.m_map.w2i.keys())
        both_vocab = train_vocab.union(morphlex_vocab)
        neither_vocab = test_vocab.difference((train_vocab.union(morphlex_vocab)))
        examples_filter_on_train = evaluate.filter_examples(examples, train_vocab)
        examples_filter_on_morphlex = evaluate.filter_examples(examples, morphlex_vocab)
        examples_filter_on_both = evaluate.filter_examples(examples, both_vocab)
        examples_filter_on_neither = evaluate.filter_examples(examples, neither_vocab)
        log.info("T = w ∈ training set, M = w ∈ morphlex set, Test = w ∈ test set")
        log.info(
            f"len(Test)={len(test_vocab)}, len(T)={len(train_vocab)}, len(M)={len(morphlex_vocab)}"
        )
        log.info(f"len(T ∪ M)={len(both_vocab)}")
        log.info(f"len(w ∉ (T ∪ M))={len(neither_vocab)}")
        log.info(f"Total acc (w ∈ Test): {evaluate.calculate_accuracy(examples)}")
        log.info(
            f"Train acc (w ∈ T): {evaluate.calculate_accuracy(examples_filter_on_train)}"
        )
        log.info(
            f"Morph acc (w ∈ M): {evaluate.calculate_accuracy(examples_filter_on_morphlex)}"
        )
        log.info(
            f"Both acc (w ∈ (T ∪ M)): {evaluate.calculate_accuracy(examples_filter_on_both)}"
        )
        log.info(
            f"Unk acc (w ∉ (T ∪ M) = w ∈ Test - (T ∪ M)): {evaluate.calculate_accuracy(examples_filter_on_neither)}"
        )
    elif report_type == "errors":
        errors = evaluate.all_errors(examples)
        log.info(pprint.pformat(errors.most_common(count)))
    else:
        raise ValueError("Unkown report_type")


@cli.command()
@click.argument("training_files", nargs=-1)
@click.argument("test_file")
@click.argument("output_dir", default="./out")
@click.option("--name", default=None, help="The name to give to the run, used in wandb")
@click.option("--gpu/--no_gpu", default=False)
@click.option("--save_model/--no_save_model", default=False)
@click.option("--save_vocab/--no_save_vocab", default=False)
@click.option(
    "--known_chars_file",
    default="./data/extra/characters_training.txt",
    help="A file which contains the characters the model should know. "
    + "File should be a single line, the line is split() to retrieve characters.",
)
@click.option(
    "--morphlex_embeddings_file",
    default="./data/extra/dmii.vectors",
    help="A file which contains the morphological embeddings.",
)
@click.option(
    "--pretrained_word_embeddings",
    is_flag=True,
    default=False,
    help="Should we use pretrained word embeddings? If set, also set --pretrained_word_embeddings_file",
)
@click.option(
    "--pretrained_word_embeddings_file",
    default="./data/extra/igc2018_wordform_sentences_cased_skipgram_dim300_epoch20_wngram2_minn3_maxn6.vec",
    help="A file which contains pretrained word embeddings.",
)
@click.option("--epochs", default=20)
@click.option("--batch_size", default=32)
@click.option("--char_lstm_layers", default=1)
@click.option("--main_lstm_layers", default=1)
@click.option("--learning_rate", default=0.20)
@click.option("--morphlex_freeze", is_flag=True, default=False)
@click.option(
    "--word_embedding_dim", default=128, help="The word/token embedding dimension."
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
    pretrained_word_embeddings,
    pretrained_word_embeddings_file,
    epochs,
    main_lstm_layers,
    char_lstm_layers,
    batch_size,
    save_model,
    save_vocab,
    gpu,
    optimizer,
    learning_rate,
    morphlex_freeze,
    word_embedding_dim,
    name,
    scheduler,
):
    """
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
    # We create a folder for this run specifically
    parameters = {
        "training_files": training_files,
        "test_file": test_file,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    model_parameters = {
        "morph_lex_freeze": morphlex_freeze,
        "lstm_dropouts": 0.1,
        "input_dropouts": 0.0,
        "emb_char_dim": 20,  # The characters are mapped to this dim
        "char_lstm_dim": 64,  # The character LSTM will output with this dim
        "char_lstm_layers": char_lstm_layers,  # The character LSTM will output with this dim
        "emb_token_dim": word_embedding_dim,  # The tokens are mapped to this dim
        "main_lstm_dim": 64,  # The main LSTM dim will output with this dim
        "main_lstm_layers": main_lstm_layers,  # The main LSTM dim will output with this dim
        "hidden_dim": 64,  # The main LSTM time-steps will be mapped to this dim
        "noise": 0.1,  # Noise to main_in, to main_bilstm
    }
    output_dir = pathlib.Path(output_dir)
    with output_dir.joinpath("hyperparamters.json").open(mode="+w") as f:
        json.dump({**parameters, **model_parameters}, f, indent=4)
    # Tracking experiments and visualization
    import wandb

    wandb.init(
        project="pos",
        config={**parameters, **model_parameters},
        dir=str(output_dir),
        name=name,
    )

    # Read train and test data
    train_tokens, train_tags = [], []
    log.info(f"Reading training files={training_files}")
    for train_file in training_files:
        toks, tags, _ = data.read_tsv(train_file)
        train_tokens.extend(toks)
        train_tags.extend(tags)
    test_tokens, test_tags, _ = data.read_tsv(test_file)

    mapper, m_embedding, w_embedding = train.create_mapper(
        train_tokens,
        test_tokens,
        train_tags,
        known_chars_file,
        morphlex_embeddings_file,
        pretrained_word_embeddings_file if pretrained_word_embeddings else None,
    )

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

    if DEBUG:
        device = torch.device("cpu")
        threads = 1
        # Set the number of threads to use for CPU
        torch.set_num_threads(threads)
        log.info(f"Using {threads} CPU threads")
        train_tokens = train_tokens[:batch_size]
        train_tags = train_tags[:batch_size]
        test_tokens = test_tokens[:batch_size]
        test_tags = test_tags[:batch_size]

    from pos.model import ABLTagger

    tagger = ABLTagger(
        mapper=mapper,
        device=device,
        char_dim=len(mapper.c_map),
        token_dim=len(mapper.w_map),
        tags_dim=len(mapper.t_map),
        morph_lex_embeddings=torch.from_numpy(m_embedding).float().to(device),
        word_embeddings=torch.from_numpy(w_embedding).float().to(device)
        if w_embedding is not None
        else None,
        **model_parameters,
    ).to(device)
    log.info(tagger)
    wandb.watch(tagger)
    for name, tensor in tagger.state_dict().items():
        log.info(f"{name}: {torch.numel(tensor)}")
    log.info(
        f"Trainable parameters={sum(p.numel() for p in tagger.parameters() if p.requires_grad)}"
    )
    log.info(
        f"Not trainable parameters={sum(p.numel() for p in tagger.parameters() if not p.requires_grad)}"
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=data.PAD_ID, reduction="none")
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
            "lr": parameters["learning_rate"] / 100,
        },
    ]
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=parameters["learning_rate"])
        log.info(f"Using SGD with lr={parameters['learning_rate']}")
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=parameters["learning_rate"])
        log.info(f"Using Adam with lr={parameters['learning_rate']}")
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
        train=(train_tokens, train_tags),
        test=(test_tokens, test_tags),
        epochs=epochs,
        batch_size=batch_size,
    )
    test_tags_tagged = train.tag_sents(
        model=tagger, sentences=test_tokens, batch_size=batch_size
    )
    data.write_tsv(
        str(output_dir.joinpath("predictions.tsv")),
        (test_tokens, test_tags, test_tags_tagged),
    )
    if save_vocab:
        save_location = output_dir.joinpath("vocab.pickle")
        with save_location.open("wb+") as f:
            pickle.dump(mapper, f)
    if save_model:
        save_location = output_dir.joinpath("tagger.pt")
        torch.save(tagger.state_dict(), str(save_location))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    log = logging.getLogger()
    cli()
