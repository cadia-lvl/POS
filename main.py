#!/usr/bin/env python
import random
import logging
import datetime
import json
import pprint
import pathlib

import click
import torch
import numpy as np
# import wandb

from pos import data
from pos import evaluate
from pos import train

DEBUG = False


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    global DEBUG
    DEBUG = debug


@cli.command()
@click.argument('inputs', nargs=-1)
@click.argument('output', type=click.File('w+'))
@click.option('--coarse', is_flag=True, help='Maps the tags "coarse".')
def gather_tags(inputs, output, coarse):
    input_data = []
    for input in inputs:
        input_data.extend(data.read_tsv(input))
    _, tags = data.tsv_to_pairs(input_data)
    if coarse:
        tags = data.coarsify(tags)
    tags = data.get_vocab(tags)
    for tag in sorted(list(tags)):
        output.write(f'{tag}\n')


@cli.command()
@click.argument('input')
@click.option('--report_type', type=click.Choice(['accuracy', 'errors']), help='Type of reporting to do')
@click.option('--count', default=10, help='The number of outputs for top-k')
def report(input, report_type, count):
    examples = evaluate.analyse_examples(
        evaluate.flatten_data(data.read_tsv(input)))
    if report_type == 'accuracy':
        log.info(evaluate.calculate_accuracy(examples))
    elif report_type == 'errors':
        errors = evaluate.all_errors(examples)
        log.info(pprint.pformat(errors.most_common(count)))


@cli.command()
@click.argument('training_files', nargs=-1)
@click.argument('test_file')
@click.argument('output_dir', default='./out')
@click.option('--known_chars_file', default='./extra/characters_training.txt',
              help='A file which contains the characters the model should know. '
              + 'File should be a single line, the line is split() to retrieve characters.')
@click.option('--c_tags_file', default='./extra/word_class_vocab.txt')
@click.option('--morphlex_embeddings_file', default='./data/format/dmii.vectors',
              help='A file which contains the morphological embeddings.')
@click.option('--coarse_epochs', default=12)
@click.option('--fine_epochs', default=20)
@click.option('--batch_size', default=32)
def train_and_tag(training_files,
                  test_file,
                  output_dir,
                  known_chars_file,
                  c_tags_file,
                  morphlex_embeddings_file,
                  coarse_epochs,
                  fine_epochs,
                  batch_size):
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
    hyperparameters = {
        'training_files': training_files,
        'test_file': test_file,
        'coarse_epochs': coarse_epochs,
        'fine_epochs': fine_epochs,
        'batch_size': batch_size,
        'lstm_dropout': 0.1,
        'input_dropouts': 0.1,
        'emb_char_dim': 20,  # The characters are mapped to this dim
        'char_lstm_dim': 64,  # The character LSTM will output with this dim
        'emb_token_dim': 128,  # The tokens are mapped to this dim
        'main_lstm_dim': 64,  # The main LSTM dim will output with this dim
        'hidden_dim': 32,  # The main LSTM time-steps will be mapped to this dim
    }
    output_dir = pathlib.Path(output_dir)
    with output_dir.joinpath('hyperparamters.json').open(mode='+w') as f:
        json.dump(hyperparameters, f, indent=4)
    # Read train and test data
    train_tokens, train_tags = [], []
    log.info(f'Reading training files={training_files}')
    for train_file in training_files:
        toks, tags, _ = data.read_tsv(train_file)
        train_tokens.extend(toks)
        train_tags.extend(tags)
    test_tokens, test_tags, _ = data.read_tsv(test_file)
    # Prepare the coarse tags
    train_tags_coarse = data.coarsify(train_tags)
    test_tags_coarse = data.coarsify(test_tags)

    coarse_mapper, fine_mapper, embedding = train.create_mappers(
        train_tokens, test_tokens, train_tags, known_chars_file, c_tags_file, morphlex_embeddings_file)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Torch will use the allocated GPUs from environment variable CUDA_VISIBLE_DEVICES
        # --gres=gpu:titanx:2
        log.info(f'Using {torch.cuda.device_count()} GPUs')
    else:
        device = torch.device('cpu')
        threads = 1
        # Set the number of threads to use for CPU
        torch.set_num_threads(threads)
        log.info(f'Using {threads} CPU threads')

    if DEBUG:
        device = torch.device('cpu')
        threads = 1
        # Set the number of threads to use for CPU
        torch.set_num_threads(threads)
        log.info(f'Using {threads} CPU threads')
        train_tokens = train_tokens[:batch_size]
        train_tags_coarse = train_tags_coarse[:batch_size]
        train_tags = train_tags[:batch_size]
        test_tokens = test_tokens[:batch_size]
        test_tags_coarse = test_tags_coarse[:batch_size]
        test_tags = test_tags[:batch_size]

    log.info('Creating coarse tagger')
    coarse_tagger = train.create_model(coarse_mapper,
                                       hyperparameters,
                                       embedding,
                                       device,
                                       c_tags_embeddings=False)
    log.info(coarse_tagger)

    # We ignore targets which have beed padded
    criterion = torch.nn.CrossEntropyLoss(ignore_index=data.PAD_ID)
    # lr as used in ABLTagger
    optimizer = torch.optim.SGD(coarse_tagger.parameters(), lr=0.13)
    # learning rate decay as in ABLTagger
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: 0.95)
    train.run_epochs(model=coarse_tagger,
                     optimizer=optimizer,
                     criterion=criterion,
                     scheduler=scheduler,
                     train=(train_tokens, train_tags_coarse),
                     test=(test_tokens, test_tags_coarse),
                     epochs=coarse_epochs,
                     batch_size=batch_size)
    start = datetime.datetime.now()
    train_tags_coarse_tagged = train.tag_sents(model=coarse_tagger,
                                               sentences=train_tokens,
                                               batch_size=batch_size)
    end = datetime.datetime.now()
    log.info(f'Tagging took={end-start} seconds')
    test_tags_coarse_tagged = train.tag_sents(model=coarse_tagger,
                                              sentences=test_tokens,
                                              batch_size=batch_size)

    data.write_tsv(str(output_dir.joinpath('coarse_predictions.tsv')),
                   (test_tokens, test_tags_coarse, test_tags_coarse_tagged))
    del coarse_tagger
    torch.cuda.empty_cache()
    log.info('Creating fine tagger')
    fine_tagger = train.create_model(fine_mapper,
                                     hyperparameters,
                                     embedding,
                                     device,
                                     c_tags_embeddings=True)
    log.info(fine_tagger)
    fine_train = [[(tok, tag) for tok, tag in zip(tok_sent, tag_sent)]
                  for tok_sent, tag_sent in zip(train_tokens, train_tags_coarse_tagged)]
    fine_test = [[(tok, tag) for tok, tag in zip(tok_sent, tag_sent)]
                 for tok_sent, tag_sent in zip(test_tokens, test_tags_coarse_tagged)]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=data.PAD_ID)
    optimizer = torch.optim.SGD(fine_tagger.parameters(), lr=0.13)
    # learning rate decay as in ABLTagger
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: 0.95)
    train.run_epochs(model=fine_tagger,
                     optimizer=optimizer,
                     criterion=criterion,
                     scheduler=scheduler,
                     train=(fine_train, train_tags),
                     test=(fine_test, test_tags),
                     epochs=fine_epochs,
                     batch_size=batch_size)
    test_tags_tagged = train.tag_sents(model=fine_tagger,
                                       sentences=fine_test,
                                       batch_size=batch_size)
    data.write_tsv(str(output_dir.joinpath('fine_predictions.tsv')),
                   (test_tokens, test_tags, test_tags_tagged))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    log = logging.getLogger()
    cli()
