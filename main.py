#!/usr/bin/env python
import random
import logging
import time

import click
import torch
import numpy as np

import data
from model import ABLTagger


@click.group()
def cli():
    pass


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
@click.argument('training_files',
                nargs=-1
                )
@click.argument('test_file'
                )
@click.argument('output_dir'
                )
@click.option('--known_chars_file',
              default='./extra/characters_training.txt',
              help='A file which contains the characters the model should know. '
              + 'File should be a single line, the line is split() to retrieve characters.'
              )
@click.option('--morphlex_embeddings_file',
              default='./extra/dmii.vectors',
              help='A file which contains the morphological embeddings.'
              )
def train(training_files,
          test_file,
          output_dir,
          known_chars_file,
          morphlex_embeddings_file):
    """
    training_files: Files to use for training (supports multiple files = globbing).
    All training files should be .tsv, with two columns, the token and tag.
    test_file: Same format as training_files. Used to evaluate the model.
    output_dir: The directory to write out model and results.
    """
    # Read train and test data
    training_corpus = []
    log.info(f'Reading training files={training_files}')
    for train_file in training_files:
        training_corpus.extend(data.read_tsv(train_file))
    test_corpus: data.Corpus = data.read_tsv(test_file)

    # Extract tokens, tags, characters
    train_tokens, train_tags = data.tsv_to_pairs(training_corpus)
    test_tokens, test_tags = data.tsv_to_pairs(test_corpus)

    # Prepare the coarse tags
    train_tags_coarse = data.coarsify(train_tags)
    test_tags_coarse = data.coarsify(test_tags)

    # Define the vocabularies and mappings
    chars = data.read_known_characters(known_chars_file)

    train_token_vocab = data.get_vocab(train_tokens)
    test_token_vocab = data.get_vocab(test_tokens)
    train_tag_vocab = data.get_vocab(train_tags)
    test_tag_vocab = data.get_vocab(test_tags)

    # We need EOS and SOS for chars
    char_vocap_map = data.VocabMap(chars, special_tokens=[
        (data.UNK, data.UNK_ID),
        (data.PAD, data.PAD_ID),
        (data.EOS, data.EOS_ID),
        (data.SOS, data.SOS_ID)
    ])
    token_vocab_map = data.VocabMap(train_token_vocab, special_tokens=[
        (data.UNK, data.UNK_ID),
        (data.PAD, data.PAD_ID),
    ])
    # The tags will be padded, but we do not support UNK
    tag_vocab_map = data.VocabMap(train_tag_vocab, special_tokens=[
        (data.PAD, data.PAD_ID),
    ])
    coarse_tag_vocab_map = data.VocabMap(data.get_vocab(train_tags_coarse), special_tokens=[
        (data.PAD, data.PAD_ID),
    ])
    log.info(f'Character vocab={len(char_vocap_map)}')
    log.info(f'Word vocab={len(token_vocab_map)}')
    log.info(f'Tag vocab={len(tag_vocab_map)}')
    log.info(f'Tag (coarse) vocab={len(coarse_tag_vocab_map)}')
    token_freqs = data.get_tok_freq(train_tokens)

    log.info('Token unk analysis')
    data.unk_analysis(train_token_vocab, test_token_vocab)
    log.info('Tag unk analysis')
    data.unk_analysis(train_tag_vocab, test_tag_vocab)
    # We filter the morphlex embeddings based on the training and test set for quicker training. This should not be done in production
    filter_on = data.get_vocab(train_tokens)
    filter_on.update(data.get_vocab(test_tokens))
    # The morphlex embeddings are similar to the tokens, no EOS or SOS needed
    m_vocab_map, embedding = data.read_embedding(morphlex_embeddings_file, filter_on=filter_on, special_tokens=[
        (data.UNK, data.UNK_ID),
        (data.PAD, data.PAD_ID)
    ])

    pos_model = ABLTagger(
        char_dim=len(char_vocap_map),
        token_dim=len(token_vocab_map),
        tags_dim=len(coarse_tag_vocab_map),
        morph_lex_embeddings=torch.from_numpy(embedding).float(),
        c_tags_embeddings=None
    )
    log.info(
        f'Trainable parameters={sum(p.numel() for p in pos_model.parameters() if p.requires_grad)}')
    log.info(
        f'Not trainable parameters={sum(p.numel() for p in pos_model.parameters() if not p.requires_grad)}')
    # Make the model data parallel
    pos_model = torch.nn.DataParallel(pos_model)
    # Move model to device, before optimizer
    pos_model.to(device)
    log.info(pos_model)
    # We ignore targets which have beed padded
    criterion = torch.nn.CrossEntropyLoss(ignore_index=data.PAD_ID)
    # lr as used in ABLTagger
    optimizer = torch.optim.SGD(pos_model.parameters(), lr=0.13)
    # learning rate decay as in ABLTagger
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: 0.95)

    epochs = 1
    batch_size = 32
    best_validation_loss = 100
    for epoch in range(1, epochs + 1):
        # Time it
        start = time.time()
        log.info(
            f'Epoch={epoch}/{epochs}, lr={list(param_group["lr"] for param_group in optimizer.param_groups)}')
        train_iter = data.make_batches((train_tokens, train_tags_coarse),
                                       batch_size=batch_size,
                                       shuffle=True,
                                       c_map=char_vocap_map,
                                       w_map=token_vocab_map,
                                       m_map=m_vocab_map,
                                       t_map=coarse_tag_vocab_map,
                                       device=device)
        train_model(pos_model, train_iter, optimizer, criterion)
        end = time.time()
        log.info(f'Training took={end-start}s')
        # We just run the validation using same batch size, to keep PAD to minimum
        test_iter = data.make_batches((test_tokens, test_tags_coarse),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      c_map=char_vocap_map,
                                      w_map=token_vocab_map,
                                      m_map=m_vocab_map,
                                      t_map=coarse_tag_vocab_map,
                                      device=device)
        val_loss, val_acc = evaluate_model(pos_model, test_iter, criterion)
        log.info(f'Validation loss={val_loss}, acc={val_acc}')
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            # torch.save(pos_model.state_dict(), 'model.pt')
        scheduler.step()
    # model.load_state_dict(torch.load('model.pt'))


def train_model(model, train, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for i, (x, y) in enumerate(train, start=1):
        optimizer.zero_grad()

        y_pred = model(x)

        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y = y.view(-1)

        loss = criterion(y_pred, y)
        acc = categorical_accuracy(y_pred, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if i % 10 == 0:
            log.info(f'batch={i}, acc={acc.item()}, loss={loss.item():.4f}')


def evaluate_model(model, iterator, criterion):
    model.eval()
    with torch.no_grad():
        y_pred_total = None
        y_total = None
        for x, y in iterator:
            y_pred = model(x)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)
            try:
                y_pred_total = torch.cat([y_pred_total, y_pred], dim=0)
                y_total = torch.cat([y_total, y], dim=0)
            except TypeError:
                y_pred_total = y_pred
                y_total = y

        loss = criterion(y_pred_total, y_total).item()
        acc = categorical_accuracy(y_pred_total, y_total).item()

    return loss, acc


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True)  # get the index of the max probability
    # nonzero to map to idexes again
    non_pad_elements = (y != data.PAD_ID).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    log = logging.getLogger()

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

    # Set the seed on all platforms
    SEED = 42
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True  # type: ignore

    cli()
