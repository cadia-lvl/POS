#!/usr/bin/env python
import random
import logging
import time
from typing import Any, List, Tuple

import click
import torch
import numpy as np
# import wandb

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
@click.option('--coarse_epochs', default=12)
@click.option('--fine_epochs', default=20)
@click.option('--batch_size', default=32)
@click.option('--debug', is_flag=True)
def train(training_files,
          test_file,
          output_dir,
          known_chars_file,
          morphlex_embeddings_file,
          coarse_epochs,
          fine_epochs,
          batch_size,
          debug):
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
    # Read the supported characters
    chars = data.read_known_characters(known_chars_file)

    # Extract tokens, tags
    train_tokens, train_tags = data.tsv_to_pairs(training_corpus)
    test_tokens, test_tags = data.tsv_to_pairs(test_corpus)

    # Prepare the coarse tags
    train_tags_coarse = data.coarsify(train_tags)
    test_tags_coarse = data.coarsify(test_tags)

    # Define the vocabularies and mappings
    coarse_mapper = data.DataVocabMap(
        tokens=train_tokens, tags=train_tags_coarse, chars=chars)
    fine_mapper = data.DataVocabMap(
        tokens=train_tokens, tags=train_tags, chars=chars, c_tags=train_tags_coarse)

    # We filter the morphlex embeddings based on the training and test set for quicker training. This should not be done in production
    filter_on = data.get_vocab(train_tokens)
    filter_on.update(data.get_vocab(test_tokens))
    # The morphlex embeddings are similar to the tokens, no EOS or SOS needed
    m_vocab_map, embedding = data.read_embedding(morphlex_embeddings_file, filter_on=filter_on, special_tokens=[
        (data.UNK, data.UNK_ID),
        (data.PAD, data.PAD_ID)
    ])
    coarse_mapper.add_morph_map(m_vocab_map)
    fine_mapper.add_morph_map(m_vocab_map)

    log.info('Creating coarse tagger')
    coarse_tagger = ABLTagger(
        char_dim=len(coarse_mapper.c_map),
        token_dim=len(coarse_mapper.w_map),
        tags_dim=len(coarse_mapper.t_map),
        morph_lex_embeddings=torch.from_numpy(embedding).float(),
        c_tags_embeddings=None
    )
    log.info(
        f'Trainable parameters={sum(p.numel() for p in coarse_tagger.parameters() if p.requires_grad)}')
    log.info(
        f'Not trainable parameters={sum(p.numel() for p in coarse_tagger.parameters() if not p.requires_grad)}')
    # Make the model data parallel
    coarse_tagger = torch.nn.DataParallel(coarse_tagger)
    # Move model to device, before optimizer
    coarse_tagger.to(device)
    log.info(coarse_tagger)
    # We ignore targets which have beed padded
    criterion = torch.nn.CrossEntropyLoss(ignore_index=data.PAD_ID)
    # lr as used in ABLTagger
    optimizer = torch.optim.SGD(coarse_tagger.parameters(), lr=0.13)
    # learning rate decay as in ABLTagger
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: 0.95)
    if debug:
        train_tokens = train_tokens[:batch_size]
        train_tags_coarse = train_tags_coarse[:batch_size]
        train_tags = train_tags[:batch_size]
        test_tokens = test_tokens[:batch_size]
        test_tags_coarse = test_tags_coarse[:batch_size]
        test_tags = test_tags[:batch_size]

    run_epochs(model=coarse_tagger,
               mapper=coarse_mapper,
               optimizer=optimizer,
               criterion=criterion,
               scheduler=scheduler,
               device=device,
               train=(train_tokens, train_tags_coarse),
               test=(test_tokens, test_tags_coarse),
               epochs=coarse_epochs,
               batch_size=batch_size)
    train_tags_coarse_tagged = tag_sents(sentences=train_tokens,
                                         model=coarse_tagger,
                                         batch_size=batch_size,
                                         device=device,
                                         mapper=coarse_mapper)
    test_tags_coarse_tagged = tag_sents(sentences=test_tokens,
                                        model=coarse_tagger,
                                        batch_size=batch_size,
                                        device=device,
                                        mapper=coarse_mapper)
    log.info('Creating fine tagger')
    fine_tagger = ABLTagger(
        char_dim=len(fine_mapper.c_map),
        token_dim=len(fine_mapper.w_map),
        tags_dim=len(fine_mapper.t_map),
        morph_lex_embeddings=torch.from_numpy(embedding).float(),
        c_tags_embeddings=torch.cat([torch.zeros(1, len(fine_mapper.c_t_map)),
                                     torch.diag(torch.ones(len(fine_mapper.c_t_map)))])
    )
    log.info(
        f'Trainable parameters={sum(p.numel() for p in fine_tagger.parameters() if p.requires_grad)}')
    log.info(
        f'Not trainable parameters={sum(p.numel() for p in fine_tagger.parameters() if not p.requires_grad)}')
    # Make the model data parallel
    fine_tagger = torch.nn.DataParallel(fine_tagger)
    # Move model to device, before optimizer
    fine_tagger.to(device)
    log.info(fine_tagger)
    fine_train = [[(tok, tag) for tok, tag in zip(tok_sent, tag_sent)]
                  for tok_sent, tag_sent in zip(train_tokens, train_tags_coarse_tagged)]
    fine_test = [[(tok, tag) for tok, tag in zip(tok_sent, tag_sent)]
                 for tok_sent, tag_sent in zip(test_tokens, test_tags_coarse_tagged)]
    optimizer = torch.optim.SGD(fine_tagger.parameters(), lr=0.13)
    # learning rate decay as in ABLTagger
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: 0.95)
    run_epochs(model=fine_tagger,
               mapper=fine_mapper,
               optimizer=optimizer,
               criterion=criterion,
               scheduler=scheduler,
               device=device,
               train=(fine_train, train_tags),
               test=(fine_test, test_tags),
               epochs=fine_epochs,
               batch_size=batch_size)


def tag_sents(sentences: data.In,
              model: torch.nn.Module,
              device,
              batch_size: int,
              mapper: data.DataVocabMap) -> List[data.SentTags]:
    model.eval()
    start = time.time()
    log.info(f'Tagging sentences len={len(sentences)}')
    iter = mapper.in_x_batches(x=sentences,
                               # Batch size to 1 to avoid dealing with PAD
                               batch_size=batch_size,
                               device=device)
    tags = []
    for i, x in enumerate(iter, start=1):
        pred = model(x)
        # (b, seq, tags)
        for b in range(pred.shape[0]):
            # (seq, f)
            sent_pred = pred[b, :, :].view(-1, pred.shape[-1])
            # x = (b, seq, f), the last few elements in f word/token, morph and maybe c_tag
            num_non_pads = torch.sum((x[b, :, -1] != data.PAD_ID)).item()
            # We use the fact that padding is placed BEHIND those features
            sent_pred = sent_pred[:num_non_pads, :]  # type: ignore
            idxs = sent_pred.argmax(dim=1).tolist()
            tags.append(tuple(mapper.t_map.i2w[idx] for idx in idxs))

    end = time.time()
    log.info(f'Tagging took={end-start} seconds')
    return tags


def run_epochs(model: torch.nn.Module,
               mapper: data.DataVocabMap,
               optimizer,
               criterion,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               device: torch.device,
               train: Tuple[List[Any], List[Any]],
               test: Tuple[List[Any], List[Any]],
               epochs: int,
               batch_size: int):
    best_validation_loss = batch_size
    for epoch in range(1, epochs + 1):
        # Time it
        start = time.time()
        log.info(
            f'Epoch={epoch}/{epochs}, lr={list(param_group["lr"] for param_group in optimizer.param_groups)}')
        train_iter = mapper.in_x_y_batches(x=train[0],
                                           y=train[1],
                                           batch_size=batch_size,
                                           shuffle=True,
                                           device=device)
        train_model(model, train_iter, optimizer, criterion)
        end = time.time()
        log.info(f'Training took={end-start} seconds')
        # We just run the validation using same batch size, to keep PAD to minimum
        test_iter = mapper.in_x_y_batches(x=test[0],
                                          y=test[1],
                                          batch_size=batch_size,
                                          shuffle=False,
                                          device=device)
        val_loss, val_acc = evaluate_model(model, test_iter, criterion)
        log.info(f'Validation acc={val_acc}, loss={val_loss}')
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
