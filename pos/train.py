from typing import Tuple, List
import logging
import datetime

import torch
import wandb

from . import data

log = logging.getLogger()


def create_mapper(train_tokens, test_tokens, train_tags, known_chars_file, morphlex_embeddings_file):

    # Read the supported characters
    chars = data.read_vocab(known_chars_file)

    # Define the vocabularies and mappings
    mapper = data.DataVocabMap(
        tokens=train_tokens, tags=data.get_vocab(train_tags), chars=chars)

    # We filter the morphlex embeddings based on the training and test set for quicker training. This should not be done in production
    filter_on = data.get_vocab(train_tokens)
    filter_on.update(data.get_vocab(test_tokens))
    # The morphlex embeddings are similar to the tokens, no EOS or SOS needed
    m_vocab_map, embedding = data.read_embedding(morphlex_embeddings_file, filter_on=filter_on, special_tokens=[
        (data.UNK, data.UNK_ID),
        (data.PAD, data.PAD_ID)
    ])
    mapper.add_morph_map(m_vocab_map)
    return mapper, embedding


def run_epochs(model,
               optimizer,
               criterion,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               train: Tuple[List[data.DataSent], List[data.DataSent]],
               test: Tuple[List[data.DataSent], List[data.DataSent]],
               epochs: int,
               batch_size: int):
    best_validation_loss = 100
    for epoch in range(1, epochs + 1):
        # Time it
        start = datetime.datetime.now()
        train_model(model=model,
                    batch_size=batch_size,
                    optimizer=optimizer,
                    criterion=criterion,
                    train=train,
                    log_prepend=f'Epoch={epoch}/{epochs}, ')
        log.info(
            f'Epoch={epoch}/{epochs}, lr={list(param_group["lr"] for param_group in optimizer.param_groups)}')
        end = datetime.datetime.now()
        log.info(f'Training took={end-start} seconds')
        # We just run the validation using a larger batch size
        val_loss, val_acc = evaluate_model(model,
                                           batch_size * 100,
                                           test,
                                           criterion)
        log.info(f'Validation acc={val_acc}, loss={val_loss}')
        wandb.log({'loss': val_loss, 'acc': val_acc})
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            # torch.save(pos_model.state_dict(), 'model.pt')
        scheduler.step()
    # model.load_state_dict(torch.load('model.pt'))


def train_model(model,
                batch_size: int,
                optimizer,
                criterion,
                train: Tuple[List[data.DataSent], List[data.DataSent]],
                log_prepend: str):
    train_iter = model.mapper.in_x_y_batches(x=train[0],
                                             y=train[1],
                                             batch_size=batch_size,
                                             shuffle=True,
                                             device=model.device)
    model.train()

    for i, (x, y) in enumerate(train_iter, start=1):
        optimizer.zero_grad()

        y_pred = model(x)

        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y = y.view(-1)

        loss = criterion(y_pred, y)
        # Filter out the pads
        loss = loss[y != data.PAD_ID]
        loss = loss.sum()
        acc = categorical_accuracy(y_pred, y)
        loss.backward()
        # Clip gardients like in DyNet
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if i % 10 == 0:
            log.info(log_prepend
                     + f'batch={i}, acc={acc.item()}, loss={loss.item():.4f}')


def evaluate_model(model,
                   batch_size,
                   test: Tuple[List[data.DataSent], List[data.DataSent]],
                   criterion):
    test_iter = model.mapper.in_x_y_batches(x=test[0],
                                            y=test[1],
                                            batch_size=batch_size,
                                            shuffle=False,
                                            device=model.device)
    model.eval()
    with torch.no_grad():
        y_pred_total = None
        y_total = None
        for x, y in test_iter:
            y_pred = model(x)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)
            if y_pred_total is None:
                y_pred_total = y_pred
                y_total = y
            else:
                y_pred_total = torch.cat([y_pred_total, y_pred], dim=0)
                y_total = torch.cat([y_total, y], dim=0)

        loss = criterion(y_pred_total, y_total)
        # Filter out the pads
        loss = loss[y_total != data.PAD_ID]
        loss = loss.sum()
        acc = categorical_accuracy(y_pred_total, y_total)

    return loss.item(), acc.item()


def tag_sents(model,
              sentences: data.DataSent,
              batch_size: int) -> data.DataSent:
    model.eval()
    with torch.no_grad():
        log.info(f'Tagging sentences len={len(sentences)}')
        start = datetime.datetime.now()
        iter = model.mapper.in_x_batches(x=sentences,
                                         batch_size=batch_size * 100,
                                         device=model.device)
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
                tags.append(
                    tuple(model.mapper.t_map.i2w[idx] for idx in idxs))
    end = datetime.datetime.now()
    log.info(f'Tagging took={end-start} seconds')
    return tags


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True)  # get the index of the max probability
    # nonzero to map to idexes again and filter out pads.
    non_pad_elements = (y != data.PAD_ID).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])
