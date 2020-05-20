from typing import Tuple, List, Any
import logging
import datetime

import torch

from . import data
from .model import ABLTagger

log = logging.getLogger()


def create_mappers(train_tokens, test_tokens, train_tags, known_chars_file, c_tags_file, morphlex_embeddings_file):

    # Read the supported characters
    chars = data.read_vocab(known_chars_file)
    c_tags = data.read_vocab(c_tags_file)

    # Define the vocabularies and mappings
    coarse_mapper = data.DataVocabMap(
        tokens=train_tokens, tags=c_tags, chars=chars, unk_to_tags=False)
    fine_mapper = data.DataVocabMap(
        tokens=train_tokens, tags=data.get_vocab(train_tags), chars=chars, c_tags=c_tags, unk_to_tags=True)

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
    return coarse_mapper, fine_mapper, embedding


def create_model(mapper, hyperparameters, embedding, device, c_tags_embeddings=False):
    tagger = ABLTagger(
        mapper=mapper,
        device=device,
        char_dim=len(mapper.c_map),
        token_dim=len(mapper.w_map),
        tags_dim=len(mapper.t_map),
        morph_lex_embeddings=torch.from_numpy(embedding).float().to(device),
        c_tags_embeddings=torch.diag(torch.ones(len(mapper.c_t_map))).to(
            device) if c_tags_embeddings else None,
        lstm_dropouts=hyperparameters['lstm_dropout'],
        input_dropouts=hyperparameters['input_dropouts'],
        # The characters are mapped to this dim
        emb_char_dim=hyperparameters['emb_char_dim'],
        # The character LSTM will output with this dim
        char_lstm_dim=hyperparameters['char_lstm_dim'],
        # The tokens are mapped to this dim
        emb_token_dim=hyperparameters['emb_token_dim'],
        # The main LSTM dim will output with this dim
        main_lstm_dim=hyperparameters['main_lstm_dim'],
        # The main LSTM time-steps will be mapped to this dim
        hidden_dim=hyperparameters['hidden_dim'],
    )
    log.info(
        f'Trainable parameters={sum(p.numel() for p in tagger.parameters() if p.requires_grad)}')
    log.info(
        f'Not trainable parameters={sum(p.numel() for p in tagger.parameters() if not p.requires_grad)}')
    # if 'cuda' in str(device):
    # # Make the model data parallel
    # tagger = torch.nn.DataParallel(tagger)
    # Move model to device, before optimizer
    tagger.to(device)
    return tagger


def run_epochs(model,
               optimizer,
               criterion,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               train: Tuple[List[Any], List[Any]],
               test: Tuple[List[Any], List[Any]],
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
        # We just run the validation using same batch size, to keep PAD to minimum
        val_loss, val_acc = evaluate_model(model,
                                           batch_size,
                                           test,
                                           criterion)
        log.info(f'Validation acc={val_acc}, loss={val_loss}')
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            # torch.save(pos_model.state_dict(), 'model.pt')
        scheduler.step()
    # model.load_state_dict(torch.load('model.pt'))


def train_model(model,
                batch_size: int,
                optimizer,
                criterion,
                train: Tuple[List[Any], List[Any]],
                log_prepend: str):
    train_iter = model.mapper.in_x_y_batches(x=train[0],
                                             y=train[1],
                                             batch_size=batch_size,
                                             shuffle=True,
                                             device=model.device)
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    for i, (x, y) in enumerate(train_iter, start=1):
        optimizer.zero_grad()

        try:
            y_pred = model(x)
        except IndexError:
            for b in range(x.shape[0]):
                print(x[b, :, :])
                print(y[b, :])
            raise

        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y = y.view(-1)

        loss = criterion(y_pred, y)
        acc = categorical_accuracy(y_pred, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if i % 10 == 0:
            log.info(log_prepend
                     + f'batch={i}, acc={acc.item()}, loss={loss.item():.4f}')


def evaluate_model(model,
                   batch_size,
                   test: Tuple[List[Any], List[Any]],
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

        loss = criterion(y_pred_total, y_total).item()
        acc = categorical_accuracy(y_pred_total, y_total).item()

    return loss, acc


def tag_sents(model,
              sentences: data.In,
              batch_size: int) -> List[data.SentTags]:
    model.eval()
    with torch.no_grad():
        log.info(f'Tagging sentences len={len(sentences)}')
        iter = model.mapper.in_x_batches(x=sentences,
                                         # Batch size to 1 to avoid dealing with PAD
                                         batch_size=batch_size,
                                         device=model.device)
        tags = []
        for i, x in enumerate(iter, start=1):
            pred = model(x)
            # (b, seq, tags)
            for b in range(pred.shape[0]):
                # (seq, f)
                sent_pred = pred[b, :, :].view(-1, pred.shape[-1])
                # x = (b, seq, f), the last few elements in f word/token, morph and maybe c_tag
                num_non_pads = torch.sum(
                    (x[b, :, -1] != data.PAD_ID)).item()
                # We use the fact that padding is placed BEHIND those features
                sent_pred = sent_pred[:num_non_pads, :]  # type: ignore
                idxs = sent_pred.argmax(dim=1).tolist()
                tags.append(
                    tuple(model.mapper.t_map.i2w[idx] for idx in idxs))
    return tags


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
