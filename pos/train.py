"""Train, evaluate and tag using a certain model."""
from typing import Iterable, Optional, Callable, Dict
import logging
import datetime

import torch

from . import data

log = logging.getLogger()

WRITTEN_GRAPH = False


def run_epochs(
    model,
    optimizer,
    criterion,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: Callable[[], Iterable[Dict[str, Optional[torch.Tensor]]]],
    test_loader: Callable[[], Iterable[Dict[str, Optional[torch.Tensor]]]],
    epochs: int,
    writer,
):
    """Run all the training epochs using the training data and evaluate on the test data."""
    best_validation_loss = 100
    for epoch in range(1, epochs + 1):
        # Time it
        start = datetime.datetime.now()
        log.info(
            f'Epoch={epoch}/{epochs}, lr={list(param_group["lr"] for param_group in optimizer.param_groups)}'
        )
        train_loss = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            log_prepend=f"Epoch={epoch}/{epochs}, ",
            writer=writer,
        )
        end = datetime.datetime.now()
        log.info(f"Training took={end-start} seconds")
        # We just run the validation using a larger batch size
        val_loss, val_acc = evaluate_model(model, test_loader, criterion)
        log.info(f"Validation acc={val_acc}, loss={val_loss}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            # torch.save(pos_model.state_dict(), 'model.pt')
        scheduler.step(val_loss)
    # model.load_state_dict(torch.load('model.pt'))


def train_model(
    model,
    optimizer,
    criterion,
    data_loader: Callable[[], Iterable[Dict[str, Optional[torch.Tensor]]]],
    log_prepend: str,
    writer,
) -> float:
    """Run a single training epoch and evaluate the model."""
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(data_loader(), start=1):
        global WRITTEN_GRAPH
        if not WRITTEN_GRAPH:
            writer.add_graph(model, batch)
            WRITTEN_GRAPH = True
        optimizer.zero_grad()
        y_pred = model(batch)
        y_pred = y_pred.view(-1, y_pred.shape[-1])
        assert batch["t"] is not None
        y = batch["t"].view(-1)

        loss = criterion(y_pred, y)
        acc = categorical_accuracy(y_pred, y)
        total_loss += loss.item()
        loss.backward()
        # Clip gardients like in DyNet
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if i % 10 == 0:
            log.info(
                log_prepend + f"batch={i}, acc={acc.item()}, loss={loss.item():.4f}"
            )
    return total_loss


def evaluate_model(
    model,
    data_loader: Callable[[], Iterable[Dict[str, Optional[torch.Tensor]]]],
    criterion,
):
    """Evaluate a model on a given test set."""
    model.eval()
    with torch.no_grad():
        y_pred_total = None
        y_total = None
        for batch in data_loader():
            y_pred = model(batch)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            assert batch["t"] is not None
            y = batch["t"].view(-1)
            if y_pred_total is None:
                y_pred_total = y_pred
                y_total = y
            else:
                y_pred_total = torch.cat([y_pred_total, y_pred], dim=0)
                y_total = torch.cat([y_total, y], dim=0)

        loss = criterion(y_pred_total, y_total)
        acc = categorical_accuracy(y_pred_total, y_total)

    return loss.item(), acc.item()


def tag_sents(
    model,
    data_loader: Callable[[], Iterable[Dict[str, Optional[torch.Tensor]]]],
    dictionaries: Dict[str, data.VocabMap],
) -> data.DataSent:
    """Tag (apply POS) on a given data set."""
    model.eval()
    with torch.no_grad():
        start = datetime.datetime.now()
        tags = []
        for i, batch in enumerate(data_loader(), start=1):
            pred = model(batch)
            # (b, seq, tags)
            for b in range(pred.shape[0]):
                # (seq, f)
                sent_pred = pred[b, :, :].view(-1, pred.shape[-1])
                # x = (b, seq, f), the last few elements in f word/token, morph and maybe c_tag
                assert batch["w"] is not None
                num_non_pads = torch.sum((batch["w"][b, :] != data.PAD_ID)).item()
                # We use the fact that padding is placed BEHIND those features
                sent_pred = sent_pred[:num_non_pads, :]  # type: ignore
                idxs = sent_pred.argmax(dim=1).tolist()
                tags.append(tuple(dictionaries["t_map"].i2w[idx] for idx in idxs))
    end = datetime.datetime.now()
    log.info(f"Tagging took={end-start} seconds")
    return data.DataSent(tags)


def categorical_accuracy(preds, y):
    """Calculate accuracy (per batch), i.e. if you get 8/10 right, this returns 0.8."""
    max_preds = preds.argmax(
        dim=1, keepdim=True
    )  # get the index of the max probability
    # nonzero to map to idexes again and filter out pads.
    non_pad_elements = (y != data.PAD_ID).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def smooth_ce_loss(pred, gold, pad_idx, smoothing=0.1):
    """Calculate cross entropy loss, apply label smoothing if needed."""
    gold = gold.contiguous().view(-1)

    n_class = pred.size(1)

    # Expand the idx to one-hot representation
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    # First smooth the one hot and then add the smoothed values
    smoothed = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    log_prb = torch.log_softmax(pred, dim=1)
    # Compute the smoothed loss
    loss = -(smoothed * log_prb).sum(dim=1)

    # Filter out the pad values
    non_pad_mask = gold.ne(pad_idx)
    # Return the pad filtered loss
    loss = loss.masked_select(non_pad_mask).sum()
    return loss
