"""Train a model.

During training this module handles:
- Logging and monitoring.
- Epochs (iterations) and evaluation.
- Schedulers and optimizers. 
"""
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import logging
from datetime import datetime
from functools import partial
from math import inf

import torch
from torch.nn.utils import clip_grad_norm_
from torch import Tensor, no_grad
from torch.utils.data import DataLoader

from .data import (
    PAD_ID,
    BATCH_KEYS,
)
from .model import ABLTagger


log = logging.getLogger(__name__)


def get_parameter_groups(parameters, **kwargs) -> List[Dict]:
    """Return the parameters groups with differing learning rates."""
    reduced_lr_names = ["token_embedding.weight"]
    params = [
        {
            "params": list(
                param
                for name, param in filter(
                    lambda kv: kv[0] not in reduced_lr_names, parameters
                )
            )
        },
        {
            "params": list(
                param
                for name, param in filter(
                    lambda kv: kv[0] in reduced_lr_names, parameters
                )
            ),
            "lr": kwargs["word_embedding_lr"],
        },
    ]
    return params


def get_optimizer(parameters, **kwargs) -> torch.optim.Optimizer:
    """Return the optimizer to use based on options."""
    optimizer = kwargs.get("optimizer", "sgd")
    lr = kwargs["learning_rate"]
    log.info(f"Setting optmizer={optimizer}")
    if optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    elif optimizer == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    else:
        raise ValueError("Unknown optimizer")


def get_criterion(**kwargs):
    """Return the criterion to use based on options."""
    label_smoothing = kwargs.get("label_smoothing", 0.0)
    log.info(f"Label smoothing={label_smoothing}")
    if not label_smoothing:
        return torch.nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    else:
        return partial(  # type: ignore
            smooth_ce_loss,
            pad_idx=PAD_ID,
            smoothing=label_smoothing,
        )


def get_scheduler(torch_optimizer, **kwargs):
    """Return the training scheduler to use based on options."""
    scheduler = kwargs.get("scheduler", "multiply")
    if scheduler == "multiply":
        return torch.optim.lr_scheduler.MultiplicativeLR(
            torch_optimizer, lr_lambda=lambda epoch: 0.95
        )
    elif scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=torch_optimizer,
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


def print_tagger(tagger: torch.nn.Module):
    """Print all information about the model."""
    log.info(tagger)
    for name, tensor in tagger.state_dict().items():
        log.info(f"{name}: {torch.numel(tensor)}")
    log.info(
        f"Trainable parameters={sum(p.numel() for p in tagger.parameters() if p.requires_grad)}"
    )
    log.info(
        f"Not trainable parameters={sum(p.numel() for p in tagger.parameters() if not p.requires_grad)}"
    )


def run_batch(
    model: ABLTagger,
    batch: Dict[BATCH_KEYS, Any],
    criterion=None,
    optimizer=None,
) -> Tuple[Tensor, float]:
    """Run a batch through the model.

    If criterion is given, it will be applied and returned (as float).
    If optimizer is given, it will be used to update parameters in conjunction with the criterion.
    """
    if optimizer is not None:
        optimizer.zero_grad()
    model_out = model(batch)[0]  # TODO: Tagger-only now
    # (b, seq, tag_features)
    loss = 0.0
    if criterion is not None:
        t_loss = criterion(
            model_out.view(-1, model_out.shape[-1]),
            batch[BATCH_KEYS.TARGET_FULL_TAGS].view(-1).to(model_out.device),
        )
        if optimizer is not None:
            t_loss.backward()
            clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        loss = t_loss.item()
    return model_out, loss


def tag_batch(
    model: ABLTagger,
    batch: Dict[BATCH_KEYS, Any],
    criterion=None,
    optimizer=None,
) -> Tuple[Iterable[Sequence[str]], float]:
    """Tag (apply POS) on a given data set."""
    preds, loss = run_batch(model, batch, criterion, optimizer)
    tagger = model.decoders[0]  # TODO: Tagger hard-coded
    return tagger.postprocess(preds, batch[BATCH_KEYS.LENGTHS]), loss


def tag_data_loader(
    model: ABLTagger,
    data_loader: DataLoader,
    criterion=None,
) -> Tuple[List[Sequence[str]], float]:
    """Tag (apply POS) on a given data set. Sets the model to evaluation mode."""
    tags: List[Sequence[str]] = []
    loss = 0.0
    model.eval()
    with no_grad():
        start = datetime.now()
        for batch in data_loader:
            b_tags, b_loss = tag_batch(model, batch, criterion)
            loss += b_loss
            tags.extend(b_tags)
        end = datetime.now()
    log.info(f"Tagged {sum((1 for sent in tags for token in sent))} tokens")
    log.info(f"Tagging took={end-start} seconds")
    return tags, loss


def train_model(
    model,
    optimizer,
    criterion,
    data_loader: DataLoader,
    log_prepend: str,
) -> float:
    """Run a single training epoch and evaluate the model."""
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(data_loader, start=1):
        y_pred, loss = run_batch(model, batch, criterion, optimizer)
        total_loss += loss
        acc = categorical_accuracy(
            y_pred, batch[BATCH_KEYS.TARGET_FULL_TAGS].to(y_pred.device)
        )
        if i % 10 == 0:
            log.info(
                f"{log_prepend}batch={i}/{len(data_loader)}, acc={acc}, loss={loss:.4f}"
            )
    return total_loss


def run_epochs(
    model: ABLTagger,
    optimizer,
    criterion,
    scheduler,
    evaluator,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    epochs: int,
    output_dir,
):
    """Run all the training epochs using the training data and evaluate on the test data."""
    from torch.utils import tensorboard

    writer = tensorboard.SummaryWriter(str(output_dir))

    best_validation_loss = inf
    # Get the tokens only
    for epoch in range(1, epochs + 1):
        # Time it
        start = datetime.now()
        log.info(
            f'Epoch={epoch}/{epochs}, lr={list(param_group["lr"] for param_group in optimizer.param_groups)}'
        )
        train_loss = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_data_loader,
            log_prepend=f"Epoch={epoch}/{epochs}, ",
        )
        end = datetime.now()
        log.info(f"Training took={end-start} seconds")
        # We just run the validation using a larger batch size
        tags, val_loss = tag_data_loader(
            model,
            data_loader=test_data_loader,
            criterion=criterion,
        )
        accuracies = evaluator(tags).all_accuracy()
        log.info(f"Validation acc={accuracies[0][0]}, loss={val_loss}")
        write_scalars(writer, accuracies, train_loss, val_loss, epoch)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            # torch.save(pos_model.state_dict(), 'model.pt')
        if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(val_loss)
        else:
            scheduler.step()
    # model.load_state_dict(torch.load('model.pt'))


def write_scalars(writer, accuracies, train_loss, val_loss, epoch):
    """Print evaluation results to tensorboard."""
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", val_loss, epoch)
    writer.add_scalar("Accuracy/Total", accuracies[0][0], epoch)
    writer.add_scalar("Accuracy/Unknown", accuracies[1][0], epoch)
    writer.add_scalar("Accuracy/Known", accuracies[2][0], epoch)
    writer.add_scalar("Accuracy/Wemb", accuracies[3][0], epoch)
    writer.add_scalar("Accuracy/Wemb+M", accuracies[4][0], epoch)
    writer.add_scalar("Accuracy/M", accuracies[5][0], epoch)
    writer.add_scalar("Accuracy/Seen", accuracies[6][0], epoch)
    writer.add_scalar("Accuracy/Unknown-Wemb", accuracies[7][0], epoch)
    writer.add_scalar("Accuracy/Unknown-Wemb+M", accuracies[8][0], epoch)
    writer.add_scalar("Accuracy/Unknown-M", accuracies[9][0], epoch)
    writer.add_scalar("Accuracy/Unseen", accuracies[10][0], epoch)


def categorical_accuracy(preds, y):
    """Calculate accuracy (per batch), i.e. if you get 8/10 right, this returns 0.8."""
    preds = preds.argmax(dim=2)  # get the index of the max probability
    # Change them to column vectors - conceptually easier
    y = y.reshape(shape=(-1,))
    preds = preds.reshape(shape=(-1,))
    # Remove elements which are paddings
    non_pad_idxs = y != PAD_ID
    preds = preds[non_pad_idxs]
    y = y[non_pad_idxs]
    correct = y == preds
    return correct.sum().item() / len(y)


def smooth_ce_loss(pred, gold, pad_idx, smoothing=0.1):
    """Calculate cross entropy loss, apply label smoothing if needed."""
    gold = gold.contiguous().view(-1)

    n_class = pred.size(1)

    # Expand the idx to one-hot representation
    one_hot = torch.zeros_like(pred, device=pred.device).scatter(1, gold.view(-1, 1), 1)
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
