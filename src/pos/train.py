"""Train, evaluate and tag using a certain model."""
from typing import Iterable, Optional, Callable, Dict, List
import logging
import datetime
from functools import partial
from math import inf

import torch

from .evaluate import Experiment
from . import data
from .types import Dataset, Vocab, PredictedDataset, PredictedSentence, VocabMap
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
        return torch.nn.CrossEntropyLoss(ignore_index=data.PAD_ID, reduction="sum")
    else:
        return partial(  # type: ignore
            smooth_ce_loss, pad_idx=data.PAD_ID, smoothing=label_smoothing,
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


def run_epochs(
    model,
    optimizer,
    criterion,
    scheduler: torch.optim.lr_scheduler._LRScheduler,  # pylint: disable=protected-access
    data_loader: Callable[
        [Dataset, bool, int], Iterable[Dict[str, Optional[torch.Tensor]]]
    ],
    dictionaries: Dict[str, VocabMap],
    batch_size: int,
    epochs: int,
    train_ds: Dataset,
    test_ds: Dataset,
    output_dir,
):
    """Run all the training epochs using the training data and evaluate on the test data."""
    from torch.utils import tensorboard

    writer = tensorboard.SummaryWriter(str(output_dir))

    best_validation_loss = inf
    # Get the tokens only
    train_vocab = Vocab.from_symbols(train_ds.unpack()[0])
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
            data_loader=data_loader(  # type: ignore
                dataset=train_ds, shuffle=True, batch_size=batch_size,
            ),
            log_prepend=f"Epoch={epoch}/{epochs}, ",
        )
        end = datetime.datetime.now()
        log.info(f"Training took={end-start} seconds")
        # We just run the validation using a larger batch size
        val_loss, accuracies = evaluate_model(
            model=model,
            data_loader=data_loader,
            test_ds=test_ds,
            batch_size=batch_size,
            dictionaries=dictionaries,
            train_vocab=train_vocab,
            criterion=criterion,
        )
        log.info(f"Validation acc={accuracies[0][0]}, loss={val_loss}")
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
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            # torch.save(pos_model.state_dict(), 'model.pt')
        scheduler.step(val_loss)
    # model.load_state_dict(torch.load('model.pt'))


def train_model(
    model,
    optimizer,
    criterion,
    data_loader: Iterable[Dict[str, Optional[torch.Tensor]]],
    log_prepend: str,
) -> float:
    """Run a single training epoch and evaluate the model."""
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(data_loader, start=1):
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
            log.info(log_prepend + f"batch={i}, acc={acc}, loss={loss.item():.4f}")
    return total_loss


def evaluate_model(
    model,
    data_loader: Callable[
        [Dataset, bool, int], Iterable[Dict[str, Optional[torch.Tensor]]]
    ],
    dictionaries: Dict[str, VocabMap],
    criterion,
    test_ds: Dataset,
    train_vocab: Vocab,
    batch_size: int,
):
    """Evaluate a model on a given test set."""
    test_tags, test_loss = model.tag_sents(
        data_loader=data_loader(  # type: ignore
            dataset=test_ds, shuffle=False, batch_size=batch_size * 10
        ),
        dictionaries=dictionaries,
        criterion=criterion,
    )
    predicted_ds = PredictedDataset(
        PredictedSentence((tokens, tags, predicted_tags))
        for tokens, tags, predicted_tags in zip(*test_ds.unpack(), test_tags)
    )
    return (
        test_loss,
        Experiment(
            predicted_ds, train_vocab=train_vocab, dicts=dictionaries
        ).all_accuracy(),
    )


def categorical_accuracy(preds, y):
    """Calculate accuracy (per batch), i.e. if you get 8/10 right, this returns 0.8."""
    max_preds = preds.argmax(
        dim=1, keepdim=True
    )  # get the index of the max probability
    # nonzero to map to idexes again and filter out pads.
    non_pad_elements = (y != data.PAD_ID).nonzero(as_tuple=False)
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return float(correct.sum().item()) / y[non_pad_elements].shape[0]


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
