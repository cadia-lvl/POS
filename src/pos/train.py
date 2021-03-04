"""Train a model.

During training this module handles:
- Logging and monitoring.
- Epochs (iterations) and evaluation.
- Schedulers and optimizers. 
"""
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging
from datetime import datetime
from functools import partial
from math import inf

from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer, SGD, Adam, SparseAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch import Tensor, no_grad, numel, cat, zeros_like, log_softmax, stack
from torch.utils.data import DataLoader

from .data import (
    PAD_ID,
    BATCH_KEYS,
)
from .model import ABLTagger, Decoder, Modules
from .core import Fields, Sentences


log = logging.getLogger(__name__)

MODULE_TO_BATCHKEY = {
    Modules.Lemmatizer: BATCH_KEYS.TARGET_LEMMAS,
    Modules.Tagger: BATCH_KEYS.TARGET_FULL_TAGS,
}
MODULE_TO_FIELD = {
    Modules.Lemmatizer: Fields.Lemmas,
    Modules.Tagger: Fields.Tags,
}


def get_parameter_groups(model: Module) -> List[Dict]:
    """Return the parameters groups with differing learning rates."""
    sparse_name = "sparse_embedding"
    params = [
        {
            "params": list(
                param
                for name, param in filter(
                    lambda kv: sparse_name not in kv[0],
                    model.named_parameters(),
                )
            ),
        },
        {
            "params": list(
                param
                for name, param in filter(
                    lambda kv: sparse_name in kv[0],
                    model.named_parameters(),
                )
            ),
            "sparse": True,
        },
    ]
    return params


class CombinedAdamOptimzer(Optimizer):
    """An optimzer which combines Adam and SparseAdam based on layer name (sparse_embedding)."""

    def __init__(self, params, lr) -> None:
        """Initialize."""
        super().__init__(params, {"lr": lr})
        self.optimizers: List[Optimizer] = []
        for group in params:
            if "sparse" in group:
                self.optimizers.append(SparseAdam([group], lr=lr))
            else:
                self.optimizers.append(Adam([group], lr=lr))

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Take a step."""
        result = None
        for optimzer in self.optimizers:
            res = optimzer.step(closure=closure)
            if res is not None:
                if result is None:
                    result = res
                else:
                    result += res
        return result


def get_optimizer(parameters, optimizer, lr):
    """Return the optimizer to use based on options."""
    log.info(f"Setting optmizer={optimizer}")
    if optimizer == "sgd":
        return SGD(parameters, lr=lr)
    elif optimizer == "adam":
        return CombinedAdamOptimzer(parameters, lr=lr)
    else:
        raise ValueError("Unknown optimizer")


def _cross_entropy(label_smoothing):
    log.info(f"Label smoothing={label_smoothing}")
    if not label_smoothing:
        return CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    return smooth_ce_loss


def get_criterion(
    decoders: Dict[Modules, Decoder], label_smoothing=0.0
) -> Callable[[Modules, Tensor, Dict[BATCH_KEYS, Any]], Tensor]:
    """Return the criterion to use based on options."""
    loss = _cross_entropy(label_smoothing)

    def weight_loss(key: Modules, pred: Tensor, batch):
        """Combine the losses with their corresponding weights."""
        return (
            loss(
                pred.reshape((-1, pred.shape[-1])),
                batch[MODULE_TO_BATCHKEY[key]].to(pred.device).view((-1)),
            )
            * decoders[key].weight
        )

    return weight_loss


def get_scheduler(torch_optimizer, scheduler):
    """Return the training scheduler to use based on options."""
    if scheduler == "multiply":
        return LambdaLR(torch_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    elif scheduler == "none":
        return LambdaLR(torch_optimizer, lr_lambda=lambda epoch: 1)
    elif scheduler == "plateau":
        return ReduceLROnPlateau(
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


def print_tagger(tagger: Module):
    """Print all information about the model."""
    log.info(tagger)
    for name, tensor in tagger.state_dict().items():
        log.info(f"{name}: {numel(tensor)}")
    log.info(f"Trainable parameters={sum(p.numel() for p in tagger.parameters() if p.requires_grad)}")
    log.info(f"Not trainable parameters={sum(p.numel() for p in tagger.parameters() if not p.requires_grad)}")


def run_batch(
    model: Module,
    batch: Dict[BATCH_KEYS, Any],
    criterion: Callable[[Modules, Tensor, Dict[BATCH_KEYS, Any]], Tensor] = None,
    optimizer=None,
) -> Tuple[Dict[Modules, Tensor], Dict[Modules, float]]:
    """Run a batch through the model.

    If criterion is given, it will be applied and returned (as float).
    If optimizer is given, it will be used to update parameters in conjunction with the criterion.
    """
    if optimizer is not None:
        optimizer.zero_grad()

    preds: Dict[Modules, Tensor] = model(batch)
    # (b, seq, features)
    losses = {
        # (b*s, c, f)
        # The target is (b*s, c)
        key: criterion(key, pred, batch) if criterion else Tensor([0.0])
        for key, pred in preds.items()
    }
    if optimizer:
        stack([loss for loss in losses.values()]).sum().backward()
        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return preds, {key: float(loss.item()) for key, loss in losses.items()}


def tag_batch(
    model: Module,
    batch: Dict[BATCH_KEYS, Any],
    criterion=None,
    optimizer=None,
) -> Tuple[Dict[Modules, float], Dict[Modules, Sentences]]:
    """Tag (apply POS) on a given data set."""
    preds, losses = run_batch(model, batch, criterion, optimizer)
    preds = {
        key: model.__getattr__(key.value).postprocess(preds, batch[BATCH_KEYS.LENGTHS])  # type: ignore
        for key, preds in preds.items()
    }
    return losses, preds  # type: ignore


def tag_data_loader(
    model: Module,
    data_loader: DataLoader,
    criterion=None,
) -> Tuple[Dict[Modules, float], Dict[Modules, Sentences]]:
    """Tag (apply POS) on a given data set. Sets the model to evaluation mode."""
    total_values = defaultdict(tuple)
    total_losses = defaultdict(float)
    total_tokens = 0
    model.eval()
    with no_grad():
        start = datetime.now()
        for batch in data_loader:
            losses, preds = tag_batch(model, batch, criterion)
            for module_name, values, loss in zip(preds.keys(), preds.values(), losses.values()):
                total_losses[module_name] += loss
                total_values[module_name] = total_values[module_name] + values
            total_tokens += sum(batch[BATCH_KEYS.LENGTHS])
        end = datetime.now()
    log.info(f"Processed {total_tokens} tokens in {end-start} seconds")
    return dict(total_losses), dict(total_values)


def train_model(
    model,
    optimizer,
    criterion,
    data_loader: DataLoader,
    log_prepend: str,
) -> Dict[Modules, float]:
    """Run a single training epoch and evaluate the model."""
    model.train()

    epoch_losses = defaultdict(float)
    for i, batch in enumerate(data_loader, start=1):
        preds, losses = run_batch(model, batch, criterion, optimizer)
        for module_name, loss in losses.items():
            epoch_losses[module_name] += loss
        if i % 10 == 0:
            for module_name, preds, loss in zip(preds.keys(), preds.values(), losses.values()):
                acc = categorical_accuracy(preds, batch[MODULE_TO_BATCHKEY[module_name]].to(preds.device))
                log.info(
                    f"{log_prepend}batch={i}/{len(data_loader)}, module_name={module_name} acc={acc}, loss={loss:.4f}"
                )
    return dict(epoch_losses)


def run_epochs(
    model: Module,
    optimizer,
    criterion,
    scheduler,
    evaluators: Dict[Modules, Callable[[Sentences], Tuple[Dict[str, float], Dict[str, int]]]],
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
        log.info(f'Epoch={epoch}/{epochs}, lr={list(param_group["lr"] for param_group in optimizer.param_groups)}')
        train_losses = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_data_loader,
            log_prepend=f"Epoch={epoch}/{epochs}, ",
        )
        write_losses(writer, "Train", train_losses, epoch)
        end = datetime.now()
        log.info(f"Training took={end-start} seconds")
        # We just run the validation using a larger batch size
        val_losses, val_preds = tag_data_loader(
            model,
            data_loader=test_data_loader,
            criterion=criterion,
        )
        write_losses(writer, "Val", val_losses, epoch)
        for module_name, evaluator in evaluators.items():
            accuracies, _ = evaluator(val_preds[module_name])
            write_accuracies(writer, module_name, accuracies, epoch)
        if type(scheduler) == ReduceLROnPlateau:
            scheduler.step(sum((loss for loss in val_losses.values())))
        else:
            scheduler.step()
    # model.load_state_dict(torch.load('model.pt'))


def write_accuracies(writer, module_name, accuracies: Dict[str, float], epoch):
    """Write accuracies to Tensorboard and log."""
    for accuracy_name, accuracy in accuracies.items():
        writer.add_scalar(f"Accuracy/{module_name}/{accuracy_name}", accuracy, epoch)
        log.info(f"Epoch: {epoch}, Accuracy/{module_name}/{accuracy_name}: {accuracy}")


def write_losses(writer, train_val, losses, epoch):
    """Write losses to Tensorboard and log."""
    for module_name, loss in losses.items():
        writer.add_scalar(f"Loss/{train_val}/{module_name}", loss, epoch)
        log.info(f"Epoch: {epoch}, Loss/{train_val}/{module_name}: {loss}")


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


def smooth_ce_loss(pred: Tensor, gold: Tensor, pad_idx=0, smoothing=0.1):
    """Calculate cross entropy loss, apply label smoothing if needed."""
    pred = pred.view(-1, pred.shape[-1])
    gold = gold.contiguous().view(-1)

    n_class = pred.size(1)

    # Expand the idx to one-hot representation
    one_hot = zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    # First smooth the one hot and then add the smoothed values
    smoothed = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    log_prb = log_softmax(pred, dim=1)
    # Compute the smoothed loss
    loss = -(smoothed * log_prb).sum(dim=1)

    # Filter out the pad values
    non_pad_mask = gold.ne(pad_idx)
    # Return the pad filtered loss
    loss = loss.masked_select(non_pad_mask).sum()
    return loss
