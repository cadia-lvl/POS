"""Train a model.

During training this module handles:
- Logging and monitoring.
- Epochs (iterations) and evaluation.
- Schedulers and optimizers.
"""
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Tuple

from torch import Tensor, log_softmax, no_grad, numel, stack, zeros_like
from torch.nn import CrossEntropyLoss, Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from pos.model.interface import EncodersDecoders

from .core import Fields, Sentences
from .data import BATCH_KEYS, PAD_ID
from .model import Decoder, Modules

log = logging.getLogger(__name__)

MODULE_TO_TARGET = {
    Modules.Lemmatizer: BATCH_KEYS.LEMMA_CHAR_IDS,
    Modules.Tagger: BATCH_KEYS.FULL_TAGS_IDS,
}
MODULE_TO_FIELD = {
    Modules.Lemmatizer: Fields.Lemmas,
    Modules.Tagger: Fields.Tags,
}


def get_optimizer(parameters, optimizer, lr):
    """Return the optimizer to use based on options."""
    log.info(f"Setting optmizer={optimizer}")
    if optimizer == "sgd":
        return SGD(parameters, lr=lr)
    elif optimizer == "adam":
        return Adam(parameters, lr=lr)
    else:
        raise ValueError("Unknown optimizer")


def _cross_entropy(label_smoothing):
    log.info(f"Label smoothing={label_smoothing}")
    if not label_smoothing:
        return CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    return smooth_ce_loss


def get_criterion(decoders: Dict[str, Decoder], label_smoothing=0.0) -> Callable[[str, Tensor, Dict[str, Any]], Tensor]:
    """Return the criterion to use based on options."""
    loss = _cross_entropy(label_smoothing)

    def weight_loss(key: str, pred: Tensor, batch: Dict[str, Any]):
        """Combine the losses with their corresponding weights."""
        return (
            loss(
                pred.reshape((-1, pred.shape[-1])),
                batch[MODULE_TO_TARGET[key]].to(pred.device).view((-1)),
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


def print_model(tagger: Module):
    """Print all information about the model."""
    log.info(tagger)
    for name, tensor in tagger.state_dict().items():
        log.info(f"{name}: {numel(tensor)}")
    log.info(f"Trainable parameters={sum(p.numel() for p in tagger.parameters() if p.requires_grad)}")
    log.info(f"Not trainable parameters={sum(p.numel() for p in tagger.parameters() if not p.requires_grad)}")


def run_batch(
    model: EncodersDecoders,
    batch: Dict[str, Any],
    criterion: Callable[[str, Tensor, Dict[str, Any]], Tensor] = None,
    optimizer=None,
) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
    """Run a batch through the model.

    If criterion is given, it will be applied and returned (as float).
    If optimizer is given, it will be used to update parameters in conjunction with the criterion.
    """
    if optimizer is not None:
        optimizer.zero_grad()

    preds: Dict[str, Any] = model(batch)
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
    model: EncodersDecoders,
    batch: Dict[str, Any],
    criterion=None,
    optimizer=None,
) -> Tuple[Dict[str, float], Dict[str, Sentences]]:
    """Tag (apply POS) on a given data set."""
    preds, losses = run_batch(model, batch, criterion, optimizer)
    preds = {key: model.decoders[key].postprocess(batch) for key in preds.keys()}  # type: ignore
    return losses, preds  # type: ignore


def tag_data_loader(
    model: EncodersDecoders,
    data_loader: DataLoader,
    criterion=None,
) -> Tuple[Dict[str, float], Dict[str, Sentences]]:
    """Tag (apply POS) on a given data set. Sets the model to evaluation mode."""
    total_values: Dict[str, Sentences] = defaultdict(tuple)
    total_losses: Dict[str, float] = defaultdict(float)
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
    log.info(f"Tokens/sec.: {total_tokens/max(((end-start).seconds),1)}")
    return dict(total_losses), dict(total_values)


def train_model(
    model: EncodersDecoders,
    optimizer,
    criterion,
    data_loader: DataLoader,
    log_prepend: str,
) -> Dict[str, float]:
    """Run a single training epoch and evaluate the model."""
    model.train()

    epoch_losses: Dict[str, float] = defaultdict(float)
    for i, batch in enumerate(data_loader, start=1):
        preds, losses = run_batch(model, batch, criterion, optimizer)
        for module_name, loss in losses.items():
            epoch_losses[module_name] += loss
        if i % 10 == 0:
            for module_name, pred, loss in zip(preds.keys(), preds.values(), losses.values()):
                acc = categorical_accuracy(pred, batch[module_name].to(pred.device))
                log.info(
                    f"{log_prepend}batch={i}/{len(data_loader)}, module_name={module_name} acc={acc}, loss={loss:.4f}"
                )
    return dict(epoch_losses)


def run_epochs(
    model: EncodersDecoders,
    optimizer,
    criterion,
    scheduler,
    evaluators: Dict[str, Callable[[Sentences], Tuple[Dict[str, float], Dict[str, int]]]],
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    epochs: int,
    output_dir,
):
    """Run all the training epochs using the training data and evaluate on the test data."""
    from torch.utils import tensorboard

    writer = tensorboard.SummaryWriter(str(output_dir))

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
