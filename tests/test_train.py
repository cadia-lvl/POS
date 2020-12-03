import pathlib

from torch.utils.data import dataloader
from pos.evaluate import Experiment
from pos.model import Modules
from pos.train import (
    get_criterion,
    get_optimizer,
    get_parameter_groups,
    get_scheduler,
    run_epochs,
    tag_data_loader,
)


def test_train_tagger(abl_tagger, data_loader, kwargs, ds_lemma, vocab_maps):
    criterion = get_criterion(**kwargs)
    parameter_groups = get_parameter_groups(abl_tagger.named_parameters(), **kwargs)
    optimizer = get_optimizer(parameter_groups, **kwargs)
    scheduler = get_scheduler(optimizer, **kwargs)
    # TODO: Add evaluator for Lemmas
    evaluators = {
        Modules.Tagger: Experiment.all_accuracy_closure(
            test_ds=ds_lemma, dicts=vocab_maps
        )
    }

    # Write all configuration to disk
    output_dir = pathlib.Path(kwargs["output_dir"])

    # Start the training
    run_epochs(
        model=abl_tagger,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        evaluators=evaluators,
        train_data_loader=data_loader,
        test_data_loader=data_loader,
        epochs=kwargs["epochs"],
        output_dir=output_dir,
    )
    _, values = tag_data_loader(
        model=abl_tagger,
        data_loader=data_loader,
    )