import pathlib

from pos.core import Dicts
from pos.model import CharacterAsWordEmbedding, CharacterDecoder, Encoder, EncodersDecoders, Modules
from pos.model.embeddings import CharacterEmbedding
from pos.train import get_criterion, get_optimizer, get_scheduler, run_epochs, tag_data_loader


def test_train_tagger(decoders, encoders, data_loader, kwargs, tagger_evaluator, lemma_evaluator):
    abl_tagger = EncodersDecoders(encoders=encoders, decoders=decoders)
    criterion = get_criterion(decoders)
    optimizer = get_optimizer(abl_tagger.parameters(), optimizer=kwargs["optimizer"], lr=kwargs["learning_rate"])
    scheduler = get_scheduler(optimizer, scheduler=kwargs["scheduler"])
    evaluators = {Modules.Tagger: tagger_evaluator, Modules.Lemmatizer: lemma_evaluator}

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
