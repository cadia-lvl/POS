import pathlib

from torch.utils.data import dataloader
from pos.core import Dicts
from pos.evaluate import Experiment
from pos.model import ABLTagger, CharacterAsWordEmbedding, Encoder, GRUDecoder, Modules
from pos.train import (
    get_criterion,
    get_optimizer,
    get_parameter_groups,
    get_scheduler,
    run_epochs,
    tag_data_loader,
)


def test_train_tagger(
    decoders, encoder, data_loader, kwargs, tagger_evaluator, lemma_evaluator
):
    abl_tagger = ABLTagger(encoder=encoder, decoders=decoders)
    criterion = get_criterion(decoders)
    parameter_groups = get_parameter_groups(abl_tagger.named_parameters(), **kwargs)
    optimizer = get_optimizer(parameter_groups, **kwargs)
    scheduler = get_scheduler(optimizer, **kwargs)
    # TODO: Add evaluator for Lemmas
    evaluators = {Modules.Tagger: tagger_evaluator, Modules.Lemmatizer: lemma_evaluator}

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


def test_character_lemmatizer(data_loader, kwargs, lemma_evaluator, vocab_maps):
    dicts = vocab_maps
    embs = {}
    embs[Modules.CharactersToTokens] = CharacterAsWordEmbedding(dicts[Dicts.Chars])
    decoders = {}
    encoder = Encoder(embeddings=embs, **kwargs)
    decoders[Modules.Lemmatizer] = GRUDecoder(
        vocab_map=dicts[Dicts.Chars],
        hidden_dim=encoder.output_dim,
        emb_dim=64,
        char_attention=True,
    )
    abl_tagger = ABLTagger(encoder=encoder, decoders=decoders)
    criterion = get_criterion(decoders)
    parameter_groups = get_parameter_groups(abl_tagger.named_parameters(), **kwargs)
    optimizer = get_optimizer(parameter_groups, **kwargs)
    scheduler = get_scheduler(optimizer, **kwargs)
    # TODO: Add evaluator for Lemmas
    evaluators = {Modules.Lemmatizer: lemma_evaluator}

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
        epochs=2,
        output_dir=output_dir,
    )
    _, values = tag_data_loader(
        model=abl_tagger,
        data_loader=data_loader,
    )