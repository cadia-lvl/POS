import pathlib
from re import M

from pos.core import Dicts
from pos.model import (
    ABLTagger,
    CharacterAsWordEmbedding,
    Encoder,
    CharacterDecoder,
    Modules,
)
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
    parameter_groups = get_parameter_groups(abl_tagger)
    optimizer = get_optimizer(
        parameter_groups, optimizer=kwargs["optimizer"], lr=kwargs["learning_rate"]
    )
    scheduler = get_scheduler(optimizer, scheduler=kwargs["scheduler"])
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
    embs[Modules.CharactersToTokens] = CharacterAsWordEmbedding(
        dicts[Dicts.Chars],
        character_embedding_dim=kwargs["char_emb_dim"],
        char_lstm_layers=kwargs["char_lstm_layers"],
        char_lstm_dim=10,
    )
    decoders = {}
    encoder = Encoder(
        embeddings=embs,
        main_lstm_dim=kwargs["main_lstm_dim"],
        main_lstm_layers=kwargs["main_lstm_layers"],
        lstm_dropouts=0.0,
        input_dropouts=0.0,
    )
    decoders[Modules.Lemmatizer] = CharacterDecoder(
        vocab_map=dicts[Dicts.Chars],
        hidden_dim=encoder.output_dim,
        context_dim=encoder.output_dim,
        char_emb_dim=64,
        num_layers=2,
        attention_dim=embs[Modules.CharactersToTokens].output_dim,
        char_attention=True,
    )
    abl_tagger = ABLTagger(encoder=encoder, decoders=decoders)
    criterion = get_criterion(decoders)
    parameter_groups = get_parameter_groups(abl_tagger)
    optimizer = get_optimizer(
        parameter_groups, optimizer=kwargs["optimizer"], lr=kwargs["learning_rate"]
    )
    scheduler = get_scheduler(optimizer, scheduler=kwargs["scheduler"])
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