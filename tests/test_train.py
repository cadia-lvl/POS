import pathlib

from pos.core import Dicts
from pos.model import ABLTagger, CharacterAsWordEmbedding, CharacterDecoder, Encoder, Modules
from pos.model.embeddings import CharacterEmbedding
from pos.train import get_criterion, get_optimizer, get_parameter_groups, get_scheduler, run_epochs, tag_data_loader


def test_train_tagger(decoders, encoder, data_loader, kwargs, tagger_evaluator, lemma_evaluator):
    abl_tagger = ABLTagger(encoder=encoder, **{key.value: value for key, value in decoders.items()})
    criterion = get_criterion(decoders)
    parameter_groups = get_parameter_groups(abl_tagger)
    optimizer = get_optimizer(parameter_groups, optimizer=kwargs["optimizer"], lr=kwargs["learning_rate"])
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


def test_character_lemmatizer(data_loader, kwargs, lemma_evaluator, vocab_maps, tagger_module):
    dicts = vocab_maps
    embs = {}
    character_embedding = CharacterEmbedding(
        vocab_map=vocab_maps[Dicts.Chars],
        embedding_dim=20,
    )
    embs[Modules.CharactersToTokens] = CharacterAsWordEmbedding(
        character_embedding=character_embedding,
        char_lstm_layers=kwargs["char_lstm_layers"],
        char_lstm_dim=10,
    )
    encoder = Encoder(
        embeddings=embs,
        main_lstm_dim=kwargs["main_lstm_dim"],
        main_lstm_layers=kwargs["main_lstm_layers"],
        lstm_dropouts=0.0,
        input_dropouts=0.0,
    )

    lemmatizer = CharacterDecoder(
        vocab_map=dicts[Dicts.Chars],
        character_embedding=character_embedding,
        hidden_dim=kwargs["lemmatizer_hidden_dim"],
        context_dim=tagger_module.output_dim,
        num_layers=2,
        attention_dim=embs[Modules.CharactersToTokens].output_dim,
        char_attention=True,
    )
    abl_tagger = ABLTagger(encoder=encoder, tagger=tagger_module, character_decoder=lemmatizer)
    criterion = get_criterion({Modules.Tagger: tagger_module, Modules.Lemmatizer: lemmatizer})
    parameter_groups = get_parameter_groups(abl_tagger)
    optimizer = get_optimizer(parameter_groups, optimizer=kwargs["optimizer"], lr=kwargs["learning_rate"])
    scheduler = get_scheduler(optimizer, scheduler=kwargs["scheduler"])
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
