from typing import Dict

from pos import core
from pos.constants import PAD, Modules
from pos.core import Dicts
from pos.model.decoders import CharacterDecoder, Tagger
from pos.model.embeddings import (
    CharacterAsWordEmbedding,
    CharacterEmbedding,
    ClassicWordEmbedding,
    TransformerEmbedding,
)
from pos.model.interface import Decoder, Encoder, EncodersDecoders
from transformers.models.auto import AutoModel


def build_model(kwargs, dicts) -> EncodersDecoders:
    embs: Dict[str, Encoder] = {}
    if kwargs["bert_encoder"]:
        path = kwargs["bert_encoder"]
        load_state_dict = True
        if "trained" in kwargs:
            path = kwargs["trained"]
            load_state_dict = False
        emb = TransformerEmbedding(Modules.BERT, path=path, dropout=kwargs["emb_dropouts"])
        if load_state_dict:
            emb.model = AutoModel.from_pretrained(kwargs["bert_encoder"], config=emb.config)  # type: ignore
        embs[emb.key] = emb

    # if kwargs["morphlex_embeddings_file"]:
    #     embs[Modules.MorphLex] = PretrainedEmbedding(
    #         vocab_map=dicts[Dicts.MorphLex],
    #         embeddings=embeddings[Dicts.MorphLex],
    #         freeze=kwargs["morphlex_freeze"],
    #         dropout=kwargs["emb_dropouts"],
    #     )
    # if kwargs["pretrained_word_embeddings_file"]:
    #     embs[Modules.Pretrained] = PretrainedEmbedding(
    #         vocab_map=dicts[Dicts.Pretrained],
    #         embeddings=embeddings[Dicts.Pretrained],
    #         freeze=True,
    #         dropout=kwargs["emb_dropouts"],
    #     )
    if kwargs["word_embedding_dim"]:
        embs[Modules.Trained] = ClassicWordEmbedding(
            Modules.Trained,
            dicts[Dicts.Tokens],
            kwargs["word_embedding_dim"],
            dropout=kwargs["emb_dropouts"],
        )
    decoders: Dict[str, Decoder] = {}
    if kwargs["tagger"]:
        decoders[Modules.Tagger] = Tagger(
            key=Modules.Tagger,
            vocab_map=dicts[Dicts.FullTag],
            # TODO: Fix so that we define the Encoder the Tagger accepts
            encoder=embs[Modules.BERT],
            weight=kwargs["tagger_weight"],
        )
    if kwargs["lemmatizer"]:
        character_embedding = CharacterEmbedding(
            Modules.Characters,
            dicts[Dicts.Chars],
            embedding_dim=kwargs["char_emb_dim"],
            dropout=kwargs["emb_dropouts"],
        )
        embs[Modules.Characters] = character_embedding
        char_as_word = CharacterAsWordEmbedding(
            Modules.CharactersToTokens,
            character_embedding=character_embedding,
            char_lstm_layers=kwargs["char_lstm_layers"],
            char_lstm_dim=kwargs["char_lstm_dim"],
            dropout=kwargs["emb_dropouts"],
        )
        embs[Modules.CharactersToTokens] = char_as_word
        tag_embedding = ClassicWordEmbedding(
            key=Modules.TagEmbedding,
            vocab_map=dicts[Dicts.FullTag],
            embedding_dim=kwargs["tag_embedding_dim"],
            padding_idx=dicts[Dicts.FullTag].w2i[PAD],
            dropout=kwargs["emb_dropouts"],
        )
        embs[Modules.TagEmbedding] = tag_embedding
        char_decoder = CharacterDecoder(
            key=Modules.Lemmatizer,
            tag_encoder=tag_embedding,
            characters_to_tokens_encoder=char_as_word,
            characters_encoder=character_embedding,
            vocab_map=dicts[Dicts.Chars],
            hidden_dim=kwargs["lemmatizer_hidden_dim"],
            char_rnn_input_dim=0 if not kwargs["lemmatizer_accept_char_rnn_last"] else char_as_word.output_dim,
            attention_dim=char_as_word.output_dim,
            char_attention=kwargs["lemmatizer_char_attention"],
            num_layers=kwargs["lemmatizer_num_layers"],
            dropout=kwargs["emb_dropouts"],
        )
        decoders[Modules.Lemmatizer] = char_decoder
    model = EncodersDecoders(encoders=embs, decoders=decoders).to(core.device)
    return model
