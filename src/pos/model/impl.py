"""The implementation of the ABLTagger."""
from typing import Any, Dict, Optional, Sequence, cast

import torch
from torch import nn

from pos.core import Sentence
from pos.data import BATCH_KEYS
from pos.model.abltagger import Embedding, Modules, get_emb_by_initial_token_masks
from pos.model.decoders import CharacterDecoder, Tagger
from pos.model.embeddings import CharacterAsWordEmbedding, ClassingWordEmbedding


class Lemmatizer(nn.Module):
    def __init__(
        self,
        tag_embedding: ClassingWordEmbedding,
        char_as_words: CharacterAsWordEmbedding,
        character_decoder: CharacterDecoder,
    ):
        super().__init__()
        self.tag_embedding = tag_embedding
        self.char_as_words = char_as_words
        self.character_decoder = character_decoder

    def forward(self, batch: Dict[BATCH_KEYS, Any]) -> Dict[Modules, torch.Tensor]:
        """Run a forward pass."""
        tag_embs = self.tag_embedding(batch[BATCH_KEYS.FULL_TAGS], batch[BATCH_KEYS.LENGTHS])
        chars = self.char_as_words(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        preds = self.character_decoder({Modules.CharactersToTokens: chars}, batch, tag_embs)
        return {Modules.Lemmatizer: preds}


class Encoder(nn.Module):
    """The Pytorch module implementing the encoder."""

    def __init__(
        self,
        embeddings: Dict[Modules, Embedding],
        main_lstm_dim=64,  # The main LSTM dim will output with this dim
        main_lstm_layers=1,  # The main LSTM layers
        lstm_dropouts=0.0,
        input_dropouts=0.0,
        residual=True,
    ):
        """Initialize the module given the parameters."""
        super().__init__()
        self.residual = residual
        self.embeddings = nn.ModuleDict({key.value: emb for key, emb in embeddings.items()})

        bilstm_in_dim: int = sum(emb.output_dim for emb in self.embeddings.values())  # type: ignore
        if self.residual:
            self.linear = nn.Linear(bilstm_in_dim, main_lstm_dim)
            bilstm_in_dim = main_lstm_dim

        # BiLSTM over all inputs
        self.bilstm = nn.LSTM(
            input_size=bilstm_in_dim,
            hidden_size=main_lstm_dim,
            num_layers=main_lstm_layers,
            dropout=lstm_dropouts,
            batch_first=True,
            bidirectional=True,
        )
        for name, param in self.bilstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError("Unknown parameter in lstm={name}")
        self.main_bilstm_out_dropout = nn.Dropout(p=input_dropouts)
        self.output_dim = main_lstm_dim * 2

    def forward(self, batch: Sequence[Sentence], lengths: Sequence[int]) -> Dict[Modules, torch.Tensor]:
        """Run a forward pass through the module. Input should be tensors."""
        # input is (batch_size=num_sentence, max_seq_len_in_batch=max(len(sentences)), max_word_len_in_batch + 1 + 1)
        # Embeddings
        embedded = {Modules(key): emb(batch, lengths) for key, emb in self.embeddings.items()}

        to_bilstm = {Modules(key): embedded[Modules(key)] for key in self.embeddings.keys()}
        if Modules.CharactersToTokens in to_bilstm:
            last_hidden = to_bilstm[Modules.CharactersToTokens][1]
            # Reshape from (b*s, f) -> (b, s, f)
            to_bilstm[Modules.CharactersToTokens] = last_hidden.reshape(len(lengths), -1, last_hidden.shape[-1])
        if Modules.BERT in to_bilstm:
            to_bilstm[Modules.BERT] = get_emb_by_initial_token_masks(to_bilstm)
        embs_to_bilstm = torch.cat(list(to_bilstm.values()), dim=2)
        if self.residual:
            embs_to_bilstm = self.linear(embs_to_bilstm)

        # Pack the paddings
        packed = nn.utils.rnn.pack_padded_sequence(
            embs_to_bilstm,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        # Make sure that the parameters are contiguous.
        self.bilstm.flatten_parameters()
        # Ignore the hidden outputs
        packed_out, _ = self.bilstm(packed)
        # Unpack and ignore the lengths
        bilstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        bilstm_out = self.main_bilstm_out_dropout(bilstm_out)
        # Use residual connections
        if self.residual:
            bilstm_out = bilstm_out + torch.cat((embs_to_bilstm, embs_to_bilstm), dim=2)
        embedded[Modules.BiLSTM] = bilstm_out

        return embedded


class ABLTagger(nn.Module):
    """The ABLTagger, consists of an Encoder(multipart) and a Tagger."""

    def __init__(self, encoder: Encoder, tagger: Tagger, character_decoder: Optional[CharacterDecoder] = None):
        """Initialize the model. The Lemmatizer depends on the Tagger."""
        super().__init__()
        self.encoder = encoder
        self.tagger = tagger
        self.character_decoder = character_decoder
        if self.tagger is None and self.character_decoder is not None:
            raise ValueError("Tagger is None, but Lemmatizer is not None. We need the Tagger!")

    def forward(self, batch: Dict[BATCH_KEYS, Any]) -> Dict[Modules, torch.Tensor]:
        """Forward pass."""
        encoded: Dict[Modules, torch.Tensor] = self.encoder(batch[BATCH_KEYS.TOKENS], batch[BATCH_KEYS.LENGTHS])
        decoded = {Modules.Tagger: self.tagger(encoded=encoded, batch=batch)}
        if self.character_decoder is not None:
            decoded[Modules.Lemmatizer] = self.character_decoder(
                tags=decoded[Modules.Tagger], encoded=encoded, batch=batch
            )
        return decoded
