"""The tagging Module."""
import logging
from typing import Optional, Dict, Iterable, List, Tuple, Union
import datetime

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import torch
from torch import Tensor, stack
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

from .core import VocabMap, Modules


log = logging.getLogger(__name__)


class ClassingWordEmbedding(nn.Module):
    """Classic word embeddings."""

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx=0):
        """Create one."""
        super(ClassingWordEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        # Skip the first index, should be zero
        nn.init.xavier_uniform_(self.embedding.weight[1:, :])
        self.output_dim = self.embedding.weight.data.shape[1]

    def forward(self, tensor):
        """Apply the module."""
        return self.embedding(tensor)


class PretrainedEmbedding(nn.Module):
    """The Morphological Lexicion embeddings."""

    def __init__(self, embeddings: Tensor, freeze=False, padding_idx=0):
        """Create one."""
        super(PretrainedEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings, freeze=freeze, padding_idx=padding_idx,
        )
        self.output_dim = self.embedding.weight.data.shape[1]

    def forward(self, tensor):
        """Apply the module."""
        return self.embedding(tensor)


class CharacterAsWordEmbedding(nn.Module):
    """A Character as Word Embedding."""

    def __init__(
        self,
        vocab_size,
        character_embedding_dim=20,
        char_lstm_dim=64,
        char_lstm_layers=1,
        padding_idx=0,
    ):
        """Create one."""
        super(CharacterAsWordEmbedding, self).__init__()
        self.character_embedding = nn.Embedding(
            vocab_size, character_embedding_dim, padding_idx=padding_idx
        )
        nn.init.xavier_uniform_(self.character_embedding.weight[1:, :])
        # The character BiLSTM
        self.char_bilstm = nn.LSTM(
            input_size=character_embedding_dim,
            hidden_size=char_lstm_dim,
            num_layers=char_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        for name, param in self.char_bilstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.xavier_uniform_(param)
        self.output_dim = 2 * char_lstm_dim

    def forward(self, tensor):
        """Apply the module."""
        # (b, seq, chars, f)
        char_embs = self.character_embeding(tensor)
        self.char_bilstm.flatten_parameters()
        # One sentence at a time
        words_as_chars = []
        for b in range(char_embs.shape[0]):
            # w = words in sent, c = chars in word, f = char features
            # (w, c, f)
            sent_chars = char_embs[b, :, :, :]
            # some sentences might only contain PADs for some words, which pack_sequence does not like
            # Count the number of non-PAD words
            num_non_zero = torch.sum(
                torch.sum(torch.sum(sent_chars, dim=2), dim=1) != 0.0
            ).item()
            # Drop them
            dropped_pads = sent_chars[: int(num_non_zero), :, :]
            packed, lengths = pack_sequence(dropped_pads)
            sent_chars_rep = self.char_bilstm(packed)[0]
            un_packed = unpack_sequence(sent_chars_rep)
            # Get the last timestep, taking the PADs on char level into account
            sent_chars_rep_last_ts = torch.cat(
                [
                    un_packed[idx, length - 1, :][None, :]
                    for idx, length in enumerate(lengths.tolist())
                ],
                dim=0,
            )
            # Re-add the PAD words we removed before
            added_pads = copy_into_larger_tensor(
                sent_chars_rep_last_ts,
                sent_chars_rep_last_ts.new_zeros(
                    size=(sent_chars.shape[0], sent_chars_rep_last_ts.shape[1])
                ),
            )
            # Collect and add dimension to sum up
            words_as_chars.append(added_pads[None, :, :])
        chars_as_word = self.char_bilstm_out_dropout(torch.cat(words_as_chars, dim=0))
        return chars_as_word


class FlairTransformerEmbedding(nn.Module):
    """A wrapper for the TransformerEmbedding from Flair. It's here to fit into the preprocessing setup."""

    def __init__(self, file_path, **kwargs):
        """Initialize the embeddings."""
        super(FlairTransformerEmbedding, self).__init__()
        self.emb = TransformerWordEmbeddings(
            file_path,
            layers=kwargs.get("transformer_layers", "-1"),
            use_scalar_mix=kwargs.get("transformer_use_scalar_mix", False),
            allow_long_sentences=kwargs.get("transformer_allow_long_sentences", True),
            fine_tune=True,
            batch_size=kwargs.get("batch_size", 1),
        )
        self.output_dim = 256

    def forward(self, sentences):
        """Run through the model. Sentences are a sequence of sequence of string."""
        f_sentences = [Sentence(" ".join(sentence)) for sentence in sentences]
        self.emb.embed(f_sentences)
        return pad_sequence(
            [
                stack([token.embedding for token in sentence])
                for sentence in f_sentences
            ],
            batch_first=True,
        )


class Tagger(nn.Module):
    """A tagger; accept some tensor input and return logits over classes."""

    def __init__(self, input_dim, output_dim):
        """Initialize."""
        super(Tagger, self).__init__()
        self.tagger = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.tagger.weight)
        self.output_dim = output_dim

    def forward(self, t_in):
        """Apply the module."""
        return self.tagger(t_in)


class Encoder(nn.Module):
    """The Pytorch module implementing the encoder."""

    def __init__(
        self,
        main_lstm_dim=64,  # The main LSTM dim will output with this dim
        main_lstm_layers=0,  # The main LSTM layers
        lstm_dropouts=0.0,
        input_dropouts=0.0,
        noise=0.1,
        morphlex_embedding: nn.Module = None,
        pretrained_word_embedding: nn.Module = None,
        word_embedding: nn.Module = None,
        chars_as_word_embedding: nn.Module = None,
        transformer_embedding: nn.Module = None,
    ):
        """Initialize the module given the parameters."""
        super(Encoder, self).__init__()
        self.noise = noise
        self.morphlex_embedding = morphlex_embedding
        self.pretrained_word_embedding = pretrained_word_embedding
        self.word_embedding = word_embedding
        self.chars_as_word_embedding = chars_as_word_embedding
        self.transformer_embedding = transformer_embedding

        self.use_bilstm = not main_lstm_layers == 0
        encoder_out_dim = 0
        if self.morphlex_embedding is not None:
            encoder_out_dim += self.morphlex_embedding.output_dim  # type: ignore
        if self.pretrained_word_embedding is not None:
            encoder_out_dim += self.pretrained_word_embedding.output_dim  # type: ignore
        if self.word_embedding is not None:
            encoder_out_dim += self.word_embedding.output_dim  # type: ignore
        if self.chars_as_word_embedding is not None:
            encoder_out_dim += self.chars_as_word_embedding.output_dim  # type: ignore
        if self.transformer_embedding is not None:
            encoder_out_dim += self.transformer_embedding.output_dim  # type: ignore

        # BiLSTM over all inputs
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=encoder_out_dim,
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
            encoder_out_dim = main_lstm_dim * 2
        self.main_bilstm_out_dropout = nn.Dropout(p=input_dropouts)
        self.output_dim = encoder_out_dim

    def forward(  # pylint: disable=arguments-differ
        self, batch_dict: Dict[Modules, Tensor]
    ):
        """Run a forward pass through the module. Input should be tensors."""
        # input is (batch_size=num_sentence, max_seq_len_in_batch=max(len(sentences)), max_word_len_in_batch + 1 + 1)
        main_in = None
        # Embeddings
        if self.word_embedding is not None:
            embs = self.word_embedding(batch_dict[Modules.WordEmbeddings])
            main_in = cat_or_return(main_in, embs)
        if self.pretrained_word_embedding is not None:
            embs = self.pretrained_word_embedding(batch_dict[Modules.Pretrained])
            main_in = cat_or_return(main_in, embs)
        if self.morphlex_embedding is not None:
            embs = self.morphlex_embedding(batch_dict[Modules.MorphLex])
            main_in = cat_or_return(main_in, embs)
        if self.chars_as_word_embedding is not None:
            embs = self.chars_as_word_embedding(batch_dict[Modules.CharsAsWord])
            main_in = cat_or_return(main_in, embs)
        if self.transformer_embedding is not None:
            embs = self.transformer_embedding(batch_dict[Modules.BERT])
            main_in = cat_or_return(main_in, embs)

        # Add noise - like in dyney
        # if self.training and main_in is not None:
        #     main_in = main_in + torch.empty_like(main_in).normal_(0, self.noise)
        # (b, seq, f)

        if self.use_bilstm:
            # Pack the paddings
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                main_in,
                batch_dict[Modules.Lengths],
                batch_first=True,
                enforce_sorted=False,
            )
            # Make sure that the parameters are contiguous.
            self.bilstm.flatten_parameters()
            # Ignore the hidden outputs
            packed_out, _ = self.bilstm(packed)
            # Unpack and ignore the lengths
            bilstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )
            main_in = self.main_bilstm_out_dropout(bilstm_out)

        return main_in


class ABLTagger(nn.Module):
    """The ABLTagger, consists of an Encoder(multipart) and a Tagger."""

    def __init__(self, encoder: nn.Module, tagger: nn.Module):
        """Initialize the tagger."""
        super(ABLTagger, self).__init__()
        self.encoder = encoder
        self.tagger = tagger

    def forward(self, batch_dict: Dict[Modules, Tensor]):
        """Forward pass."""
        encoded = self.encoder(batch_dict)
        return self.tagger(encoded)


def cat_or_return(t1, t2, dim=2):
    """Concatenate two tensors if the first one is not None, otherwise return second."""
    if t1 is None:
        return t2
    else:
        return torch.cat((t1, t2), dim=dim)


def pack_sequence(padded_sequence):
    """Pack the PAD in a sequence. Assumes that PAD=0.0 and appended."""
    # input:
    # (b, s, f)
    # lengths = (b, s)
    lengths = torch.sum(torch.pow(padded_sequence, 2), dim=2)
    # lengths = (b)
    lengths = torch.sum(
        lengths != torch.Tensor([0.0]).to(padded_sequence.device), dim=1,
    )
    return (
        torch.nn.utils.rnn.pack_padded_sequence(
            padded_sequence, lengths, batch_first=True, enforce_sorted=False
        ),
        lengths,
    )


def unpack_sequence(packed_sequence):
    """Inverse of pack_sequence."""
    return torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=True)[0]


def copy_into_larger_tensor(
    tensor: torch.Tensor, like_tensor: torch.Tensor
) -> torch.Tensor:
    """Create a larger tensor based on given tensor. Only works for 2-dims."""
    base = torch.zeros_like(like_tensor)
    base[: tensor.shape[0], : tensor.shape[1]] = tensor
    return base


def batch_first_to_batch_second(tensor: torch.Tensor):
    """Move the batch size from the first index to second. Only works for 3-D tensor."""
    # Move the dimensions and fix the indices (with contiguous) so they can be used.
    return tensor.permute(1, 0, 2).contiguous()


def batch_second_to_batch_first(tensor: torch.Tensor):
    """Move the batch size from the first index to second. Only works for 3-D tensor."""
    # Move the dimensions and fix the indices (with contiguous) so they can be used.
    return tensor.permute(1, 0, 2).contiguous()
