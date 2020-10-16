"""The tagging Module."""
import logging
from typing import Optional, Dict, Iterable, List, Tuple, Union
import datetime

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import torch
import torch.nn as nn

from .types import Symbols, SimpleDataset, VocabMap
from .data import (
    emb_pairs_to_dict,
    map_embedding,
    wemb_str_to_emb_pair,
    UNK,
    UNK_ID,
    PAD,
    PAD_ID,
)


log = logging.getLogger(__name__)


class PretrainedWordEmbeddings(nn.Embedding):
    """A wrapper for constructing pretrained word embeddings."""

    def __init__(self, file_path):
        """Initialize a pretrained word embedding."""
        with open(file_path) as f:
            it = iter(f)
            # pop the number of vectors and dimension
            next(it)
            embedding_dict = emb_pairs_to_dict(it, wemb_str_to_emb_pair)
        w_map, w_embedding = map_embedding(
            embedding_dict=embedding_dict,
            filter_on=None,
            special_tokens=[(UNK, UNK_ID), (PAD, PAD_ID)],
        )
        self.map = w_map
        embedding = torch.from_numpy(w_embedding).float()
        super().from_pretrained(
            embedding, padding_idx=PAD_ID,
        )


def load_transformer_embeddings(file_path, **kwargs) -> TransformerWordEmbeddings:
    """Return a torch.nn.Module which can accept a list of tokens as input."""
    return TransformerWordEmbeddings(
        file_path,
        layers=kwargs.get("transformer_layers", "-1"),
        use_scalar_mix=kwargs.get("transformer_use_scalar_mix", False),
        allow_long_sentences=kwargs.get("transformer_allow_long_sentences", True),
        fine_tune=True,
        batch_size=kwargs.get("batch_size", 1),
    )


class ABLTagger(nn.Module):
    """The Pytorch module implementing the tagger."""

    def __init__(
        self,
        w_emb: str,  # The type of word embedding to use
        c_emb: str,  # The type of character embeddings to use
        m_emb: str,  # The type of morphlex embeddings to use
        char_dim: int,  # The number of characters in dictionary
        token_dim: int,  # The number of tokens in dictionary
        tags_dim: int,  # The number of tags in dictionary - to predict
        morphlex_extra_dim: int,  # The dimension to map morphlex embeddings to. Only used if m_emb == "extra"
        emb_char_dim: int,  # The characters are mapped to this dim
        char_lstm_dim: int,  # The character LSTM will output with this dim
        char_lstm_layers: int,  # The character LSTM will output with this dim
        emb_token_dim: int,  # The tokens are mapped to this dim, ignored if pretrained
        main_lstm_dim: int,  # The main LSTM dim will output with this dim
        main_lstm_layers: int,  # The main LSTM layers
        final_layer: str,  # The layer type to use after main LSTM.
        final_layer_attention_heads: int,  # The number of attention heads to use in the attention layer.
        final_dim: int,  # The main LSTM time-steps will be mapped to this dim
        lstm_dropouts: float,
        input_dropouts: float,
        noise: float,
        morphlex_freeze: bool,
        morphlex_embeddings: torch.Tensor = None,
        pretrained_word_embeddings: nn.Embedding = None,
        transformer_embedding=None,
    ):
        """Initialize the module given the parameters."""
        super(ABLTagger, self).__init__()
        self.noise = noise
        self.m_emb = m_emb
        self.c_emb = c_emb
        self.w_emb = w_emb
        self.final_layer = final_layer
        self.use_bilstm = not main_lstm_layers == 0
        # Morphlex embeddings
        input_dim_tagger = 0
        if (
            m_emb == "standard" or m_emb == "extra"
        ) and morphlex_embeddings is not None:
            self.morphlex_embedding = nn.Embedding.from_pretrained(
                morphlex_embeddings, freeze=morphlex_freeze, padding_idx=PAD_ID,
            )
            if m_emb == "extra":
                self.morphlex_extra_layer = nn.Linear(
                    self.morphlex_embedding.weight.data.shape[1], morphlex_extra_dim
                )
                input_dim_tagger += morphlex_extra_dim
            else:
                input_dim_tagger += self.morphlex_embedding.weight.data.shape[1]

        # Word embeddings
        if w_emb == "pretrained" and pretrained_word_embeddings is not None:
            self.w_embs_dropout = nn.Dropout(p=input_dropouts)
            self.pretrained_word_embeddings = pretrained_word_embeddings
            input_dim_tagger += self.pretrained_word_embeddings.weight.data.shape[1]
        elif w_emb == "standard":
            self.token_embedding = nn.Embedding(
                token_dim, emb_token_dim, padding_idx=PAD_ID
            )
            nn.init.xavier_uniform_(self.token_embedding.weight[1:, :])
            self.w_embs_dropout = nn.Dropout(p=input_dropouts)
            input_dim_tagger += self.token_embedding.weight.data.shape[1]
        elif w_emb == "electra":
            # The electra embeddings are 256 dimensions
            self.transformer_embedding = transformer_embedding
            input_dim_tagger += 256

        # Character embeddings
        if c_emb == "standard":
            self.character_embedding = nn.Embedding(
                char_dim, emb_char_dim, padding_idx=PAD_ID
            )
            nn.init.xavier_uniform_(self.character_embedding.weight[1:, :])
            # The character BiLSTM
            self.char_bilstm = nn.LSTM(
                input_size=emb_char_dim,
                hidden_size=char_lstm_dim,
                num_layers=char_lstm_layers,
                dropout=lstm_dropouts,
                batch_first=True,
                bidirectional=True,
            )
            for name, param in self.char_bilstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                elif "weight" in name:
                    nn.init.xavier_uniform_(param)
                else:
                    raise ValueError("Unknown parameter in lstm={name}")
            self.c_embs_dropout = nn.Dropout(p=input_dropouts)
            self.char_bilstm_out_dropout = nn.Dropout(p=input_dropouts)
            input_dim_tagger += 2 * char_lstm_dim

        # BiLSTM over all inputs
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=input_dim_tagger,
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
            input_dim_tagger = main_lstm_dim * 2
        self.main_bilstm_out_dropout = nn.Dropout(p=input_dropouts)

        # Extra mapping after BiLSTM
        if final_layer == "dense":
            self.linear = nn.Linear(input_dim_tagger, final_dim)
            nn.init.xavier_uniform_(self.linear.weight)
            input_dim_tagger = final_dim
        elif final_layer == "none":
            # Nothing to do.
            pass
        else:
            raise ValueError(f"Unkown final_layer type={final_layer}")

        # Tagger
        self.tagger = nn.Linear(input_dim_tagger, tags_dim)
        nn.init.xavier_uniform_(self.tagger.weight)

    def forward(  # pylint: disable=arguments-differ
        self, batch_dict: Dict[str, Optional[torch.Tensor]]
    ):
        """Run a forward pass through the module. Input should be tensors."""
        # input is (batch_size=num_sentence, max_seq_len_in_batch=max(len(sentences)), max_word_len_in_batch + 1 + 1)
        # (b, seq, chars)
        chars = batch_dict["c"]
        # (b, seq)
        w = batch_dict["w"]
        # (b, seq)
        m = batch_dict["m"]

        main_in = None
        # Word embeddings
        if self.w_emb == "standard":
            assert w is not None
            w_embs = self.w_embs_dropout(self.token_embedding(w))
            main_in = w_embs
        elif self.w_emb == "pretrained":
            assert w is not None
            w_embs = self.w_embs_dropout(self.pretrained_word_embeddings(w))
            main_in = w_embs
        elif self.w_emb == "electra":
            assert w is not None
            w_embs = w
            main_in = w_embs

        # Morphlex embeddings
        if self.m_emb == "standard" or self.m_emb == "extra":
            assert m is not None
            # (b, seq, f)
            m_embs = self.morphlex_embedding(m)
            if self.m_emb == "extra":
                m_embs = torch.tanh(self.morphlex_extra_layer(m_embs))
            if main_in is not None:
                main_in = torch.cat((main_in, m_embs), dim=2)
            else:
                main_in = m_embs

        # Character embeddings
        if self.c_emb == "standard":
            assert chars is not None
            # (b, seq, chars, f)
            char_embs = self.c_embs_dropout(self.character_embedding(chars))
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
                packed, lengths = self.pack_sequence(dropped_pads)
                sent_chars_rep = self.char_bilstm(packed)[0]
                un_packed = self.unpack_sequence(sent_chars_rep)
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
            chars_as_word = self.char_bilstm_out_dropout(
                torch.cat(words_as_chars, dim=0)
            )
            if main_in is not None:
                main_in = torch.cat((main_in, chars_as_word), dim=2)
            else:
                main_in = chars_as_word

        # Add noise - like in dyney
        if self.training and main_in is not None:
            main_in = main_in + torch.empty_like(main_in).normal_(0, self.noise)
        # (b, seq, f)

        if self.use_bilstm:
            # Pack the paddings
            packed = torch.nn.utils.rnn.pack_padded_sequence(  # type: ignore
                main_in, batch_dict["lens"], batch_first=True, enforce_sorted=False,
            )
            # Make sure that the parameters are contiguous.
            self.bilstm.flatten_parameters()
            # Ignore the hidden outputs
            packed_out, _ = self.bilstm(packed)
            # Unpack and ignore the lengths
            main_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True  # type: ignore
            )
            main_out = self.main_bilstm_out_dropout(main_out)
        else:
            main_out = main_in

        # We apply the final transformation
        if self.final_layer == "none":
            out = main_out
        elif self.final_layer == "dense":
            # We map each word to our targets
            out = torch.tanh(self.linear(main_out))
        else:
            raise ValueError(
                f"Unimplemented final transformation type={self.final_layer}"
            )
        return self.tagger(out)

    def pack_sequence(self, padded_sequence):
        """Pack the PAD in a sequence. Assumes that PAD=0.0 and appended."""
        # input:
        # (b, s, f)
        # lengths = (b, s)
        lengths = torch.sum(torch.pow(padded_sequence, 2), dim=2)
        # lengths = (b)
        lengths = torch.sum(
            lengths
            != torch.tensor([0.0]).to(  # pylint: disable=not-callable
                padded_sequence.device
            ),
            dim=1,
        )
        try:
            return (
                torch.nn.utils.rnn.pack_padded_sequence(
                    padded_sequence, lengths, batch_first=True, enforce_sorted=False
                ),
                lengths,
            )
        except RuntimeError:
            log.debug(f"Lengths={lengths}")
            raise

    def unpack_sequence(self, packed_sequence):
        """Inverse of pack_sequence."""
        return torch.nn.utils.rnn.pad_packed_sequence(
            packed_sequence, batch_first=True
        )[0]

    def tag_sents(
        self,
        data_loader: Iterable[Dict[str, Optional[torch.Tensor]]],
        dictionaries: Dict[str, VocabMap],
        criterion,
    ) -> Union[SimpleDataset, Tuple[SimpleDataset, float]]:
        """Tag (apply POS) on a given data set."""
        start = datetime.datetime.now()
        self.eval()
        tags: List[Symbols] = []
        loss = None
        with torch.no_grad():
            for batch in data_loader:
                softmax_out = self(batch)
                # (b, seq, tag_features)
                if criterion is not None and "t" in batch:
                    tmp_loss = criterion(
                        softmax_out.view(-1, softmax_out.shape[-1]),
                        batch["t"].view(-1),  # type:ignore
                    )
                    loss = tmp_loss if loss is None else loss + tmp_loss
                idxs = softmax_out.argmax(dim=2).tolist()
                tags.extend(
                    (
                        tuple(
                            dictionaries["t_map"].i2w[tag_idx]
                            for token_count, tag_idx in enumerate(sent)
                            # All sentences are padded (at the right end) to be of equal length.
                            # We do not want to return tags for the paddings.
                            # We check the information about lengths and paddings.
                            if token_count < batch["lens"][sent_idx]  # type: ignore
                        )
                        for sent_idx, sent in enumerate(idxs)
                    )
                )
        end = datetime.datetime.now()
        log.info(f"Tagged {sum((1 for sent in tags for token in sent))} tokens")
        log.info(f"Tagging took={end-start} seconds")
        if loss is None:
            return SimpleDataset(tags)
        return SimpleDataset(tags), loss.item()


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
