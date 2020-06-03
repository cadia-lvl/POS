import logging

import torch
import torch.nn as nn

from . import data


log = logging.getLogger()


class ABLTagger(nn.Module):
    def __init__(self,
                 mapper: data.DataVocabMap,
                 device,
                 char_dim: int,  # The number of characters in dictionary
                 token_dim: int,  # The number of tokens in dictionary
                 tags_dim: int,  # The number of tags in dictionary - to predict
                 morph_lex_embeddings: torch.Tensor,
                 c_tags_embeddings: torch.Tensor,
                 emb_char_dim=20,  # The characters are mapped to this dim
                 char_lstm_dim=64,  # The character LSTM will output with this dim
                 emb_token_dim=128,  # The tokens are mapped to this dim
                 main_lstm_dim=64,  # The main LSTM dim will output with this dim
                 hidden_dim=32,  # The main LSTM time-steps will be mapped to this dim
                 lstm_dropouts=0.0,
                 input_dropouts=0.0,
                 noise=0.1):
        super(ABLTagger, self).__init__()
        self.mapper = mapper
        self.device = device
        self.noise = noise
        # Start with embeddings
        if morph_lex_embeddings is not None:
            self.morph_lex_embedding = nn.Embedding.from_pretrained(morph_lex_embeddings,
                                                                    freeze=False,
                                                                    padding_idx=data.PAD_ID,
                                                                    sparse=True)
        if c_tags_embeddings is not None:
            self.c_tags_embedding = nn.Embedding.from_pretrained(c_tags_embeddings,
                                                                 freeze=False,
                                                                 padding_idx=data.PAD_ID,
                                                                 sparse=True)
        self.token_embedding = nn.Embedding(token_dim,
                                            emb_token_dim,
                                            padding_idx=data.PAD_ID,
                                            sparse=True)
        nn.init.xavier_uniform_(self.token_embedding.weight[1:, :])
        self.character_embedding = nn.Embedding(char_dim,
                                                emb_char_dim,
                                                padding_idx=data.PAD_ID,
                                                sparse=True)
        nn.init.xavier_uniform_(self.character_embedding.weight[1:, :])
        # The character BiLSTM
        self.char_bilstm = nn.LSTM(input_size=emb_char_dim,
                                   hidden_size=char_lstm_dim,
                                   num_layers=1,
                                   dropout=lstm_dropouts,
                                   batch_first=True,
                                   bidirectional=True)
        for name, param in self.char_bilstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError('Unknown parameter in lstm={name}')
        # 2 * char-bilstm + token emb + morph_lex + coarse tags
        main_bilstm_dim = 0
        main_bilstm_dim += 2 * char_lstm_dim
        main_bilstm_dim += emb_token_dim
        main_bilstm_dim += self.morph_lex_embedding.weight.data.shape[1] if morph_lex_embeddings is not None else 0
        main_bilstm_dim += self.c_tags_embedding.weight.data.shape[1] if c_tags_embeddings is not None else 0
        self.bilstm = nn.LSTM(input_size=main_bilstm_dim,
                              hidden_size=main_lstm_dim,
                              num_layers=1,
                              dropout=lstm_dropouts,
                              batch_first=True,
                              bidirectional=True)
        for name, param in self.bilstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError('Unknown parameter in lstm={name}')
        # no bias in DyNet
        self.linear = nn.Linear(main_lstm_dim * 2, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.final = nn.Linear(hidden_dim, tags_dim, bias=False)
        nn.init.xavier_uniform_(self.final.weight)
        self.c_embs_dropout = nn.Dropout(p=input_dropouts)
        self.w_embs_dropout = nn.Dropout(p=input_dropouts)
        self.char_bilstm_out_dropout = nn.Dropout(p=input_dropouts)
        self.main_bilstm_out_dropout = nn.Dropout(p=input_dropouts)

    def forward(self, input):
        # input is (batch_size=num_sentence, max_seq_len_in_batch=max(len(sentences)), max_word_len_in_batch + 1 + 1)
        # (b, seq, chars)
        if hasattr(self, 'c_tags_embedding'):
            chars = input[:, :, :-3]
            # (b, seq, 1)
            w = input[:, :, -3]
            m = input[:, :, -2]
            c_tag = input[:, :, -1]
            c_tag_embs = self.c_tags_embedding(c_tag)
        else:
            chars = input[:, :, :-2]
            # (b, seq, 1)
            w = input[:, :, -2]
            m = input[:, :, -1]
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
                torch.sum(torch.sum(sent_chars, dim=2), dim=1) != 0.0).item()
            # Drop them
            dropped_pads = sent_chars[:int(num_non_zero), :, :]
            packed, lengths = self.pack_sequence(dropped_pads)
            sent_chars_rep = self.char_bilstm(packed)[0]
            un_packed = self.unpack_sequence(sent_chars_rep)
            # Get the last timestep, taking the PADs on char level into account
            sent_chars_rep_last_ts = torch.cat(
                [un_packed[idx, length - 1, :][None, :] for idx, length in enumerate(lengths.tolist())], dim=0)
            # Re-add the PAD words we removed before
            added_pads = copy_into_larger_tensor(sent_chars_rep_last_ts,
                                                 torch.zeros(size=(sent_chars.shape[0],
                                                                   sent_chars_rep_last_ts.shape[1])).to(
                                                     self.device)
                                                 )
            # Collect and add dimension to sum up
            words_as_chars.append(added_pads[None, :, :])
        chars_as_word = torch.cat(words_as_chars, dim=0)

        w_embs = self.w_embs_dropout(self.token_embedding(w))
        m_embs = self.morph_lex_embedding(m)
        if hasattr(self, 'c_tags_embedding'):
            main_in = torch.cat(
                (chars_as_word, w_embs, m_embs, c_tag_embs), dim=2)
        else:
            main_in = torch.cat((chars_as_word, w_embs, m_embs), dim=2)
        # Add noise - like in dyney
        if self.training:
            main_in = main_in + \
                torch.empty_like(main_in).normal_(0, self.noise)
        # (b, seq, f)
        self.bilstm.flatten_parameters()
        main_out = self.main_bilstm_out_dropout(
            self.unpack_sequence(self.bilstm(self.pack_sequence(main_in)[0])[0]))
        # We map each word to our targets
        out = self.final(torch.tanh(self.linear(main_out)))
        return out

    def pack_sequence(self, padded_sequence):
        """
        Packs the PAD in a sequence. Assumes that PAD=0.0 and appended.
        """
        # input:
        # (b, s, f)
        # lengths = (b, s)
        lengths = torch.sum(torch.pow(padded_sequence, 2), dim=2)
        # lengths = (b)
        lengths = torch.sum(lengths != torch.tensor(
            [0.0]).to(self.device), dim=1)
        return torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, lengths, batch_first=True, enforce_sorted=False), lengths

    def unpack_sequence(self, packed_sequence):
        """
        Inverse of pack_sequence
        """
        return torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=True)[0]


def copy_into_larger_tensor(tensor: torch.Tensor, like_tensor: torch.Tensor) -> torch.Tensor:
    """
    Only works for 2-dims
    """
    base = torch.zeros_like(like_tensor)
    base[:tensor.shape[0], :tensor.shape[1]] = tensor
    return base
