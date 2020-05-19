import torch
import torch.nn as nn

from . import data


class ABLTagger(nn.Module):
    def __init__(self,
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
                 input_dropouts=0.0):
        super(ABLTagger, self).__init__()
        # Start with embeddings
        if morph_lex_embeddings is not None:
            self.morph_lex_embedding = nn.Embedding(num_embeddings=morph_lex_embeddings.shape[0],
                                                    embedding_dim=morph_lex_embeddings.shape[1],
                                                    padding_idx=data.PAD_ID)
            # Here we could also use weight=torch.nn.Parameter(tensor, requires_grad=False)
            self.morph_lex_embedding.weight.data = morph_lex_embeddings
            self.morph_lex_embedding.weight.requires_grad = False
        if c_tags_embeddings is not None:
            self.c_tags_embedding = nn.Embedding(num_embeddings=c_tags_embeddings.shape[0],
                                                 embedding_dim=c_tags_embeddings.shape[1],
                                                 padding_idx=data.PAD_ID)
            # Don't update these weights
            self.c_tags_embedding.weight.data = c_tags_embeddings
            self.c_tags_embedding.weight.requires_grad = False

        self.token_embedding = nn.Embedding(
            token_dim, emb_token_dim, padding_idx=data.PAD_ID)
        self.character_embedding = nn.Embedding(
            char_dim, emb_char_dim, padding_idx=data.PAD_ID)

        # The character BiLSTM
        self.char_bilstm = nn.LSTM(input_size=emb_char_dim,
                                   hidden_size=char_lstm_dim,
                                   num_layers=1,
                                   dropout=lstm_dropouts,
                                   batch_first=True,
                                   bidirectional=True)
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
        self.linear = nn.Linear(main_lstm_dim * 2, hidden_dim)
        self.final = nn.Linear(hidden_dim, tags_dim)
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
        # We process a single word at a time (many chars) in the LSTM
        # [(b, 1, f) for s in seq],
        chars_as_word = self.char_bilstm_out_dropout(torch.cat(
            [self.char_bilstm(
                char_embs[:, s, :, :]
            )[0][:, -1, :][:, None, :] for s in range(char_embs.shape[1])],
            dim=1))
        w_embs = self.w_embs_dropout(self.token_embedding(w))
        m_embs = self.morph_lex_embedding(m)
        if hasattr(self, 'c_tags_embedding'):
            main_in = torch.cat(
                (chars_as_word, w_embs, m_embs, c_tag_embs), dim=2)
        else:
            main_in = torch.cat((chars_as_word, w_embs, m_embs), dim=2)
        # (b, seq, f)
        self.bilstm.flatten_parameters()
        main_out = self.main_bilstm_out_dropout(self.bilstm(main_in)[0])
        # We map each word to our targets
        # [(b, 1, f) for s in seq]:
        out = torch.cat(
            [self.final(
                torch.tanh(
                    self.linear(main_out[:, s, :])
                )
            )[:, None, :] for s in range(main_out.shape[1])],
            dim=1)
        # (b, s, target)
        return out
