import torch
import torch.nn as nn

import data


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
                 lstm_dropouts=0.0):
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

        self.token_embedding = nn.Embedding(token_dim, emb_token_dim)
        self.character_embedding = nn.Embedding(char_dim, emb_char_dim)

        # The character BiLSTM
        self.char_bilstm = nn.LSTM(input_size=emb_char_dim,
                                   hidden_size=char_lstm_dim,
                                   num_layers=1,
                                   dropout=lstm_dropouts,
                                   batch_first=True,
                                   bidirectional=True)
        # 2 * char-bilstm + token emb + morph_lex + coarse tags
        main_bilstm_dim = (2 * char_lstm_dim
                           + emb_token_dim
                           + self.morph_lex_embedding.weight.data.shape[1] if self.morph_lex_embedding is not None else 0
                           + self.c_tags_embedding.weight.data.shape[1] if self.c_tags_embedding is not None else 0)
        self.bilstm = nn.LSTM(input_size=main_bilstm_dim,
                              hidden_size=main_lstm_dim,
                              num_layers=1,
                              dropout=lstm_dropouts,
                              batch_first=True,
                              bidirectional=True)
        self.linear = nn.Linear(main_lstm_dim * 2, hidden_dim)
        self.final = nn.Linear(hidden_dim, tags_dim)

    def forward(self, input):
        # input is (batch_size=num_sentence, max_seq_len_in_batch=max(len(sentences)), max_word_len_in_batch + 1 + 1)
        # (b, seq, chars)
        chars = input[:, :, :-2]
        # (b, seq, 1)
        w = input[:, :, -2]
        m = input[:, :, -1]
        # (b, seq, chars, f)
        char_embs = self.character_embedding(chars)
        # We process a single word at a time (many chars) in the LSTM
        # [(b, 1, f) for s in seq],
        chars_as_word = torch.cat(
            [self.char_bilstm(
                char_embs[:, s, :, :]
            )[0][:, -1, :][:, None, :] for s in range(char_embs.shape[1])],
            dim=1)

        w_embs = self.token_embedding(w)
        m_embs = self.morph_lex_embedding(m)
        main_in = torch.cat((chars_as_word, w_embs, m_embs), dim=2)
        # (b, seq, f)
        main_out = self.bilstm(main_in)[0]
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
