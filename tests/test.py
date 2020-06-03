from pprint import pprint

import torch

from pos import train
from pos import data
from pos import model

train_file = './data/format/IFD-10TM.tsv'
test_file = './data/format/IFD-10PM.tsv'
c_tags_file = './data/extra/word_class_vocab.txt'
known_chars_file = './data/extra/characters_training.txt'
morphlex_embeddings_file = './data/format/dmii.vectors_short'
device = torch.device('cpu')

test_sent, test_sent_tags, _ = data.read_tsv('./test.tsv')
print(test_sent, test_sent_tags)

train_tokens, train_tags, _ = data.read_tsv(train_file)
test_tokens, test_tags, _ = data.read_tsv(test_file)
# Prepare the coarse tags
train_tags_coarse = data.coarsify(train_tags)
test_tags_coarse = data.coarsify(test_tags)

coarse_mapper, fine_mapper, embedding = train.create_mappers(
    train_tokens, test_tokens, train_tags, known_chars_file, c_tags_file, morphlex_embeddings_file)


x_format = coarse_mapper.make_x(test_sent)
pprint(x_format)
x_pad = coarse_mapper.pad_x(x_format)
pprint(x_pad)
x_idx = coarse_mapper.to_idx_x(x_pad)
pprint(x_idx)
train_iter = coarse_mapper.in_x_y_batches(x=test_sent,
                                          y=test_sent_tags,
                                          batch_size=2,
                                          shuffle=True,
                                          device=device)
model_parameters = {
    'lstm_dropouts': 0.1,
    'input_dropouts': 0.0,
    'emb_char_dim': 2,  # The characters are mapped to this dim
    'char_lstm_dim': 4,  # The character LSTM will output with this dim
    'emb_token_dim': 3,  # The tokens are mapped to this dim
    'main_lstm_dim': 64,  # The main LSTM dim will output with this dim
    'hidden_dim': 32,  # The main LSTM time-steps will be mapped to this dim
    'noise': 0.1,  # Noise to main_in, to main_bilstm
}
coarse_tagger = train.create_model(
    coarse_mapper, model_parameters, embedding, device, c_tags_embeddings=False)
input = x_idx
chars = input[:, :, :-2]
# (b, seq, 1)
w = input[:, :, -2]
m = input[:, :, -1]
char_embs = coarse_tagger.c_embs_dropout(
    coarse_tagger.character_embedding(chars))
print('char_embs')
pprint(char_embs)

words_as_chars = []
for b in range(char_embs.shape[0]):
    # b = words in sent, s = chars in word, f = char features
    # (b, s, f)
    sent_chars = char_embs[b, :, :, :]
    print('sent_chars')
    pprint(sent_chars)
    # some b's might only contain PADs, which pack_sequence does not like
    num_non_zero = torch.sum(
        torch.sum(torch.sum(sent_chars, dim=2), dim=1) != 0.0).item()
    dropped_pads = sent_chars[:int(num_non_zero), :, :]
    print('dropped_pads')
    pprint(dropped_pads)
    packed, lengths = coarse_tagger.pack_sequence(dropped_pads)
    sent_chars_rep = coarse_tagger.char_bilstm(packed)[0]
    un_packed = coarse_tagger.unpack_sequence(sent_chars_rep)
    print('un_packed')
    pprint(un_packed)
    sent_chars_rep_last_ts = torch.cat(
        [un_packed[idx, length - 1, :][None, :] for idx, length in enumerate(lengths.tolist())], dim=0)
    print('last_ts')
    pprint(sent_chars_rep_last_ts)
    added_pads = model.copy_into_larger_tensor(sent_chars_rep_last_ts,
                                               torch.zeros(size=(sent_chars.shape[0],
                                                                 sent_chars_rep_last_ts.shape[1])
                                                           )
                                               )
    print('added_pads')
    pprint(added_pads)
    words_as_chars.append(added_pads[None, :, :])
chars_as_word = torch.cat(words_as_chars, dim=0)
print('chars_as_word')
pprint(chars_as_word)
w_embs = coarse_tagger.w_embs_dropout(coarse_tagger.token_embedding(w))
print('w_embs')
pprint(w_embs)
m_embs = coarse_tagger.morph_lex_embedding(m)
pprint(m_embs)
# for x, y in train_iter:
#     print(x[0, :, :])
#     print(y[0, :])
#     print(x[1, :, :])
#     print(y[1, :])
