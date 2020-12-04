#!/bin/bash
#    --morphlex_embeddings_file data/extra/dmii.vectors_filtered \
#    --morphlex_freeze \
#    --pretrained_word_embeddings_file data/extra/igc2018.vec_filtered \
#    --known_chars_file data/extra/characters_training.txt \
#    --label_smoothing 0.1 \
#    --main_lstm_layers 2 \
#    --word_embedding_dim 128 \
#    --pretrained_model_folder bull \
# --tagger \
# --lemmatizer \
pos \
train-and-tag \
tests/test_lemma.tsv \
tests/test_lemma.tsv \
debug/ \
--tagger \
--lemmatizer \
--word_embedding_dim 100 \
--main_lstm_layers 1 \
--label_smoothing 0.1 \
--epochs 25 \
--batch_size 16 \
--save_vocab \
--save_model \
--optimizer adam \
--learning_rate 5e-5 