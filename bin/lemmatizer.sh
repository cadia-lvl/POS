#!/bin/bash
OUT_DIR=$1
TRAIN="/home/haukurpj/Resources/Data/MIM-GOLD-SETS.21.05/sets/10TM.tsv /home/haukurpj/Resources/Data/bin_data.tsv"
TEST="/home/haukurpj/Resources/Data/MIM-GOLD-SETS.21.05/sets/10PM.tsv"

mkdir -p $OUT_DIR
echo $OUT_DIR
echo $TRAIN
echo $TEST
shift;
echo $*
#    --morphlex_embeddings_file data/extra/dmii.vectors_filtered \
#    --morphlex_freeze \
#    --pretrained_word_embeddings_file data/extra/igc2018.vec_filtered \
#    --known_chars_file data/extra/characters_training.txt \
#    --label_smoothing 0.1 \
#    --main_lstm_layers 2 \
#    --word_embedding_dim 128 \
#    --pretrained_model_folder bull \
pos \
train-and-tag \
$TRAIN \
"$TEST" \
"$OUT_DIR" \
--adjust_lengths 1 \
--lemmatizer \
--lemmatizer_hidden_dim 512 \
--lemmatizer_state_dict ~/Resources/Models/Lemmatizer/lemmatizer-bin-data-512-trained-1e-4/state_dict_fixed.pt \
--tag_embedding_dim 128 \
--char_lstm_layers 1 \
--char_lstm_dim 128 \
--char_emb_dim 64 \
--label_smoothing 0.0 \
--epochs 40 \
--batch_size 512 \
--optimizer adam \
--learning_rate 5e-4 \
--scheduler none \
--gpu \
$*