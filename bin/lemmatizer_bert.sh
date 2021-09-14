#!/bin/bash
OUT_DIR=$1
TRAIN="/home/haukurpj/Datasets/MIM-Correct/10TM.tsv"
TEST="/home/haukurpj/Datasets/MIM-Correct/10PM.tsv"

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
--adjust_lengths 128 \
--lemmatizer \
--lemmatizer_hidden_dim 512 \
--tag_embedding_dim 128 \
--char_lstm_layers 1 \
--char_lstm_dim 256 \
--char_emb_dim 128 \
--bert_encoder /data2/scratch/haukurpj/Models/Models/LM/electra-small-pytorch \
--no_lemmatizer_char_attention \
--label_smoothing 0.1 \
--epochs 20 \
--batch_size 16 \
--optimizer adam \
--learning_rate 5e-4 \
--scheduler none \
--gpu \
$*
