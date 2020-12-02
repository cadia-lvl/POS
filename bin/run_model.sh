#!/bin/bash
OUT_DIR=$1
TRAIN=$2
TEST=$3

echo $OUT_DIR
echo $TRAIN
echo $TEST
shift; shift; shift
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
"$TRAIN" \
"$TEST" \
"$OUT_DIR" \
--bert_encoder electra-small-pytorch \
--label_smoothing 0.1 \
--epochs 25 \
--batch_size 3 \
--save_vocab \
--save_model \
--optimizer adam \
--learning_rate 5e-5 \
$*