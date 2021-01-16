#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09"
DATA_DIR=./data/raw

NAME="$1"
# Move the arguments forward

# --begin=now+8hour \
for fold in $FOLDS
do
    OUT_DIR=./out/"$NAME"/$fold
    mkdir -p $OUT_DIR
    sbatch \
    --output="$OUT_DIR/slurm-%j.out" \
    --gres=gpu \
    --mem=10G \
    --wrap="pos \
    train-and-tag \
    $DATA_DIR/otb/${fold}TM.plain \
    $DATA_DIR/otb/${fold}PM.plain \
    $DATA_DIR/mim/${fold}TM.plain \
    $DATA_DIR/mim/${fold}PM.plain \
    $OUT_DIR \
    --tagger \
    --bert_encoder electra-small-pytorch \
    --morphlex_embeddings_file data/extra/dmii.vectors_filtered \
    --morphlex_freeze \
    --pretrained_word_embeddings_file data/extra/igc2018.vec_filtered \
    --known_chars_file data/extra/characters_training.txt \
    --char_lstm_layers 1 \
    --char_emb_dim 128 \
    --main_lstm_dim 256 \
    --main_lstm_layers 1 \
    --label_smoothing 0.1 \
    --epochs 20 \
    --batch_size 8 \
    --save_vocab \
    --save_model \
    --optimizer sgd \
    --learning_rate 5e-2"
done