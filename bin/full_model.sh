#!/bin/bash
DATA_DIR=./data/raw

NAME="$1"
# Move the arguments forward
shift

FIRST_STEP=1
LAST_STEP=1
if ((FIRST_STEP <= 1 && LAST_STEP >= 1)); then
    out_folder=./out/"$NAME"
    mkdir -p "$out_folder"
    sbatch \
    --output="$out_folder/slurm-%j.out" \
    --gres=gpu \
    --mem=30G \
    --wrap="pos \
    train-and-tag \
    $DATA_DIR/otb/01TM.plain \
    $DATA_DIR/otb/01PM.plain \
    $DATA_DIR/mim/10TM.plain \
    $DATA_DIR/mim/10PM.plain \
    $out_folder \
    --pretrained_word_embeddings_file data/extra/igc2018.vec \
    --morphlex_embeddings_file data/extra/dmii.vectors \
    --morphlex_freeze \
    --known_chars_file data/extra/characters_training.txt \
    --main_lstm_layers 2 \
    --label_smoothing 0.1 \
    --epochs 40 \
    --batch_size 16 \
    --save_vocab \
    --save_model \
    --gpu \
    --optimizer sgd \
    --learning_rate 0.2"
fi