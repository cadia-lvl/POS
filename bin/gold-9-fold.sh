#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09"
DATA_DIR=./data/raw/mim

NAME="$1"
# Move the arguments forward
shift

# --begin=now+8hour \
#        --morphlex_embeddings_file data/extra/dmii.vectors_filtered \
#        --morphlex_freeze \
#        --morphlex_extra_dim 32 \
#        --pretrained_word_embeddings_file data/extra/igc2018.vec_filtered \
#        --known_chars_file data/extra/characters_training.txt \
#        --label_smoothing 0.1 \
#        --main_lstm_layers 2 \
#        --word_embedding_dim 128 \
FIRST_STEP=1
LAST_STEP=1
if ((FIRST_STEP <= 1 && LAST_STEP >= 1)); then
    for fold in $FOLDS; do
        out_folder=./out/"$NAME"/$fold
        mkdir -p "$out_folder"
        sbatch \
        --output="$out_folder/slurm-%j.out" \
        --gres=gpu \
        --mem=10G \
        --wrap="pos \
        train-and-tag \
        $DATA_DIR/${fold}TM.plain \
        $DATA_DIR/${fold}PM.plain \
        $out_folder \
        --morphlex_embeddings_file data/extra/dmii.vectors_filtered \
        --morphlex_freeze \
        --pretrained_word_embeddings_file data/extra/igc2018.vec_filtered \
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
    done
fi