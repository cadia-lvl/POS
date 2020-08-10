#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09"
DATA_DIR=./data/raw

NAME="$1"
# Move the arguments forward
shift

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
        $DATA_DIR/otb/${fold}TM.plain \
        $DATA_DIR/otb/${fold}PM.plain \
        $DATA_DIR/mim/${fold}TM.plain \
        $DATA_DIR/mim/${fold}PM.plain \
        $out_folder \
        --epochs 20 \
        --batch_size 16 \
        --save_vocab \
        --save_model \
        --gpu \
        --optimizer sgd \
        --learning_rate 0.2 \
        --final_dim 32"
    done
fi