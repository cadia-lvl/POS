#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09 10"
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
        $DATA_DIR/mim_otb/${fold}TM.plain \
        $DATA_DIR/mim_otb/${fold}PM.plain \
        $out_folder \
        --bert_encoder electra_model \
        --main_lstm_layers 0 \
        --final_layer none \
        --label_smoothing 0.1 \
        --epochs 25 \
        --batch_size 16 \
        --save_vocab \
        --save_model \
        --gpu \
        --optimizer adam \
        --learning_rate 5e-5"
    done
fi