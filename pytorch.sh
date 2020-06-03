#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09 10"
SPLITS="TM PM"
DATA_DIR=./data
FORMAT_DIR="$DATA_DIR"/format
RAW_DIR="$DATA_DIR"/raw

FIRST_STEP=1
LAST_STEP=1
if ((FIRST_STEP <= 1 && LAST_STEP >= 1)); then
    dt=$(date '+%Y-%m-%d_%H-%M-%S');
    out_folder="./out/$dt"
    mkdir -p "$out_folder"
    debug=""
    sbatch \
    --output="$out_folder/slurm-%j.out" \
    --gres=gpu \
    --mem=10G \
    --wrap="./main.py \
    train-and-tag \
    $RAW_DIR/otb/10TM.plain \
    $RAW_DIR/otb/10PM.plain \
    $out_folder \
    --epochs 1 \
    --batch_size 16 \
    --save_vocab \
    --save_model \
    --gpu \
    $debug"
fi