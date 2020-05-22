#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09 10"
SPLITS="TM PM"
DATA_DIR=./data
FORMAT_DIR="$DATA_DIR"/format

FIRST_STEP=1
LAST_STEP=1
if ((FIRST_STEP <= 1 && LAST_STEP >= 1)); then
    dt=$(date '+%Y-%m-%d_%H-%M-%S');
    out_folder="./out/$dt"
    mkdir -p "$out_folder"
    batch_size=64
    coarse_epochs=1
    fine_epochs=1
    debug=""
    sbatch \
    --gres=gpu \
    --mem=10G \
    --wrap="./main.py \
    train-and-tag \
    $FORMAT_DIR/IFD-10TM.tsv \
    $FORMAT_DIR/IFD-10PM.tsv \
    $out_folder \
    --known_chars_file data/extra/characters_training.txt \
    --morphlex_embeddings_file $FORMAT_DIR/dmii.vectors \
    --coarse_epochs $coarse_epochs \
    --fine_epochs $fine_epochs \
    --batch_size $batch_size \
    --save_vocab \
    --save_model \
    $debug"
fi