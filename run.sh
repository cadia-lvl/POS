#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09 10"
SPLITS="TM PM"
TAG_TYPES="fine coarse"
DATA_DIR=./data
RAW_DIR="$DATA_DIR"/raw
FORMAT_DIR="$DATA_DIR"/format

FIRST_STEP=1
LAST_STEP=1
if ((FIRST_STEP <= 1 && LAST_STEP >= 1)); then
    mkdir -p "$FORMAT_DIR"
    echo "Formatting folds"
    for FOLD in $FOLDS; do
        for SPLIT in $SPLITS; do
            for TAG_TYPE in $TAG_TYPES; do
                python preprocess/format.py format-tags "$RAW_DIR"/IFD2_SETS/"$FOLD""$SPLIT".txt "$FORMAT_DIR"/IFD-"$FOLD""$SPLIT"."$TAG_TYPE" --tag_type "$TAG_TYPE"
            done
        done
    done
    echo "Formatting DIIM"
    python preprocess/vectorize_dim.py -i "$RAW_DIR"/SHsnid.csv/SHsnid.csv -o "$FORMAT_DIR"/dmii.vectors
fi
if ((FIRST_STEP <= 2 && LAST_STEP >= 2)); then
    echo "Running training and evaluation"
    python evaluate.py --dataset_fold 1
    python evaluate.py --dataset_fold 10
fi