#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09 10"
SPLITS="TM PM"
TAG_TYPES="fine coarse"
DATA_DIR=./data
RAW_DIR="$DATA_DIR"/raw
FORMAT_DIR="$DATA_DIR"/format

FIRST_STEP=2
LAST_STEP=2
if ((FIRST_STEP <= 1 && LAST_STEP >= 1)); then
    mkdir -p "$FORMAT_DIR"
    echo "Formatting folds"
    for FOLD in $FOLDS; do
        for SPLIT in $SPLITS; do
            for TAG_TYPE in $TAG_TYPES; do
                IN_FILE="$RAW_DIR"/IFD2_SETS/"$FOLD""$SPLIT".txt
                OUT_FILE="$FORMAT_DIR"/IFD-"$FOLD""$SPLIT"."$TAG_TYPE"
                echo "$IN_FILE -> $OUT_FILE"
                python preprocess/format.py format-tags "$IN_FILE" "$OUT_FILE" --tag_type "$TAG_TYPE"

                IN_FILE="$RAW_DIR"/MIM-GOLD-1_0_SETS/MIM-GOLD-1_0_SETS/"$FOLD""$SPLIT".plain
                OUT_FILE="$FORMAT_DIR"/GOLD-"$FOLD""$SPLIT"."$TAG_TYPE"
                echo "$IN_FILE -> $OUT_FILE"
                python preprocess/format.py format-tags "$IN_FILE" "$OUT_FILE" --tag_type "$TAG_TYPE"
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