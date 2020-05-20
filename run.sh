#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09 10"
SPLITS="TM PM"
TAG_TYPES="fine"
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
                IN_FILE="$RAW_DIR"/otb/"$FOLD""$SPLIT".plain
                OUT_FILE="$FORMAT_DIR"/IFD-"$FOLD""$SPLIT".tsv
                echo "$IN_FILE -> $OUT_FILE"
                python preprocess/format.py format-tags "$IN_FILE" "$OUT_FILE" --tag_type "$TAG_TYPE"

                IN_FILE="$RAW_DIR"/mim/"$FOLD""$SPLIT".plain
                OUT_FILE="$FORMAT_DIR"/GOLD-"$FOLD""$SPLIT".tsv
                echo "$IN_FILE -> $OUT_FILE"
                python preprocess/format.py format-tags "$IN_FILE" "$OUT_FILE" --tag_type "$TAG_TYPE"
            done
        done
    done
    echo "Formatting DIIM"
    #python preprocess/vectorize_dim.py -i "$RAW_DIR"/SHsnid.csv/SHsnid.csv -o "$FORMAT_DIR"/dmii.vectors
fi
if ((FIRST_STEP <= 2 && LAST_STEP >= 2)); then
    echo "Running DyNet training and evaluation"
    out_folder="./out/DyNetIFD-1"
    mkdir -p $out_folder
    sbatch \
    --mem=40G \
    --partition=longrunning \
    --cpus-per-task 2 \
    --time 2-17:00:00 \
    --wrap="python evaluate.py --out_folder $out_folder --dataset_fold 1"
    #python evaluate.py --out_folder ./out/DyNetIFD-10 --dataset_fold 10
fi
if ((FIRST_STEP <= 3 && LAST_STEP >= 3)); then
    dt=$(date '+%Y-%m-%d_%H-%M-%S');
    out_folder="./out/$dt"
    mkdir -p "$out_folder"
    batch_size=32
    coarse_epochs=1
    fine_epochs=1
    debug=""
    sbatch \
    --gres=gpu \
    --mem=10G \
    --wrap="./main.py train-and-tag data/format/IFD-10TM.tsv data/format/IFD-10PM.tsv $out_folder --known_chars_file data/extra/characters_training.txt --morphlex_embeddings_file data/format/dmii.vectors --coarse_epochs $coarse_epochs --fine_epochs $fine_epochs --batch_size $batch_size $debug"
fi