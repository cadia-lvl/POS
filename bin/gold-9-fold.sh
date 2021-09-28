#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09"
FOLDS="01 02 03 04 05 06 07 09"
DATA_DIR="/home/haukurpj/Datasets/MIM-GOLD-SETS.21.05/sets"

MODEL=$1
NAME="$2"
# Move the arguments forward
shift; shift

# --begin=now+8hour \
for fold in $FOLDS
do
    TRAIN=$DATA_DIR/${fold}TM.tsv
    TEST=$DATA_DIR/${fold}PM.tsv
    FOLD_NAME="$NAME/gold-$fold"
    sbatch $MODEL $FOLD_NAME $TRAIN $TEST $*
done
