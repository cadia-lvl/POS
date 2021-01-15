#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09 10"
DATA_DIR=./data/raw/mim_otb

MODEL=$1
NAME="$2"
# Move the arguments forward
shift; shift

# --begin=now+8hour \
for fold in $FOLDS
do
    TRAIN=$DATA_DIR/${fold}TM.plain
    TEST=$DATA_DIR/${fold}PM.plain
    OUT_DIR=./out/"$NAME"/$fold
    ./bin/wrap_sbatch.sh $MODEL $OUT_DIR $TRAIN $TEST $*
done