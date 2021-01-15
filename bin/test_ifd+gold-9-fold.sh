#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09"
DATA_DIR=./data/raw

MODEL=$1
NAME="$2"
# Move the arguments forward
shift; shift

# --begin=now+8hour \
for fold in $FOLDS
do
    TRAIN="$DATA_DIR/mim/${fold}TM.plain $DATA_DIR/otb/${fold}TM.plain $DATA_DIR/otb/${fold}PM.plain"
    TEST=$DATA_DIR/mim/${fold}PM.plain
    OUT_DIR=./out/"$NAME"/$fold
    ./bin/wrap_sbatch.sh $MODEL $OUT_DIR $TRAIN $TEST $*
done