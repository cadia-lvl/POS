#!/bin/bash
# We use relative paths so the script needs to be run in the correct place.
DATA_DIR=./data
RAW_DIR="$DATA_DIR"/raw

MODEL="$1"
NAME="$2"
COUNT="$3"
TRAIN=$RAW_DIR/mim/10TM.plain
TEST=$RAW_DIR/mim/10PM.plain
for i in $(seq $COUNT)
do
    OUT_DIR=./out/"$NAME"/$i
    ./bin/wrap_sbatch.sh $MODEL $TRAIN $TEST $OUT_DIR $*
done