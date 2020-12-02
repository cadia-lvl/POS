#!/bin/bash
# We use relative paths so the script needs to be run in the correct place.
DATA_DIR=./data
RAW_DIR="$DATA_DIR"/raw

NAME="$1"
COUNT="$2"
TRAIN=$RAW_DIR/mim/10TM.plain
TRAIN=tests/test.tsv
TEST=$RAW_DIR/mim/10PM.plain
TEST=tests/test.tsv
for i in $(seq $COUNT)
do
    out_folder=./out/"$NAME"/$i
    mkdir -p "$out_folder"
    sbatch \
    --output="$out_folder/slurm-%j.out" \
    --gres=gpu \
    --mem=10G \
    --wrap="bin/run_model.sh $out_folder $TRAIN $TEST"
done