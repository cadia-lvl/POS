#!/bin/bash
FOLDS="01 02 03 04 05 06 07 08 09"
DATA_DIR=./data/raw/mim

NAME="$1"
# Move the arguments forward
shift

# --begin=now+8hour \
for fold in $FOLDS
do
    out_folder=./out/"$NAME"/$fold
    mkdir -p "$out_folder"
    sbatch \
    --output="$out_folder/slurm-%j.out" \
    --gres=gpu \
    --mem=10G \
    --wrap="bin/run_model.sh $out_folder $TRAIN $TEST --gpu"
done