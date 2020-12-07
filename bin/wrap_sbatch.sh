#!/bin/bash
MODEL=$1
OUT_DIR=$2
TRAIN=$3
TEST=$4

# Pop the model.sh
shift
sbatch \
--output="$OUT_DIR/slurm-%j.out" \
--gres=gpu \
--mem=10G \
--wrap="$MODEL $* --gpu"