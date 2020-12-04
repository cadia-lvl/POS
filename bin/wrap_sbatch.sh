#!/bin/bash
OUT_DIR=$1
TRAIN=$2
TEST=$3
sbatch \
--output="$OUT_DIR/slurm-%j.out" \
--gres=gpu \
--mem=10G \
--wrap="bin/run_model.sh $* --gpu"