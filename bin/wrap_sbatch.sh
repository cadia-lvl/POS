#!/bin/bash
OUT_DIR=$1
mkdir -p $OUT_DIR
sbatch \
--output="$OUT_DIR/slurm-%j.out" \
--gres=gpu \
--mem=10G \
--wrap="bin/run_model.sh $* --gpu"