#!/bin/bash
echo $*
MODEL="$1"
OUT_DIR="$2"
TRAIN="$3"
TEST="$4"
mkdir -p $OUT_DIR
# Pop the model.sh, out_dir, train and test
shift; shift; shift; shift
echo $*
echo $MODEL
echo $OUT_DIR
sbatch \
--output="$OUT_DIR/slurm-%j.out" \
--gres=gpu \
--mem=10G \
--wrap="$MODEL $TRAIN $TEST $OUT_DIR --gpu"