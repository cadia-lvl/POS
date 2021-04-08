#!/bin/bash
MODEL="$1"
OUT_DIR="$2"
TRAIN="$3"
TEST="$4"
echo "model $MODEL"
echo "out_dir $OUT_DIR"
echo "train $TRAIN"
echo "test $TEST"
mkdir -p $OUT_DIR
# Pop the model.sh, out_dir, train and test
shift; shift; shift; shift
sbatch \
--output="$OUT_DIR/slurm-%j.out" \
--gres=gpu \
--mem=16G \
--wrap="$MODEL $OUT_DIR $TRAIN $TEST --gpu $*"