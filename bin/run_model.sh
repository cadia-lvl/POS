#!/bin/bash
OUT_DIR=$1
TRAIN=$2
TEST=$3

echo $OUT_DIR
echo $TRAIN
echo $TEST
shift; shift; shift
echo $*
pos \
train-and-tag \
"$TRAIN" \
"$TEST" \
"$OUT_DIR" \
--bert_encoder electra-small-pytorch \
--label_smoothing 0.1 \
--epochs 25 \
--batch_size 3 \
--save_vocab \
--save_model \
--optimizer adam \
--learning_rate 5e-5 \
$*