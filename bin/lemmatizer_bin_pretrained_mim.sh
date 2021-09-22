#!/bin/bash
#SBATCH --job-name=pos_lemmatizer
#SBATCH --gres=gpu:1
OUT_DIR=$1
TRAIN="/home/haukurpj/Datasets/MIM-Correct/10TM.tsv"
TRAIN=$2
TEST="/home/haukurpj/Datasets/MIM-Correct/10PM.tsv"
TEST=$3

mkdir -p $OUT_DIR
echo $OUT_DIR
echo $TRAIN
echo $TEST
shift; shift; shift;
echo $*
#    --morphlex_embeddings_file data/extra/dmii.vectors_filtered \
#    --morphlex_freeze \
#    --pretrained_word_embeddings_file data/extra/igc2018.vec_filtered \
#    --known_chars_file data/extra/characters_training.txt \
#    --label_smoothing 0.1 \
#    --main_lstm_layers 2 \
#    --word_embedding_dim 128 \
#    --pretrained_model_folder bull \
pos \
train-and-tag \
$TRAIN \
"$TEST" \
"$OUT_DIR" \
--lemmatizer \
--lemmatizer_hidden_dim 512 \
--lemmatizer_state_dict ~/Models/Lemmatizer/lemmatizer_bin/model_12.pt \
--bert_encoder ~/Models/LM/electra-small-pytorch \
--tag_embedding_dim 128 \
--char_lstm_layers 1 \
--char_lstm_dim 256 \
--char_emb_dim 128 \
--label_smoothing 0.1 \
--epochs 40 \
--batch_size 64 \
--optimizer adam \
--learning_rate 5e-5 \
--scheduler none \
--gpu \
$*
