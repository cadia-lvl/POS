#!/bin/bash
#SBATCH --job-name=pos_lemmatizer
#SBATCH --gres=gpu:1
NAME=$1
# TRAIN=$2
# TEST=$3
# shift; 
# shift;
shift;
TRAIN="/home/haukurpj/Datasets/MIM-Correct/10TM.tsv"
TEST="/home/haukurpj/Datasets/MIM-Correct/10PM.tsv"
OUT_DIR="out/$NAME"
mkdir -p $OUT_DIR
echo $OUT_DIR
echo $TRAIN
echo $TEST
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
--run_name $NAME \
--lemmatizer \
--lemmatizer_hidden_dim 512 \
--lemmatizer_state_dict out/lemma-bin-no-clone-d32-drop08/model.pt \
--bert_encoder ~/Models/LM/electra-small-pytorch \
--tag_embedding_dim 32 \
--tag_embedding_dropout 0.0 \
--char_lstm_layers 1 \
--char_lstm_dim 256 \
--char_emb_dim 128 \
--label_smoothing 0.1 \
--epochs 20 \
--batch_size 64 \
--optimizer adam \
--learning_rate 5e-5 \
--scheduler none \
--gpu \
$*
