#!/bin/bash
# We use relative paths so the script needs to be run in the correct place.
DATA_DIR=./data
RAW_DIR="$DATA_DIR"/raw

NAME="$1"
# Move the arguments forward
shift
#    --morphlex_extra_dim 32 \
#    --morphlex_embeddings_file data/extra/dmii.vectors_filtered \
#    --morphlex_freeze \
#    --pretrained_word_embeddings_file data/extra/igc2018.vec_filtered \
#    --known_chars_file data/extra/characters_training.txt \
#    --label_smoothing 0.1 \
#    --main_lstm_layers 2 \
#    --word_embedding_dim 128 \
#    --final_layer attention \
#    --final_layer_attention_heads 2 \
#    --pretrained_model_folder bull \
FIRST_STEP=1
LAST_STEP=1
if ((FIRST_STEP <= 1 && LAST_STEP >= 1)); then
    # dt=$(date '+%Y-%m-%d_%H-%M-%S');
    out_folder=./out/"$NAME"
    mkdir -p "$out_folder"
    sbatch \
    --output="$out_folder/slurm-%j.out" \
    --gres=gpu \
    --mem=10G \
    --wrap="pos \
    train-and-tag \
    $RAW_DIR/mim/10TM.plain \
    $RAW_DIR/mim/10PM.plain \
    $out_folder \
    --bert_encoder electra_model \
    --main_lstm_layers 0 \
    --final_layer none \
    --label_smoothing 0.1 \
    --epochs 25 \
    --batch_size 16 \
    --save_vocab \
    --save_model \
    --gpu \
    --optimizer adam \
    --learning_rate 5e-5"
fi