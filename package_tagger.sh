#!/bin/bash
FILE_NAME=$1
MODEL_DIR=$2
zip $FILE_NAME \
    $MODEL_DIR/config.json \
    $MODEL_DIR/dictionaries.pickle \
    $MODEL_DIR/hyperparamters.json \
    $MODEL_DIR/known_lemmas.txt \
    $MODEL_DIR/known_toks.txt \
    $MODEL_DIR/special_tokens_map.json \
    $MODEL_DIR/tagger.pt \
    $MODEL_DIR/tokenizer_config.json \
    $MODEL_DIR/vocab.txt