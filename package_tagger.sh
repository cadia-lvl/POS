#!/bin/bash
FILE_NAME=$1
MODEL_DIR=$2
tar czf $FILE_NAME -C $MODEL_DIR \
    config.json \
    dictionaries.pickle \
    hyperparamters.json \
    known_lemmas.txt \
    known_toks.txt \
    special_tokens_map.json \
    model.pt \
    tokenizer_config.json \
    vocab.txt