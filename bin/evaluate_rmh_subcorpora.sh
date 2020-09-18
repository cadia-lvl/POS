#!/bin/bash
MODEL_PATH="out/full-v2"
OUT_DIR="data/pred"
for file in data/raw/gull2_/*.plain; do
    pos tag $MODEL_PATH/tagger.pt $MODEL_PATH/dictionaries.pickle $file $OUT_DIR/"$(basename $file)".pred --contains_tags
done