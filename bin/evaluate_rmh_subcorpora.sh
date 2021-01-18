#!/bin/bash
MODEL_PATH="out/m4_full/10"
OUT_DIR="data/pred"
for file in data/raw/gull2_/*.plain; do
    pos tag $MODEL_PATH/tagger.pt $file $OUT_DIR/"$(basename $file)".pred --contains_tags
done