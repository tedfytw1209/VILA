#!/bin/bash
ANS_PATH=$1
CKPT=$2
OUTPUT=$3

echo "$ANS_PATH $CKPT $OUTPUT"

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_iu_xray \
    --answer $ANS_PATH \
    --ground-truth $CKPT \
    --output $OUTPUT
