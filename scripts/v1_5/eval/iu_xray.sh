#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

echo "$MODEL_PATH $CKPT"

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/iu_xray/simple_questions.jsonl \
    --image-folder /orange/bianjiang/VLM_dataset/ReportGeneration/IU_X-Ray/Kaggle/images/images_normalized \
    --answers-file ./eval_output/$CKPT/iu_xray/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE
