#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=16:00:00
#SBATCH --output=%x.%j.out

MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

echo "$MODEL_PATH $CKPT"

module load singularity
CUDA_VISIBLE_DEVICES=0

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda source activate base; conda activate vila; python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/iu_xray/simple_questions.jsonl \
    --image-folder /orange/bianjiang/VLM_dataset/ReportGeneration/IU_X-Ray/Kaggle/images/images_normalized \
    --answers-file ./eval_output/$CKPT/iu_xray/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE
