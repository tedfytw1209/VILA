#!/bin/bash

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

STAGE3_PATH=$1
OUTPUT=$2
DATA_FILE=$3

model_max_length=131072
rope_theta=15300000

mkdir -p $OUTPUT

torchrun --nnodes=$n_node --nproc_per_node=4 --master_port=25001 \
        --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
        llava/train/train_llm_to_long.py  \
        --model_name_or_path $STAGE3_PATH/llm \
        --bf16 True \
        --data_file $DATA_FILE \
        --output_dir $OUTPUT       \
        --cache_dir ./cache-64k \
        --model_max_length $model_max_length \
        --data_max_length $model_max_length \
        --rope_theta $rope_theta \
        --use_flash_attn True \
        --low_rank_training True \
        --max_steps 40  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 4     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 40     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 2     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "./scripts/zero2.json" \
        --tf32 True

cp -r $STAGE3_PATH/vision_tower $OUTPUT
cp $STAGE3_PATH/config.json $OUTPUT
cp -r $STAGE3_PATH/mm_projector $OUTPUT

python3 llava/utils/merge_lora_weights_and_save_hf_model.py --base_model $STAGE3_PATH/llm \
	--peft_model $OUTPUT \
	--save_path $OUTPUT/llm \
	--rope_theta $rope_theta \
	--max_position_embeddings $model_max_length
