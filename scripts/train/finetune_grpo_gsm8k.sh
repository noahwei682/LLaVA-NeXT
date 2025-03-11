#!/bin/bash

# Distributed training settings
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=br-intranet
export NCCL_DEBUG=INFO

# Distributed training configuration
export ACCELERATE_CPU_AFFINITY=1 
export NPROC_PER_NODE=8  # Number of GPUs per node
export NODES=1           # Total number of nodes
export NODE_RANK=0      # Rank of the current node
export MASTER_ADDR=172.17.100.112  # Master node IP
export MASTER_PORT=23456           # Master node port

# Model configuration
LLM_VERSION="lmms-lab/llava-onevision-qwen2-7b-ov" 
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
# RANK=${RANK:-0}
# PORT=${PORT:-12345} 
# NUM_GPUS=${NUM_GPUS:5}
# NNODES=${NNODES:1}
############### Pretrain ################
BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
############### Finetune with GRPO on GSM8K ################
# Stage 2PROMPT_VERSION="qwen_1_5"
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-gsm8k-${LLM_VERSION_CLEAN}-grpo" 
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"

# Run training
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node $NPROC_PER_NODE --nnodes $NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/train_grpo.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --dataset_name "openai/gsm8k" \
    --dataset_config "main" \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last False \
    --group_by_length True \
    --logging_first_step True \
    --ddp_find_unused_parameters False \
    --seed 42 \
    --group_size 8 \
    --group_weight 0.5 \
    --relative_loss_type "log" \
    --group_margin 0.05 \
    --use_group_advantages True \
    --group_temperature 0.5 \
    --normalize_group_rewards True

exit 0; 
