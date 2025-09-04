#!/bin/bash

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_resume_train.txt"

# 配置路径
export DATA_PATH="/DATASET/LISA_64"
export CKPT_PATH="/share_models/Qwen2-VL-2B-Instruct"
export SAVE_PATH="/share_models/Qwen2-VL-2B-Instruct_LISA"



# 总目标步数
TOTAL_STEPS=200

# 清理旧日志
rm -f $LOG_PATH

# 查找最新 checkpoint
latest_checkpoint=$(ls -d ${SAVE_PATH}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
echo "📢 Latest checkpoint detected: $latest_checkpoint"

# 默认值
final_steps=$TOTAL_STEPS
resume_flag=""

# 解析 trainer_state.json
if [ -n "$latest_checkpoint" ]; then
    trainer_state_file="$latest_checkpoint/trainer_state.json"
    if [ -f "$trainer_state_file" ]; then
        echo "✅ Found trainer_state.json, parsing..."
        trained_steps=$(python3 -c "import json; print(int(json.load(open('$trainer_state_file'))['global_step']))")
        echo "📚 Already trained $trained_steps steps."
        final_steps=$((TOTAL_STEPS - trained_steps))
        if [ $final_steps -le 0 ]; then
            final_steps=1
        fi
        resume_flag="--resume_from_checkpoint $latest_checkpoint"
    else
        echo "⚠️ No trainer_state.json found, starting fresh."
    fi
else
    echo "⚠️ No checkpoint found, starting from scratch."
fi

echo "🚀 Will continue training for $final_steps steps."

# 启动训练
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12345 \
    /data2/HSH/VRFT/Visual-RFT/code/grpo_lisa.py \
    --output_dir $SAVE_PATH \
    --model_name_or_path $CKPT_PATH \
    --dataset_name $DATA_PATH \
    --deepspeed /local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B-Instruct_LISA\
    --num_generations 8 \
    --save_steps 10 \
    --save_only_model true \
    --learning_rate 2e-6 \
    --save_total_limit 2 \
    --max_steps $final_steps \
    $resume_flag