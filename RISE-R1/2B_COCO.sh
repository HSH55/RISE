#!/bin/bash

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_resume_train.txt"

export DATA_PATH="dataset/COCO/COCO_train"
export CKPT_PATH="Qwen2-VL-2B-Instruct"
export SAVE_PATH="MODEL/Qwen2-VL-2B-Instruct_COCO"

TOTAL_STEPS=200

run_training() {
    local steps=$1
    local ckpt=$2
    local save=$3

    latest_checkpoint=$(ls -d ${save}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    echo "📢 Latest checkpoint detected: $latest_checkpoint"

    final_steps=$steps
    resume_flag=""

    if [ -n "$latest_checkpoint" ]; then
        trainer_state_file="$latest_checkpoint/trainer_state.json"
        if [ -f "$trainer_state_file" ]; then
            echo "✅ Found trainer_state.json, parsing..."
            trained_steps=$(python3 -c "import json; print(int(json.load(open('$trainer_state_file'))['global_step']))")
            echo "📚 Already trained $trained_steps steps."
            final_steps=$((steps - trained_steps))
            if [ $final_steps -le 0 ]; then
                final_steps=1
            fi
            resume_flag="--resume_from_checkpoint $latest_checkpoint"
        fi
    fi

    echo "🚀 Will continue training for $final_steps steps."

    torchrun --nproc_per_node=8 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="127.0.0.1" \
        --master_port=12341 \
        /RISE-R1/grpo_multidet.py \
        --output_dir $save \
        --model_name_or_path $ckpt \
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
        --run_name Qwen2-VL-2B-Instruct_COCO \
        --num_generations 8 \
        --save_steps 10 \
        --save_only_model true \
        --learning_rate 2e-6 \
        --save_total_limit 2 \
        --max_steps $final_steps \
        $resume_flag
}


run_training $TOTAL_STEPS $CKPT_PATH $SAVE_PATH


