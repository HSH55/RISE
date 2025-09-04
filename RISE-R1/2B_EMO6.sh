export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_GRPO_DIOR.txt"

export DATA_PATH=/Emotion6/emodataset_rft
export CKPT_PATH=/Qwen2-VL-2B-Instruct
export SAVE_PATH=/share_models/Qwen2-VL-2B-Instruct_emo6


torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    /RISE-R1/grpo_emo6.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed /local_scripts/zero3.json \
    --max_prompt_length 1024\
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps 200 \
    --weight_decay 0.1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 14 \
    --run_name Qwen2-VL-2B_train_EMO6\
    --num_generations 8 \
    --save_steps 50 \
    --save_total_limit 20 \
    --save_only_model true \
