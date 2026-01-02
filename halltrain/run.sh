#!/bin/bash
# Qwen3-VL-8B-Instruct 训练脚本 (双GPU)

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 --master-port 21443 train.py \
    --model_name_or_path ./basemodel \
    --training_data_path ./demo_data/train.json \
    --training_image_dir ./demo_data/img \
    --output_dir your dir \
    --save_total_limit 2 \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 3 \
    --deepspeed ds/ds_z3_config.json \
    --bf16 true \
    --resume_from_checkpoint False \
    --save_strategy epoch \
    --logging_steps 2 \
    --remove_unused_columns False \
    --lambda_orth 0.0 \
    --lambda_ctr 0.0 \
    --tau 0.07 \
    --aux_layers ""
