#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master-port 29500 \
  train/train_ministral_modified.py \
  --model_name_or_path /home/tos_data/LLM_HM_3_models/halltrain/basemodel/Ministral-3-3B-Instruct-BF16 \
  --training_data_path ./data/coco_2017.json \
  --training_image_dir ./data/data \
  --output_dir /home/tos_data/LLM_HM_3_models/test-models/Ministral-3epoch/+M \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 3 \
  --bf16 true \
  --save_strategy no \
  --save_total_limit 1 \
  --logging_steps 2 \
  --remove_unused_columns False \
  --deepspeed ./ds/ds_z2_config.json \
  --finetune_type full \
  --freeze_base_model true \
  --train_evidence_modules true \
  --enable_evidence true \
  --inject_position first_layer_input \
  --inject_op add \
  --use_utilization false \
  --evidence_source candidate \
  --gate_layers all \
  --lambda_orth 0.0 \
  --lambda_ctr 0.0
