#!/usr/bin/env bash
set -euo pipefail

# One-click training for 3 base models × 3 variants (+M, +M+A, Ours).
# Run from: /home/tos_data/LLM_HM_3_model/halltrain
#
# Notes:
# - This script assumes data is available at ./data/train.json and ./data/img
# - Base models can be local dirs under ./basemodel/*, or HF model ids.
# - By default we use single-process `python`. For multi-GPU, set NPROC_PER_NODE>1
#   and ENABLE_DEEPSPEED=1 to add `--deepspeed ds/ds_z2_config.json`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-Hall}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/tos_data/LLM_HM_3_model/Fintune_model_output}"
DATA_JSON="${DATA_JSON:-$ROOT_DIR/data/train.json}"
IMG_DIR="${IMG_DIR:-$ROOT_DIR/data/img}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-21443}"

ENABLE_DEEPSPEED="${ENABLE_DEEPSPEED:-0}"
DS_CONFIG="${DS_CONFIG:-$ROOT_DIR/ds/ds_z2_config.json}"

LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

maybe_activate_conda() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "$CONDA_ENV" || true
  fi
}

run_train() {
  local entry_script="$1"
  local model_name_or_path="$2"
  local output_dir="$3"
  shift 3

  mkdir -p "$output_dir"

  local -a cmd_base
  cmd_base=(
    --model_name_or_path "$model_name_or_path"
    --training_data_path "$DATA_JSON"
    --training_image_dir "$IMG_DIR"
    --output_dir "$output_dir"
    --save_total_limit 2
    --report_to none
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LEARNING_RATE"
    --num_train_epochs "$NUM_EPOCHS"
    --bf16 true
    --resume_from_checkpoint False
    --save_strategy epoch
    --logging_steps 2
    --remove_unused_columns False
  )

  if [[ "$ENABLE_DEEPSPEED" == "1" ]]; then
    cmd_base+=( --deepspeed "$DS_CONFIG" )
  fi

  if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    echo "[RUN] torchrun -n $NPROC_PER_NODE $entry_script -> $output_dir"
    torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" --master-port "$MASTER_PORT" \
      "$entry_script" "${cmd_base[@]}" "$@"
  else
    echo "[RUN] python $entry_script -> $output_dir"
    python "$entry_script" "${cmd_base[@]}" "$@"
  fi
}

main() {
  maybe_activate_conda

  mkdir -p "$OUTPUT_ROOT"

  # Base model locations (change these if you prefer HF model ids)
  QWEN_2B="${QWEN_2B:-$ROOT_DIR/basemodel/Qwen3-VL-2B-Instruct}"
  QWEN_4B="${QWEN_4B:-$ROOT_DIR/basemodel/Qwen3-VL-4B-Instruct}"
  MINISTRAL_3B="${MINISTRAL_3B:-$ROOT_DIR/basemodel/Ministral-3-3B-Instruct-2512}"

  # 1) Qwen3-VL-2B
  run_train "train/train_qwen.py" "$QWEN_2B" "$OUTPUT_ROOT/output_model_Qwen3-VL-2B_+M" \
    --enable_evidence true --inject_position first_layer_input --inject_op add \
    --use_utilization false --evidence_source candidate \
    --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""

  run_train "train/train_qwen.py" "$QWEN_2B" "$OUTPUT_ROOT/output_model_Qwen3-VL-2B_+M+A" \
    --enable_evidence true --inject_position per_layer --inject_op ours \
    --use_utilization true --evidence_source aligned \
    --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""

  run_train "train/train_qwen.py" "$QWEN_2B" "$OUTPUT_ROOT/output_model_Qwen3-VL-2B_Ours" \
    --enable_evidence true --inject_position per_layer --inject_op ours \
    --use_utilization true --evidence_source aligned \
    --lambda_orth 1.0 --lambda_ctr 1.0 --tau 0.07 --aux_layers ""

  # 2) Qwen3-VL-4B
  run_train "train/train_qwen.py" "$QWEN_4B" "$OUTPUT_ROOT/output_model_Qwen3-VL-4B_+M" \
    --enable_evidence true --inject_position first_layer_input --inject_op add \
    --use_utilization false --evidence_source candidate \
    --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""

  run_train "train/train_qwen.py" "$QWEN_4B" "$OUTPUT_ROOT/output_model_Qwen3-VL-4B_+M+A" \
    --enable_evidence true --inject_position per_layer --inject_op ours \
    --use_utilization true --evidence_source aligned \
    --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""

  run_train "train/train_qwen.py" "$QWEN_4B" "$OUTPUT_ROOT/output_model_Qwen3-VL-4B_Ours" \
    --enable_evidence true --inject_position per_layer --inject_op ours \
    --use_utilization true --evidence_source aligned \
    --lambda_orth 1.0 --lambda_ctr 1.0 --tau 0.07 --aux_layers ""

  # 3) Ministral-3-3B (2512)
  run_train "train/train_ministral.py" "$MINISTRAL_3B" "$OUTPUT_ROOT/output_model_Ministral-3-3B_+M" \
    --enable_evidence true --inject_position first_layer_input --inject_op add \
    --use_utilization false --evidence_source candidate \
    --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""

  run_train "train/train_ministral.py" "$MINISTRAL_3B" "$OUTPUT_ROOT/output_model_Ministral-3-3B_+M+A" \
    --enable_evidence true --inject_position per_layer --inject_op ours \
    --use_utilization true --evidence_source aligned \
    --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""

  run_train "train/train_ministral.py" "$MINISTRAL_3B" "$OUTPUT_ROOT/output_model_Ministral-3-3B_Ours" \
    --enable_evidence true --inject_position per_layer --inject_op ours \
    --use_utilization true --evidence_source aligned \
    --lambda_orth 1.0 --lambda_ctr 1.0 --tau 0.07 --aux_layers ""
}

main "$@"
