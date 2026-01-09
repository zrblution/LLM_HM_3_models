#!/bin/bash
################################################################################
# run_finetune_all.sh - LLM-HM 9个模型自动化训练脚本
################################################################################
#
# 【功能说明】
# 本脚本用于自动化训练 9 个模型：
#   - Qwen3-VL-2B: +M, +M+A, Ours
#   - Qwen3-VL-4B: +M, +M+A, Ours
#   - Ministral-3-3B: +M, +M+A, Ours
#
# 【硬件要求】
#   - 4张 GPU (CUDA_VISIBLE_DEVICES=0,1,2,3)
#   - 串行执行，每个任务完成后自动开始下一个
#
# 【三种模型变体参数配置】
#
# +M (首层注入，仅 CE 损失):
#   --enable_evidence true
#   --inject_position first_layer_input
#   --inject_op add
#   --use_utilization false
#   --evidence_source candidate
#   --lambda_orth 0.0
#   --lambda_ctr 0.0
#
# +M+A (全层注入，仅 CE 损失):
#   --enable_evidence true
#   --inject_position per_layer
#   --inject_op ours
#   --use_utilization true
#   --evidence_source aligned
#   --lambda_orth 0.0
#   --lambda_ctr 0.0
#
# Ours (全层注入 + 正则化损失):
#   --enable_evidence true
#   --inject_position per_layer
#   --inject_op ours
#   --use_utilization true
#   --evidence_source aligned
#   --lambda_orth 1.0
#   --lambda_ctr 1.0
#
################################################################################

# 失败任务记录
FAILED_TASKS=()
SUCCESS_COUNT=0
FAIL_COUNT=0

# ============================================================================
# 路径配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 基础模型路径
QWEN2B_BASE="./basemodel/Qwen3-VL-2B-Instruct"
QWEN4B_BASE="./basemodel/Qwen3-VL-4B-Instruct"
MINISTRAL3B_BASE="./basemodel/Ministral-3-3B-Instruct"

# 输出根目录
OUTPUT_ROOT="/home/tos_data/LLM_HM_3_models/Fitntune_model_output_new"

# 训练数据路径
DATA_JSON="./data/coco_2017.json"
IMG_DIR="./data/data"

# DeepSpeed 配置
DS_CONFIG="./ds/ds_z2_config.json"

# GPU 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=1
LR="1.0e-5"

# 端口起始值（每个任务递增避免冲突）
BASE_PORT=29500
CURRENT_PORT=$BASE_PORT

# ============================================================================
# 辅助函数
# ============================================================================

# 日志函数
log() {
    echo ""
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "========================================================================"
    echo ""
}

# 清理 GPU 显存和僵尸进程
cleanup_gpu() {
    log "清理 GPU 显存和僵尸进程..."
    
    # 杀掉可能残留的 Python 训练进程
    pkill -9 -f "train_qwen.py" 2>/dev/null || true
    pkill -9 -f "train_ministral.py" 2>/dev/null || true
    pkill -9 -f "torchrun" 2>/dev/null || true
    
    # 清理 GPU 上的僵尸进程
    for gpu_id in 0 1 2 3; do
        fuser -k /dev/nvidia${gpu_id} 2>/dev/null || true
    done
    
    # 等待进程完全退出
    sleep 10
    
    # 清理 PyTorch 缓存
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # 清理分布式训练残留的共享内存
    rm -rf /dev/shm/nccl* 2>/dev/null || true
    
    log "GPU 清理完成"
}

# 获取下一个可用端口
get_next_port() {
    CURRENT_PORT=$((CURRENT_PORT + 1))
    echo $CURRENT_PORT
}

# ============================================================================
# 训练函数
# ============================================================================

# Qwen 模型训练函数
train_qwen() {
    local model_path="$1"
    local output_dir="$2"
    local variant_name="$3"
    local enable_evidence="$4"
    local inject_position="$5"
    local inject_op="$6"
    local use_utilization="$7"
    local evidence_source="$8"
    local lambda_orth="$9"
    local lambda_ctr="${10}"
    
    local port=$(get_next_port)
    
    log "开始训练 Qwen 模型: $variant_name"
    echo "  模型路径: $model_path"
    echo "  输出目录: $output_dir"
    echo "  端口: $port"
    echo "  参数配置:"
    echo "    - enable_evidence: $enable_evidence"
    echo "    - inject_position: $inject_position"
    echo "    - inject_op: $inject_op"
    echo "    - use_utilization: $use_utilization"
    echo "    - evidence_source: $evidence_source"
    echo "    - lambda_orth: $lambda_orth"
    echo "    - lambda_ctr: $lambda_ctr"
    
    mkdir -p "$output_dir"
    
    # 设置环境变量
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    if torchrun --nnodes 1 --nproc_per_node $NPROC_PER_NODE --master-port $port \
        train/train_qwen.py \
        --model_name_or_path "$model_path" \
        --training_data_path "$DATA_JSON" \
        --training_image_dir "$IMG_DIR" \
        --output_dir "$output_dir" \
        --save_total_limit 2 \
        --report_to none \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LR \
        --num_train_epochs $NUM_EPOCHS \
        --bf16 true \
        --resume_from_checkpoint False \
        --save_strategy epoch \
        --logging_steps 2 \
        --remove_unused_columns False \
        --deepspeed "$DS_CONFIG" \
        --finetune_type full \
        --freeze_base_model true \
        --train_evidence_modules true \
        --enable_evidence $enable_evidence \
        --inject_position "$inject_position" \
        --inject_op "$inject_op" \
        --use_utilization $use_utilization \
        --evidence_source "$evidence_source" \
        --gate_layers all \
        --lambda_orth $lambda_orth \
        --lambda_ctr $lambda_ctr \
        --tau 0.07 \
        --aux_layers ""; then
        log "✅ 完成训练: $variant_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log "❌ 训练失败: $variant_name (继续执行下一个任务)"
        FAILED_TASKS+=("$variant_name")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# Ministral 模型训练函数
train_ministral() {
    local model_path="$1"
    local output_dir="$2"
    local variant_name="$3"
    local enable_evidence="$4"
    local inject_position="$5"
    local inject_op="$6"
    local use_utilization="$7"
    local evidence_source="$8"
    local lambda_orth="$9"
    local lambda_ctr="${10}"
    
    local port=$(get_next_port)
    
    log "开始训练 Ministral 模型: $variant_name"
    echo "  模型路径: $model_path"
    echo "  输出目录: $output_dir"
    echo "  端口: $port"
    echo "  参数配置:"
    echo "    - enable_evidence: $enable_evidence"
    echo "    - inject_position: $inject_position"
    echo "    - inject_op: $inject_op"
    echo "    - use_utilization: $use_utilization"
    echo "    - evidence_source: $evidence_source"
    echo "    - lambda_orth: $lambda_orth"
    echo "    - lambda_ctr: $lambda_ctr"
    
    mkdir -p "$output_dir"
    
    # 设置环境变量
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    if torchrun --nnodes 1 --nproc_per_node $NPROC_PER_NODE --master-port $port \
        train/train_ministral.py \
        --model_name_or_path "$model_path" \
        --training_data_path "$DATA_JSON" \
        --training_image_dir "$IMG_DIR" \
        --output_dir "$output_dir" \
        --save_total_limit 2 \
        --report_to none \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LR \
        --num_train_epochs $NUM_EPOCHS \
        --bf16 true \
        --resume_from_checkpoint False \
        --save_strategy epoch \
        --logging_steps 2 \
        --remove_unused_columns False \
        --deepspeed "$DS_CONFIG" \
        --finetune_type full \
        --freeze_base_model true \
        --train_evidence_modules true \
        --enable_evidence $enable_evidence \
        --inject_position "$inject_position" \
        --inject_op "$inject_op" \
        --use_utilization $use_utilization \
        --evidence_source "$evidence_source" \
        --gate_layers all \
        --lambda_orth $lambda_orth \
        --lambda_ctr $lambda_ctr \
        --tau 0.07 \
        --aux_layers ""; then
        log "✅ 完成训练: $variant_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log "❌ 训练失败: $variant_name (继续执行下一个任务)"
        FAILED_TASKS+=("$variant_name")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ============================================================================
# 主执行流程
# ============================================================================

log "开始 LLM-HM 9个模型自动化训练流程"
echo "总计 9 个训练任务："
echo "  - Qwen3-VL-2B: +M, +M+A, Ours"
echo "  - Qwen3-VL-4B: +M, +M+A, Ours"
echo "  - Ministral-3-3B: +M, +M+A, Ours"
echo ""
echo "训练参数："
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Learning Rate: $LR"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Gradient Accumulation: $GRAD_ACCUM"
echo "  - GPUs: $CUDA_VISIBLE_DEVICES"

# 初始清理
cleanup_gpu

TASK_COUNT=0
TOTAL_TASKS=9

# ============================================================================
# Qwen3-VL-2B 训练
# ============================================================================

log "========== Qwen3-VL-2B 模型训练 =========="

# Qwen3-VL-2B +M
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen3-VL-2B +M (首层注入，仅 CE 损失)"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-2B/+M" \
    "Qwen3-VL-2B_+M" \
    "true" \
    "first_layer_input" \
    "add" \
    "false" \
    "candidate" \
    "0.0" \
    "0.0"
cleanup_gpu

# Qwen3-VL-2B +M+A
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen3-VL-2B +M+A (全层注入，仅 CE 损失)"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-2B/+M+A" \
    "Qwen3-VL-2B_+M+A" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0"
cleanup_gpu

# Qwen3-VL-2B Ours
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen3-VL-2B Ours (全层注入 + 正则化损失)"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-2B/Ours" \
    "Qwen3-VL-2B_Ours" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "1.0" \
    "1.0"
cleanup_gpu

# ============================================================================
# Qwen3-VL-4B 训练
# ============================================================================

log "========== Qwen3-VL-4B 模型训练 =========="

# Qwen3-VL-4B +M
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen3-VL-4B +M (首层注入，仅 CE 损失)"
train_qwen "$QWEN4B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-4B/+M" \
    "Qwen3-VL-4B_+M" \
    "true" \
    "first_layer_input" \
    "add" \
    "false" \
    "candidate" \
    "0.0" \
    "0.0"
cleanup_gpu

# Qwen3-VL-4B +M+A
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen3-VL-4B +M+A (全层注入，仅 CE 损失)"
train_qwen "$QWEN4B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-4B/+M+A" \
    "Qwen3-VL-4B_+M+A" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0"
cleanup_gpu

# Qwen3-VL-4B Ours
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen3-VL-4B Ours (全层注入 + 正则化损失)"
train_qwen "$QWEN4B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-4B/Ours" \
    "Qwen3-VL-4B_Ours" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "1.0" \
    "1.0"
cleanup_gpu

# ============================================================================
# Ministral-3-3B 训练
# ============================================================================

log "========== Ministral-3-3B 模型训练 =========="

# Ministral-3-3B +M
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ministral-3-3B +M (首层注入，仅 CE 损失)"
train_ministral "$MINISTRAL3B_BASE" \
    "$OUTPUT_ROOT/Ministral-3-3B/+M" \
    "Ministral-3-3B_+M" \
    "true" \
    "first_layer_input" \
    "add" \
    "false" \
    "candidate" \
    "0.0" \
    "0.0"
cleanup_gpu

# Ministral-3-3B +M+A
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ministral-3-3B +M+A (全层注入，仅 CE 损失)"
train_ministral "$MINISTRAL3B_BASE" \
    "$OUTPUT_ROOT/Ministral-3-3B/+M+A" \
    "Ministral-3-3B_+M+A" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0"
cleanup_gpu

# Ministral-3-3B Ours
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ministral-3-3B Ours (全层注入 + 正则化损失)"
train_ministral "$MINISTRAL3B_BASE" \
    "$OUTPUT_ROOT/Ministral-3-3B/Ours" \
    "Ministral-3-3B_Ours" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "1.0" \
    "1.0"
cleanup_gpu

# ============================================================================
# 训练总结
# ============================================================================

log "========== 训练流程完成 =========="
echo ""
echo "总任务数: $TOTAL_TASKS"
echo "成功: $SUCCESS_COUNT"
echo "失败: $FAIL_COUNT"
echo ""

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "失败的任务列表:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
    exit 1
else
    echo "🎉 所有训练任务成功完成！"
    echo ""
    echo "输出目录结构："
    echo "$OUTPUT_ROOT/"
    echo "├── Qwen3-VL-2B/"
    echo "│   ├── +M/"
    echo "│   ├── +M+A/"
    echo "│   └── Ours/"
    echo "├── Qwen3-VL-4B/"
    echo "│   ├── +M/"
    echo "│   ├── +M+A/"
    echo "│   └── Ours/"
    echo "└── Ministral-3-3B/"
    echo "    ├── +M/"
    echo "    ├── +M+A/"
    echo "    └── Ours/"
    echo ""
    exit 0
fi
