#!/bin/bash
################################################################################
# run_finetune.sh - LLM-HM 自动化微调脚本
################################################################################
#
# 【功能说明】
# 本脚本用于自动化执行 LLM-HM 项目的所有微调任务，包括：
#   - 任务一：基础模型微调 (Qwen2B, Qwen4B, Ministral3B 的 +M, +M+A, Ours 变体)
#   - 任务二：消融实验 4.1 (Injection Position/Operator)
#   - 任务三：消融实验 4.3 (Loss Components)
#
# 【硬件要求】
#   - 4张 A100 GPU (CUDA_VISIBLE_DEVICES=0,1,2,3)
#   - 串行执行，每个任务完成后自动开始下一个
#
# 【参数配置参考】
# 根据 https://github.com/yuanheTian/LLM-HM README:
#
# +M (First-layer Injection, CE only):
#   --enable_evidence true --inject_position first_layer_input --inject_op add
#   --use_utilization false --evidence_source candidate
#   --lambda_orth 0.0 --lambda_ctr 0.0
#
# +M+A (All-layer Injection, CE only):
#   --enable_evidence true --inject_position per_layer --inject_op ours
#   --use_utilization true --evidence_source aligned
#   --lambda_orth 0.0 --lambda_ctr 0.0
#
# Ours (All-layer Injection + Evidence Regularization):
#   --enable_evidence true --inject_position per_layer --inject_op ours
#   --use_utilization true --evidence_source aligned
#   --lambda_orth 1.0 --lambda_ctr 1.0
#
# 【消融实验 4.1 - Injection Position and Operator】
#   First-layer only:    --inject_position first_layer_input --inject_op add --use_utilization false
#   All-layer Concat:    --inject_position per_layer --inject_op concat
#   All-layer Add:       --inject_position per_layer --inject_op add
#   All-layer Ours:      --inject_position per_layer --inject_op ours
#   (所有都用 --lambda_orth 0.0 --lambda_ctr 0.0)
#
# 【消融实验 4.3 - Loss Components】
#   固定: --inject_position per_layer --inject_op ours --use_utilization true
#   Full:       --lambda_orth 1.0 --lambda_ctr 1.0
#   w/o L_ctr:  --lambda_orth 1.0 --lambda_ctr 0.0
#   w/o L_orth: --lambda_orth 0.0 --lambda_ctr 1.0
#   CE only:    --lambda_orth 0.0 --lambda_ctr 0.0
#
################################################################################

# 注意：不使用 set -e，以便单个任务失败时继续执行后续任务
# set -e  # 遇到错误立即退出

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
    local custom_epochs="${11}"  # 可选的自定义epoch参数
    
    # 如果提供了自定义epoch，使用它；否则使用全局NUM_EPOCHS
    local epochs=${custom_epochs:-$NUM_EPOCHS}
    
    local port=$(get_next_port)
    
    log "开始训练 Qwen 模型: $variant_name"
    echo "  模型路径: $model_path"
    echo "  输出目录: $output_dir"
    echo "  端口: $port"
    echo "  参数: enable_evidence=$enable_evidence, inject_position=$inject_position"
    echo "        inject_op=$inject_op, use_utilization=$use_utilization"
    echo "        evidence_source=$evidence_source"
    echo "        lambda_orth=$lambda_orth, lambda_ctr=$lambda_ctr"
    
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
        --enable_evidence $enable_evidence \
        --inject_position "$inject_position" \
        --inject_op "$inject_op" \
        --use_utilization $use_utilization \
        --evidence_source "$evidence_source" \
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
    echo "  参数: enable_evidence=$enable_evidence, inject_position=$inject_position"
    echo "        inject_op=$inject_op, use_utilization=$use_utilization"
    echo "        evidence_source=$evidence_source"
    echo "        lambda_orth=$lambda_orth, lambda_ctr=$lambda_ctr"
    
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
        --enable_evidence $enable_evidence \
        --inject_position "$inject_position" \
        --inject_op "$inject_op" \
        --use_utilization $use_utilization \
        --evidence_source "$evidence_source" \
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

log "开始 LLM-HM 自动化微调流程"
echo "总计 17 个训练任务"
echo "  - 任务一：基础模型微调 (9个: Qwen2B×3 + Qwen4B×3 + Ministral3B×3)"
echo "  - 任务二：消融实验 4.1 (4个)"
echo "  - 任务三：消融实验 4.3 (4个)"

# 初始清理
cleanup_gpu

TASK_COUNT=0
TOTAL_TASKS=17

# ============================================================================
# 任务一：基础模型微调
# ============================================================================

log "========== 任务一：基础模型微调 =========="

# --------------------------------------------------------------------------
# 第一阶段：先训练所有模型的 Ours 版本
# --------------------------------------------------------------------------

log "【第一阶段】训练所有模型的 Ours 版本"

# Qwen2B Ours
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen2B Ours"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-2B/Ours" \
    "Qwen2B_Ours" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "1.0" \
    "1.0"
cleanup_gpu

# Qwen4B Ours
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen4B Ours"
train_qwen "$QWEN4B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-4B/Ours" \
    "Qwen4B_Ours" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "1.0" \
    "1.0"
cleanup_gpu

# Ministral3B Ours
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ministral3B Ours"
train_ministral "$MINISTRAL3B_BASE" \
    "$OUTPUT_ROOT/Ministral-3-3B/Ours" \
    "Ministral3B_Ours" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "1.0" \
    "1.0"
cleanup_gpu

# --------------------------------------------------------------------------
# 第二阶段：依次训练每个模型的 +M 和 +M+A 版本
# --------------------------------------------------------------------------

log "【第二阶段】训练每个模型的 +M 和 +M+A 版本"

# Qwen2B: +M, +M+A
log "--- Qwen2B 变体 ---"

# Qwen2B +M
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen2B +M"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-2B/+M" \
    "Qwen2B_+M" \
    "true" \
    "first_layer_input" \
    "add" \
    "false" \
    "candidate" \
    "0.0" \
    "0.0"
cleanup_gpu

# Qwen2B +M+A
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen2B +M+A"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-2B/+M+A" \
    "Qwen2B_+M+A" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0"
cleanup_gpu

# Qwen4B: Ours, +M, +M+A (所有使用 epoch=1)
log "--- Qwen4B 变体 (epoch=1) ---"

# Qwen4B Ours (重新训练, epoch=1)
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen4B Ours (重新训练, epoch=1)"
train_qwen "$QWEN4B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-4B/Ours" \
    "Qwen4B_Ours" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "1.0" \
    "1.0" \
    "1"
cleanup_gpu

# Qwen4B +M (epoch=1)
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen4B +M (epoch=1)"
train_qwen "$QWEN4B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-4B/+M" \
    "Qwen4B_+M" \
    "true" \
    "first_layer_input" \
    "add" \
    "false" \
    "candidate" \
    "0.0" \
    "0.0" \
    "1"
cleanup_gpu

# Qwen4B +M+A (epoch=1)
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Qwen4B +M+A (epoch=1)"
train_qwen "$QWEN4B_BASE" \
    "$OUTPUT_ROOT/Qwen3-VL-4B/+M+A" \
    "Qwen4B_+M+A" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0" \
    "1"
cleanup_gpu

# Ministral3B: +M, +M+A
log "--- Ministral3B 变体 ---"

# Ministral3B +M
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ministral3B +M"
train_ministral "$MINISTRAL3B_BASE" \
    "$OUTPUT_ROOT/Ministral-3-3B/+M" \
    "Ministral3B_+M" \
    "true" \
    "first_layer_input" \
    "add" \
    "false" \
    "candidate" \
    "0.0" \
    "0.0"
cleanup_gpu

# Ministral3B +M+A
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ministral3B +M+A"
train_ministral "$MINISTRAL3B_BASE" \
    "$OUTPUT_ROOT/Ministral-3-3B/+M+A" \
    "Ministral3B_+M+A" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0"
cleanup_gpu

# ============================================================================
# 任务二：消融实验 4.1 - Injection Position and Operator
# ============================================================================

log "========== 任务二：消融实验 4.1 (Injection Position/Operator) =========="
log "所有消融实验基于 Qwen2B 基础模型"

# Ablation 4.1 - First-layer only
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ablation 4.1: First-layer only"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Ablation_4.1/First-layer-only" \
    "Ablation4.1_First-layer-only" \
    "true" \
    "first_layer_input" \
    "add" \
    "false" \
    "candidate" \
    "0.0" \
    "0.0"
cleanup_gpu

# Ablation 4.1 - All-layer Concat
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ablation 4.1: All-layer Concat"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Ablation_4.1/All-layer-Concat" \
    "Ablation4.1_All-layer-Concat" \
    "true" \
    "per_layer" \
    "concat" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0"
cleanup_gpu

# Ablation 4.1 - All-layer Add
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ablation 4.1: All-layer Add"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Ablation_4.1/All-layer-Add" \
    "Ablation4.1_All-layer-Add" \
    "true" \
    "per_layer" \
    "add" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0"
cleanup_gpu

# Ablation 4.1 - All-layer Ours-style
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ablation 4.1: All-layer Ours-style"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Ablation_4.1/All-layer-Ours" \
    "Ablation4.1_All-layer-Ours" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0"
cleanup_gpu

# ============================================================================
# 任务三：消融实验 4.3 - Loss Components
# ============================================================================

log "========== 任务三：消融实验 4.3 (Loss Components) =========="
log "结构固定为 All-layer Ours-style，仅改变损失函数参数"

# Ablation 4.3 - Full (lambda_orth=1, lambda_ctr=1)
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ablation 4.3: Full"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Ablation_4.3/Full" \
    "Ablation4.3_Full" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "1.0" \
    "1.0"
cleanup_gpu

# Ablation 4.3 - w/o L_ctr (lambda_orth=1, lambda_ctr=0)
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ablation 4.3: w/o L_ctr"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Ablation_4.3/wo_L_ctr" \
    "Ablation4.3_wo_L_ctr" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "1.0" \
    "0.0"
cleanup_gpu

# Ablation 4.3 - w/o L_orth (lambda_orth=0, lambda_ctr=1)
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ablation 4.3: w/o L_orth"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Ablation_4.3/wo_L_orth" \
    "Ablation4.3_wo_L_orth" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "0.0" \
    "1.0"
cleanup_gpu

# Ablation 4.3 - CE only (lambda_orth=0, lambda_ctr=0)
TASK_COUNT=$((TASK_COUNT + 1))
log "[$TASK_COUNT/$TOTAL_TASKS] Ablation 4.3: CE only"
train_qwen "$QWEN2B_BASE" \
    "$OUTPUT_ROOT/Ablation_4.3/CE_only" \
    "Ablation4.3_CE_only" \
    "true" \
    "per_layer" \
    "ours" \
    "true" \
    "aligned" \
    "0.0" \
    "0.0"
cleanup_gpu

# ============================================================================
# 完成
# ============================================================================

log "========== 所有训练任务完成！=========="
echo ""
echo "========================================"
echo "           训练结果统计"
echo "========================================"
echo "  ✅ 成功: $SUCCESS_COUNT 个任务"
echo "  ❌ 失败: $FAIL_COUNT 个任务"
echo ""

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "失败的任务列表："
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
fi

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
echo "├── Ministral-3-3B/"
echo "│   ├── +M/"
echo "│   ├── +M+A/"
echo "│   └── Ours/"
echo "├── Ablation_4.1/"
echo "│   ├── First-layer-only/"
echo "│   ├── All-layer-Concat/"
echo "│   ├── All-layer-Add/"
echo "│   └── All-layer-Ours/"
echo "└── Ablation_4.3/"
echo "    ├── Full/"
echo "    ├── wo_L_ctr/"
echo "    ├── wo_L_orth/"
echo "    └── CE_only/"
echo ""
log "训练流程结束于 $(date '+%Y-%m-%d %H:%M:%S')"

# 如果有失败的任务，返回非零退出码（但不影响脚本执行）
if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "⚠️  有 $FAIL_COUNT 个任务失败，请检查日志"
    exit 1
fi
