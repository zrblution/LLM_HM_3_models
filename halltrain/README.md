# halltrain (LLM_HM_3_model)

本目录为 **Hallucination Mitigation (LLM-HM)** 的训练工程统一版：同一套工程结构下提供两套训练入口（Qwen / Ministral）与两份模型实现。

> 注意：本仓库面向上传 GitHub，默认不包含大模型权重与大规模图片数据；请按下方 **Data** 部分自行准备 `data/` 与 `basemodel/`。

## Directory Layout

```
halltrain/
  model/
    qwen_vl_model.py
    ministral_vl_model.py
  train/
    train_qwen.py
    train_ministral.py
  data/
    train.json
    img/
  ds/
    ds_z2_config.json
  run.sh
```

## Environment

### Hall 环境依赖版本（当前服务器）

以下版本来自你的 conda 环境 `Hall`（`conda list -n Hall`）：

```
python==3.11.14
torch==2.4.1+cu121
transformers==5.0.0.dev0
datasets==4.4.2
deepspeed==0.15.1
accelerate==1.12.0
numpy==2.3.5
pillow==12.0.0
tokenizers==0.22.1
safetensors==0.7.0
```

### 安装建议（参考）

推荐使用 Python 3.11，关键依赖（供参考）：

- `torch` / `pytorch`
- `transformers`
- `datasets`
- `deepspeed`（多卡或 ZeRO 训练时）
- `Pillow`

示例（conda）：

```bash
conda create -n llm_hm python=3.11 -y
conda activate llm_hm
pip install -U pip
pip install torch transformers datasets deepspeed pillow
```

## Data

训练脚本通过参数读取数据路径：

- 标注：`data/train.json`
- 图片目录：`data/img/`



## Base Model Weights

训练脚本默认 `--model_name_or_path ./basemodel`。由于 `basemodel/` 不进入 GitHub，请自行准备三个基础模型（可直接用 HF model id，或提前下载为本地目录）：

- https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

建议本地落盘结构（推荐，避免每次重复下载）：

```
halltrain/basemodel/
  Ministral-3-3B-Instruct-2512/
  Qwen3-VL-2B-Instruct/
  Qwen3-VL-4B-Instruct/
```

- 方式 A：将底座模型权重放到 `halltrain/basemodel/`
- 方式 B：运行时显式传入 `--model_name_or_path <HF model id 或本地路径>`

## Training

### 9 个训练命令（3 个基础模型 × 3 个版本）

输出根目录固定为：

- `/home/tos_data/LLM_HM_3_model/Fintune_model_output`

每个基础模型都需要训练 3 个版本（`+M` / `+M+A` / `Ours`），共 9 个命令。下面以 **单卡** 为例（多卡时将 `python` 换成 `torchrun ...`，并追加 `--deepspeed ds/ds_z2_config.json` 即可）。

一键串行训练全部 9 个实验：

```bash
bash ./train_all_models.sh
```

#### A) Qwen3-VL-2B-Instruct（`train/train_qwen.py`）

`+M`

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_qwen.py \
  --model_name_or_path ./basemodel/Qwen3-VL-2B-Instruct \
  --training_data_path ./data/train.json \
  --training_image_dir ./data/img \
  --output_dir /home/tos_data/LLM_HM_3_model/Fintune_model_output/output_model_Qwen3-VL-2B_+M \
  --save_total_limit 2 --report_to none --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 --num_train_epochs 3 --bf16 true --resume_from_checkpoint False \
  --save_strategy epoch --logging_steps 2 --remove_unused_columns False \
  --enable_evidence true --inject_position first_layer_input --inject_op add \
  --use_utilization false --evidence_source candidate \
  --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""
```

`+M+A`

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_qwen.py \
  --model_name_or_path ./basemodel/Qwen3-VL-2B-Instruct \
  --training_data_path ./data/train.json \
  --training_image_dir ./data/img \
  --output_dir /home/tos_data/LLM_HM_3_model/Fintune_model_output/output_model_Qwen3-VL-2B_+M+A \
  --save_total_limit 2 --report_to none --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 --num_train_epochs 3 --bf16 true --resume_from_checkpoint False \
  --save_strategy epoch --logging_steps 2 --remove_unused_columns False \
  --enable_evidence true --inject_position per_layer --inject_op ours \
  --use_utilization true --evidence_source aligned \
  --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""
```

`Ours`

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_qwen.py \
  --model_name_or_path ./basemodel/Qwen3-VL-2B-Instruct \
  --training_data_path ./data/train.json \
  --training_image_dir ./data/img \
  --output_dir /home/tos_data/LLM_HM_3_model/Fintune_model_output/output_model_Qwen3-VL-2B_Ours \
  --save_total_limit 2 --report_to none --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 --num_train_epochs 3 --bf16 true --resume_from_checkpoint False \
  --save_strategy epoch --logging_steps 2 --remove_unused_columns False \
  --enable_evidence true --inject_position per_layer --inject_op ours \
  --use_utilization true --evidence_source aligned \
  --lambda_orth 1.0 --lambda_ctr 1.0 --tau 0.07 --aux_layers ""
```

#### B) Qwen3-VL-4B-Instruct（`train/train_qwen.py`）

`+M`

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_qwen.py \
  --model_name_or_path ./basemodel/Qwen3-VL-4B-Instruct \
  --training_data_path ./data/train.json \
  --training_image_dir ./data/img \
  --output_dir /home/tos_data/LLM_HM_3_model/Fintune_model_output/output_model_Qwen3-VL-4B_+M \
  --save_total_limit 2 --report_to none --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 --num_train_epochs 3 --bf16 true --resume_from_checkpoint False \
  --save_strategy epoch --logging_steps 2 --remove_unused_columns False \
  --enable_evidence true --inject_position first_layer_input --inject_op add \
  --use_utilization false --evidence_source candidate \
  --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""
```

`+M+A`

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_qwen.py \
  --model_name_or_path ./basemodel/Qwen3-VL-4B-Instruct \
  --training_data_path ./data/train.json \
  --training_image_dir ./data/img \
  --output_dir /home/tos_data/LLM_HM_3_model/Fintune_model_output/output_model_Qwen3-VL-4B_+M+A \
  --save_total_limit 2 --report_to none --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 --num_train_epochs 3 --bf16 true --resume_from_checkpoint False \
  --save_strategy epoch --logging_steps 2 --remove_unused_columns False \
  --enable_evidence true --inject_position per_layer --inject_op ours \
  --use_utilization true --evidence_source aligned \
  --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""
```

`Ours`

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_qwen.py \
  --model_name_or_path ./basemodel/Qwen3-VL-4B-Instruct \
  --training_data_path ./data/train.json \
  --training_image_dir ./data/img \
  --output_dir /home/tos_data/LLM_HM_3_model/Fintune_model_output/output_model_Qwen3-VL-4B_Ours \
  --save_total_limit 2 --report_to none --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 --num_train_epochs 3 --bf16 true --resume_from_checkpoint False \
  --save_strategy epoch --logging_steps 2 --remove_unused_columns False \
  --enable_evidence true --inject_position per_layer --inject_op ours \
  --use_utilization true --evidence_source aligned \
  --lambda_orth 1.0 --lambda_ctr 1.0 --tau 0.07 --aux_layers ""
```

#### C) Ministral-3-3B-Instruct-2512（`train/train_ministral.py`）

`+M`

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_ministral.py \
  --model_name_or_path ./basemodel/Ministral-3-3B-Instruct-2512 \
  --training_data_path ./data/train.json \
  --training_image_dir ./data/img \
  --output_dir /home/tos_data/LLM_HM_3_model/Fintune_model_output/output_model_Ministral-3-3B_+M \
  --save_total_limit 2 --report_to none --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 --num_train_epochs 3 --bf16 true --resume_from_checkpoint False \
  --save_strategy epoch --logging_steps 2 --remove_unused_columns False \
  --enable_evidence true --inject_position first_layer_input --inject_op add \
  --use_utilization false --evidence_source candidate \
  --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""
```

`+M+A`

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_ministral.py \
  --model_name_or_path ./basemodel/Ministral-3-3B-Instruct-2512 \
  --training_data_path ./data/train.json \
  --training_image_dir ./data/img \
  --output_dir /home/tos_data/LLM_HM_3_model/Fintune_model_output/output_model_Ministral-3-3B_+M+A \
  --save_total_limit 2 --report_to none --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 --num_train_epochs 3 --bf16 true --resume_from_checkpoint False \
  --save_strategy epoch --logging_steps 2 --remove_unused_columns False \
  --enable_evidence true --inject_position per_layer --inject_op ours \
  --use_utilization true --evidence_source aligned \
  --lambda_orth 0.0 --lambda_ctr 0.0 --tau 0.07 --aux_layers ""
```

`Ours`

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_ministral.py \
  --model_name_or_path ./basemodel/Ministral-3-3B-Instruct-2512 \
  --training_data_path ./data/train.json \
  --training_image_dir ./data/img \
  --output_dir /home/tos_data/LLM_HM_3_model/Fintune_model_output/output_model_Ministral-3-3B_Ours \
  --save_total_limit 2 --report_to none --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 --num_train_epochs 3 --bf16 true --resume_from_checkpoint False \
  --save_strategy epoch --logging_steps 2 --remove_unused_columns False \
  --enable_evidence true --inject_position per_layer --inject_op ours \
  --use_utilization true --evidence_source aligned \
  --lambda_orth 1.0 --lambda_ctr 1.0 --tau 0.07 --aux_layers ""
```

## Evidence-regularization arguments

The training script supports optional evidence-regularization terms used by the custom Trainer:

- `--lambda_orth`: weight for the orthogonality regularizer L_orth.
- `--lambda_ctr`: weight for the contrastive regularizer L_ctr.
- `--tau`: temperature for the contrastive loss.
- `--aux_layers`: comma-separated decoder layer indices to apply the regularizers to (e.g., "0,1,2"). Use an empty string to apply to all layers that emit aux tensors.

To enable regularization, set `--lambda_orth` and/or `--lambda_ctr` to a positive value.

## Experiment Configurations

This section describes how to obtain different experimental settings **only by changing command-line arguments**. All other training options remain the same as in the examples above.

## Output Directory Convention

你的微调模型输出根目录：

- `/home/tos_data/LLM_HM_3_model/Fintune_model_output`

输出子目录命名参考 `/home/tos_data/code/halltrain/train_all_models.sh`。

- `output_model_Qwen3-VL-2B_+M`
- `output_model_Qwen3-VL-2B_+M+A`
- `output_model_Qwen3-VL-2B_Ours`
- `output_model_Qwen3-VL-4B_+M`
- `output_model_Qwen3-VL-4B_+M+A`
- `output_model_Qwen3-VL-4B_Ours`
- `output_model_Ministral-3-3B_+M`
- `output_model_Ministral-3-3B_+M+A`
- `output_model_Ministral-3-3B_Ours`

### 1. Base (Table 2)

Standard fine-tuning without evidence modules.

Set:
- `--enable_evidence false`
- `--inject_position none`
- `--lambda_orth 0.0`
- `--lambda_ctr 0.0`

---

### 2. +M (First-layer Injection, CE only)

Use visual memory retrieval, but inject evidence **only once at the first layer input**, without utilization weighting.

Set:
- `--enable_evidence true`
- `--inject_position first_layer_input`
- `--inject_op add`
- `--use_utilization false`
- `--evidence_source candidate`
- `--lambda_orth 0.0`
- `--lambda_ctr 0.0`

---

### 3. +M+A (All-layer Injection, CE only)

Inject evidence at **all decoder layers** using utilization-weighted correction, trained with cross-entropy only.

Set:
- `--enable_evidence true`
- `--inject_position per_layer`
- `--inject_op ours`
- `--use_utilization true`
- `--evidence_source aligned`
- `--lambda_orth 0.0`
- `--lambda_ctr 0.0`

---

### 4. Ours (All-layer Injection + Evidence Regularization)

Full model with all-layer utilization-weighted injection and additional evidence regularization losses.

Set:
- `--enable_evidence true`
- `--inject_position per_layer`
- `--inject_op ours`
- `--use_utilization true`
- `--evidence_source aligned`
- `--lambda_orth > 0` (e.g., `1.0`)
- `--lambda_ctr > 0` (e.g., `1.0`)

Optional:
- `--aux_layers "l1,l2,..."` to restrict regularization to specific layers

---

### 5. Ablation 4.1: Injection Position and Operator

Run on a single Qwen3-VL model size.

**First-layer only**
- `--inject_position first_layer_input`
- `--inject_op add`
- `--use_utilization false`

**All-layer variants**
- Concat: `--inject_position per_layer --inject_op concat`
- Add: `--inject_position per_layer --inject_op add`
- Ours-style: `--inject_position per_layer --inject_op ours`

All above use:
- `--lambda_orth 0.0`
- `--lambda_ctr 0.0`

---

### 6. Ablation 4.3: Loss Components

Keep structure fixed to all-layer Ours-style injection:
- `--inject_position per_layer`
- `--inject_op ours`
- `--use_utilization true`

Vary losses:
- Full: `--lambda_orth 1`, `--lambda_ctr 1`
- w/o L_ctr: `--lambda_orth 1`, `--lambda_ctr 0.0`
- w/o L_orth: `--lambda_orth 0.0`, `--lambda_ctr 1`
- CE only: `--lambda_orth 0.0`, `--lambda_ctr 0.0`

---

## Known Issues

1. Due to the current on-the-fly image processing in the data collator, `--per_device_train_batch_size` must be set to `1`. To simulate a larger effective batch size, increase `--gradient_accumulation_steps` instead.

2. 若出现 `FileNotFoundError`（找不到图片或 json），优先检查 `--training_data_path` 与 `--training_image_dir` 是否指向本工程 `data/`。
