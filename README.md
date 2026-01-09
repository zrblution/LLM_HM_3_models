# LLM-HM Training

**Hallucination Mitigation (LLM-HM)** 训练工程，支持 Qwen 和 Ministral 视觉语言模型的微调。

## 目录结构

```
halltrain/
├── model/                    # 模型实现
│   ├── qwen_vl_model.py
│   └── ministral_vl_model.py
├── train/                    # 训练脚本
│   ├── train_qwen.py
│   └── train_ministral.py
|   |__ train_qwen_baseline.py
|   |__ train—_ministral_baseline.py
├── data/                     # 训练数据（需自行准备）
│   ├── coco_2017.json
│   └── data/                 # 图片目录
├── basemodel/                # 基础模型（需自行准备）
│   ├── Qwen3-VL-2B-Instruct/
│   ├── Qwen3-VL-4B-Instruct/
│   └── Ministral-3-3B-Instruct/
├── ds/
│   └── ds_z2_config.json     # DeepSpeed ZeRO-2 配置
└── run_finetune.sh           # 自动化训练脚本
```

## 环境配置

### 依赖版本

```
python==3.11.14
torch==2.4.1+cu121
transformers==5.0.0.dev0
datasets==4.4.2
deepspeed==0.15.1
accelerate==1.12.0
```

## 数据准备

1. **训练数据**：`data/coco_2017.json`
2. **图片目录**：`data/data/`

## 基础模型

从 Hugging Face 下载到 `basemodel/` 目录：

- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Ministral-3-3B-Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)（切记使用BF16版本）

## 训练配置

### 三种模型变体

| 变体 | 说明 | 关键参数 |
|------|------|----------|
| **+M** | 首层注入，仅 CE 损失 | `--inject_position first_layer_input --inject_op add`<br>`--lambda_orth 0.0 --lambda_ctr 0.0` |
| **+M+A** | 全层注入，仅 CE 损失 | `--inject_position per_layer --inject_op ours`<br>`--lambda_orth 0.0 --lambda_ctr 0.0` |
| **Ours** | 全层注入 + 正则化损失 | `--inject_position per_layer --inject_op ours`<br>`--lambda_orth 1.0 --lambda_ctr 1.0` |
**若需要训练对应版本的模型，则改动上述对应地方的参数即可**

### 训练示例

#### Qwen3-VL-2B Ours版本（4卡 DeepSpeed）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3（改为自己GPU数量） torchrun --nnodes 1 --nproc_per_node 4 --master-port 29500 \
  train/train_qwen_modified.py \
  --model_name_or_path  model_path\
  --training_data_path  image_path \
  --training_image_dir  data(json文本)_path \
  --output_dir output_path \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 3 \
  --bf16 true \
  --save_strategy no \
  --logging_steps 2 \
  --remove_unused_columns False \
  --deepspeed ./ds/ds_z2_config.json \
  --finetune_type full \
  --freeze_base_model true \
  --train_evidence_modules true \
  --enable_evidence true \
  --inject_position per_layer \
  --inject_op ours \
  --use_utilization true \
  --evidence_source aligned \
  --gate_layers all \
  --lambda_orth 1.0 \
  --lambda_ctr 1.0
  Optional:
  --aux_layers "l1,l2,..." 将正则限制到特定层数
```

#### Ministral-3-3B Ours版本（4卡 DeepSpeed）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3*改为自己GPU数量（ torchrun --nnodes 1 --nproc_per_node 4 --master-port 29500 \
  train/train_ministral_modified.py \
  --model_name_or_path  model_path\
  --training_data_path  image_path \
  --training_image_dir  data(json文本)_path \
  --output_dir output_path \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 3 \
  --bf16 true \
  --save_strategy no \
  --logging_steps 2 \
  --remove_unused_columns False \
  --deepspeed ./ds/ds_z2_config.json \
  --finetune_type full \
  --freeze_base_model true \
  --train_evidence_modules true \
  --enable_evidence true \
  --inject_position per_layer \
  --inject_op ours \
  --use_utilization true \
  --evidence_source aligned \
  --gate_layers all \
  --lambda_orth 1.0 \
  --lambda_ctr 1.0
  Optional:
  --aux_layers "l1,l2,..." 将正则限制到特定层数
```

## 自动化训练

### 一键训练所有模型

```bash
bash run_finetune_all.sh
```

该脚本自动执行：
- **任务**：3个基础模型 × 3个变体 = 9个训练任务

### 输出目录结构

```
/home/tos_data/LLM_HM_3_models/Fitntune_model_output_new/
├── Qwen3-VL-2B/
│   ├── +M/
│   ├── +M+A/
│   └── Ours/
├── Qwen3-VL-4B/
│   ├── +M/
│   ├── +M+A/
│   └── Ours/
├── Ministral-3-3B/
│   ├── +M/
│   ├── +M+A/
│   └── Ours/
```

## 关键参数说明

### Evidence 注入参数

- `--enable_evidence`: 是否启用 evidence 模块（`true`/`false`）
- `--inject_position`: 注入位置
  - `first_layer_input`: 仅首层
  - `per_layer`: 所有层
- `--inject_op`: 注入操作
  - `add`: 直接相加
  - `concat`: 拼接
  - `ours`: 利用率加权
- `--use_utilization`: 是否使用利用率加权（`true`/`false`）
- `--evidence_source`: Evidence 来源（`candidate`/`aligned`）

### 正则化损失参数

- `--lambda_orth`: 正交性损失权重（推荐 `1.0`）
- `--lambda_ctr`: 对比损失权重（推荐 `1.0`）
- `--tau`: 对比损失温度（推荐 `0.07`）
- `--aux_layers`: 应用正则化的层索引（空字符串表示所有层）

### 训练超参数

- `--learning_rate`: 学习率（推荐 `1.0e-5`）
- `--num_train_epochs`: 训练轮数（推荐 `3`）
- `--per_device_train_batch_size`: 每卡 batch size（必须为 `1`）
- `--gradient_accumulation_steps`: 梯度累积步数（增大以模拟更大 batch）

## 消融实验

### 4.1 注入位置和操作符

基于 Qwen3-VL-2B，固定 `--lambda_orth 0.0 --lambda_ctr 0.0`：

- **First-layer only**: `--inject_position first_layer_input --inject_op add`
- **All-layer Concat**: `--inject_position per_layer --inject_op concat`
- **All-layer Add**: `--inject_position per_layer --inject_op add`
- **All-layer Ours**: `--inject_position per_layer --inject_op ours`

### 4.3 损失函数组件

固定 `--inject_position per_layer --inject_op ours --use_utilization true`：

- **Full**: `--lambda_orth 1.0 --lambda_ctr 1.0`
- **w/o L_ctr**: `--lambda_orth 1.0 --lambda_ctr 0.0`
- **w/o L_orth**: `--lambda_orth 0.0 --lambda_ctr 1.0`
- **CE only**: `--lambda_orth 0.0 --lambda_ctr 0.0`

## 模型评测

### POPE 评测

训练完成后，使用 POPE 基准测试模型的幻觉缓解效果。

#### Qwen 模型评测

**COCO 数据集**
```bash
cd /home/tos_data/LLM_HM_3_models/halleval_qwen && \
source /root/miniconda3/etc/profile.d/conda.sh && \
conda activate Hall && \
python /home/tos_data/LLM_HM_3_models/halleval_qwen/POPE/run_eval.py \
  --dataset coco \
  --model_dir path_model \
  --model_name Qwen3-VL-2B-Ours \
  --batch_size 64 \
  --multi_gpu \
  --gpus 0,1
```

**GQA 数据集**
```bash
cd /home/tos_data/LLM_HM_3_models/halleval_qwen && \
source /root/miniconda3/etc/profile.d/conda.sh && \
conda activate Hall && \
python /home/tos_data/LLM_HM_3_models/halleval_qwen/POPE/run_eval.py \
  --dataset gqa \
  --model_dir path_model \
  --model_name Qwen3-VL-2B-Ours \
  --batch_size 64 \
  --multi_gpu \
  --gpus 0,1
```

#### Ministral 模型评测

**COCO 数据集**
```bash
cd /home/tos_data/LLM_HM_3_models/halleval_ministral && \
source /root/miniconda3/etc/profile.d/conda.sh && \
conda activate Hall && \
python /home/tos_data/LLM_HM_3_models/halleval_ministral/POPE/run_eval.py \
  --dataset coco \
  --model_dir path_model \
  --model_name Ministral-3-3B-Ours \
  --batch_size 64 \
  --multi_gpu \
  --gpus 0,1
```

**GQA 数据集**
```bash
cd /home/tos_data/LLM_HM_3_models/halleval_ministral && \
source /root/miniconda3/etc/profile.d/conda.sh && \
conda activate Hall && \
python /home/tos_data/LLM_HM_3_models/halleval_ministral/POPE/run_eval.py \
  --dataset gqa \
  --model_dir path_model \
  --model_name Ministral-3-3B-Ours \
  --batch_size 64 \
  --multi_gpu \
  --gpus 0,1
```

**评测参数说明**：
- `--dataset`: 评测数据集（`coco` / `gqa`）
- `--model_dir`: 微调后的模型路径
- `--model_name`: 模型名称（用于结果保存）
- `--batch_size`: 推理批次大小
- `--multi_gpu`: 启用多卡推理
- `--gpus`: 指定使用的 GPU（如 `0,1`）

### CHAIR 评测

CHAIR (Caption Hallucination Assessment with Image Relevance) 评测用于评估模型在图像描述任务中的幻觉程度。

#### Qwen 模型评测

```bash
cd /home/tos_data/LLM_HM_3_models/halleval_qwen && \
source /root/miniconda3/etc/profile.d/conda.sh && \
conda activate Hall && \
python /home/tos_data/LLM_HM_3_models/halleval_qwen/CHAIR/run_eval.py \
  --model_dir path_model \
  --model_name Qwen3-VL-2B-Ours \
  --image_dir /home/tos_data/LLM_HM_3_models/halleval_qwen/CHAIR/val2014_1000 \
  --annotation_path /home/tos_data/LLM_HM_3_models/halleval_qwen/CHAIR/annotations_1000 \
  --synonyms_file /home/tos_data/LLM_HM_3_models/halleval_qwen/CHAIR/synonyms.txt \
  --result_root /home/tos_data/LLM_HM_3_models/halleval_qwen/CHAIR/result \
  --num_samples 1000 \
  --batch_size 64 \
  --multi_gpu \
  --gpus 0,1 \
  --prompt "Please describe this image in detail."
```

#### Ministral 模型评测

```bash
cd /home/tos_data/LLM_HM_3_models/halleval_ministral && \
source /root/miniconda3/etc/profile.d/conda.sh && \
conda activate Hall && \
python /home/tos_data/LLM_HM_3_models/halleval_ministral/CHAIR/run_eval.py \
  --model_dir path_model \
  --model_name Ministral-3-3B-Ours \
  --image_dir /home/tos_data/LLM_HM_3_models/halleval_ministral/CHAIR/val2014_1000 \
  --annotation_path /home/tos_data/LLM_HM_3_models/halleval_ministral/CHAIR/annotations_1000 \
  --synonyms_file /home/tos_data/LLM_HM_3_models/halleval_ministral/CHAIR/synonyms.txt \
  --result_root /home/tos_data/LLM_HM_3_models/halleval_ministral/CHAIR/result \
  --num_samples 1000 \
  --batch_size 64 \
  --multi_gpu \
  --gpus 0,1 \
  --prompt "Please describe this image in detail."
```

**评测参数说明**：
- `--model_dir`: 微调后的模型路径
- `--model_name`: 模型名称（用于结果保存）
- `--image_dir`: 评测图像目录（COCO val2014 的 1000 张采样图像）
- `--annotation_path`: COCO 标注文件目录
- `--synonyms_file`: 同义词文件路径
- `--result_root`: 结果保存根目录
- `--num_samples`: 评测样本数量（默认 1000）
- `--batch_size`: 推理批次大小
- `--multi_gpu`: 启用多卡推理
- `--gpus`: 指定使用的 GPU（如 `0,1`）
- `--prompt`: 图像描述提示词

## 注意事项

1. **Batch Size 限制**：由于数据处理方式，`--per_device_train_batch_size` 必须为 `1`，使用 `--gradient_accumulation_steps` 增大有效 batch size
2. **GPU 要求**：推荐 4×A100 (40GB)，使用 DeepSpeed ZeRO-2
3. **端口冲突**：多任务并行时需修改 `--master-port`
4. **数据路径**：确保 `--training_data_path` 和 `--training_image_dir` 正确指向数据目录
5. **评测环境**：POPE 和 CHAIR 评测需要激活 `Hall` conda 环境
