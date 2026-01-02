# LLM_HM_3_models

本目录包含 **LLM-HM (Hallucination Mitigation)** 的训练与测评代码：

- 训练：`halltrain/`（支持 Qwen3-VL 与 Ministral 两套入口）
- 测评：`halleval_qwen/`、`halleval_ministral/`（CHAIR / POPE）

---

## 1) 依赖 / 环境

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

快速安装：

```bash
conda create -n llm_hm python=3.11 -y
conda activate llm_hm
pip install -U pip
pip install torch transformers datasets accelerate deepspeed pillow safetensors
```

---

## 2) 数据与权重路径

### 2.1 训练数据（`halltrain/`）

- 标注：`halltrain/data/train.json`
- 图片：`halltrain/data/img/`

> 若报 `FileNotFoundError`，优先检查 `--training_data_path` / `--training_image_dir` 是否指向上述路径。

### 2.2 底座模型（`halltrain/basemodel/`）

训练参数默认 `--model_name_or_path ./basemodel/...`，可用本地目录或直接用 HF model id。

推荐本地落盘结构：

```
halltrain/basemodel/
  Ministral-3-3B-Instruct-2512/
  Qwen3-VL-2B-Instruct/
  Qwen3-VL-4B-Instruct/
```

对应参考：

- `mistralai/Ministral-3-3B-Instruct-2512`
- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`

### 2.3 测评数据（`halleval_*/`）

- CHAIR：内置小样本（1000 张）
  - 图片：`CHAIR/val2014_1000/`
  - 标注：`CHAIR/annotations_1000/`
- POPE：内置 COCO/GQA（每个 split：random/popular/adversarial）
  - `POPE/coco/*` 与 `POPE/gqa/*`（输入 json 在 `*/output/`，图片在 `*/{random,popular,adversarial}/`）

---

## 3) 训练（`halltrain/`）

### 3.1 一键训练（9 个实验：3 底座 × 3 版本）

```bash
cd /home/tos_data/LLM_HM_3_models/halltrain
bash ./train_all_models.sh
```

常用可选环境变量（不改脚本即可切换）：

- `OUTPUT_ROOT`：输出根目录（默认：`/home/tos_data/LLM_HM_3_model/Fintune_model_output`）
- `DATA_JSON` / `IMG_DIR`：训练数据路径
- `NPROC_PER_NODE`：多卡进程数（>1 时自动用 `torchrun`）
- `ENABLE_DEEPSPEED=1`：追加 `--deepspeed ds/ds_z2_config.json`

> 说明：`OUTPUT_ROOT` 的默认值来自历史路径，建议按你的机器/目录显式设置。

### 3.2 三种版本（只改参数）

- `+M`：`--enable_evidence true --inject_position first_layer_input --inject_op add --use_utilization false --evidence_source candidate`
- `+M+A`：`--enable_evidence true --inject_position per_layer --inject_op ours --use_utilization true --evidence_source aligned`
- `Ours`：在 `+M+A` 基础上再加 `--lambda_orth >0 --lambda_ctr >0`

注意：当前数据整理/图像处理逻辑下，`--per_device_train_batch_size` 通常需要为 `1`；想增大等效 batch 请增大 `--gradient_accumulation_steps`。

---

## 4) 测评（`halleval_qwen/` / `halleval_ministral/`）

> 下面命令默认从对应 `halleval_*` 目录执行；`--model_dir` 指向你的微调模型目录（如某个 `checkpoint-*` 或导出的模型目录）。

### 4.1 CHAIR（caption → CHAIRs/CHAIRi）

```bash
cd /home/tos_data/LLM_HM_3_models/halleval_qwen   # 或 halleval_ministral
python CHAIR/run_eval.py \
  --model_dir /path/to/model \
  --annotation_path CHAIR/annotations_1000 \
  --image_dir CHAIR/val2014_1000 \
  --num_samples 1000 \
  --device cuda \
  --model_type auto
```

可选（不加则不启用）：`--use_vcd` / `--use_inter`

输出：

- captions：`CHAIR/result/<模型目录名>/generated_captions.json`
- 结果：`CHAIR/result/<模型目录名>/chair_results.json`

### 4.2 POPE（Yes/No QA → Acc/Prec/Rec/F1）

POPE 标准流程：先用模型跑推理生成预测，再和 GT 对齐算指标。

示例（以 COCO 为例；GQA 将 `coco` 改为 `gqa`）：

```bash
cd /home/tos_data/LLM_HM_3_models/halleval_ministral/POPE   # 或 halleval_qwen/POPE
MODEL_DIR=/path/to/model
DATASET=coco  # or gqa
MODEL_NAME="$(basename "$MODEL_DIR")"

mkdir -p "result/$MODEL_NAME/$DATASET/eval"
for v in random popular adversarial; do
  python convert_output.py \
    --input_json "$DATASET/output/${DATASET}_pope_${v}.json" \
    --model_dir "$MODEL_DIR" \
    --image_root "$DATASET/$v" \
    --output_json "result/$MODEL_NAME/$DATASET/${v}.json"
  python pope.py \
    --gt_file "$DATASET/output/${DATASET}_pope_${v}.json" \
    --pred_file "result/$MODEL_NAME/$DATASET/${v}.json" \
    --output_json "result/$MODEL_NAME/$DATASET/eval/${v}_results.json"
done
```

可选（不加则不启用）：在 `convert_output.py` 后追加 `--use_vcd` / `--use_inter`。

---

## 5) 目录速览

```
LLM_HM_3_models/
  halltrain/            # 训练（Qwen / Ministral）
  halleval_qwen/        # 测评（CHAIR/POPE）
  halleval_ministral/   # 测评（CHAIR/POPE）
```
