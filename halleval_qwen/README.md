# Hallucination Evaluation (halleval)

本目录封装了两套常用的多模态幻觉测评：

- **CHAIR**：基于 COCO GT objects，衡量 caption 中“幻觉物体”比例（CHAIRs/CHAIRi）。
- **POPE**：基于 Yes/No 问答的幻觉评测，输出 Accuracy/Precision/Recall/F1。

下面的说明主要参考并汇总自 `CHAIR/readme.md` 与 `POPE/readme.md`，并结合本目录中的实际脚本路径整理为可直接运行的命令。

## 0. 目录结构与数据

- CHAIR
  - 小规模可直接跑的示例数据：
    - 图片：`CHAIR/val2014_1000/`（1000 张）
    - 标注：`CHAIR/annotations_1000/`（包含 `instances_val2014.json` 等）
- POPE
  - 本项目已内置两套数据（每个 split 各 3 个变体 random/popular/adversarial，图片各 500 张）：
    - COCO：`POPE/coco/{random,popular,adversarial}/` + `POPE/coco/output/*`
    - GQA：`POPE/gqa/{random,popular,adversarial}/` + `POPE/gqa/output/*`

## 1. 运行环境

请确保你的运行环境能够正常加载测评模型（`--model_dir` 指向本地模型目录）。

## 2. CHAIR 测评

CHAIR 的流程是两步：

1) 用模型对图片生成 caption（`generate_captions.py`）  
2) 用 COCO 标注对 caption 进行评测（`chair.py`）

### ：一键脚本（生成 caption + 评测）

在 `halleval` 目录下执行：

```bash
cd /home/tos_data/LLM_HM_3_model/halleval

# Qwen3-VL（或与你的模型加载逻辑匹配的脚本）
python CHAIR/run_eval/run_eval_Qwen3_VL.py \
  --model_dir /path/to/model \
  --image_dir CHAIR/val2014_1000 \
  --annotation_path CHAIR/annotations_1000 \
  --synonyms_file CHAIR/synonyms.txt \
  --result_root CHAIR/result \
  --num_samples 1000 \
  --device cuda
```

可选参数（用于启用降低幻觉的封装模块；不加则不启用）：

- `--use_vcd`
- `--use_inter`

输出文件：

- captions：`CHAIR/result/<模型目录名>/generated_captions.json`
- 评测结果：`CHAIR/result/<模型目录名>/chair_results.json`

```

## 3. POPE 测评

POPE 的流程同样是两步：

1) 对 POPE 的问题集跑推理，得到模型预测（`convert_output.py` 会把模型输出归一化为 yes/no）  
2) 将预测与 GT 对齐计算指标（`pope.py`）

### 一键脚本（推理 + 评测）

在 `halleval` 目录下执行：

```bash
cd /home/tos_data/LLM_HM_3_model/halleval

# COCO
python POPE/run_eval/run_eval_Qwen3_VL.py \
  --dataset coco \
  --model_dir /path/to/model \
  --gpu 0

# GQA
python POPE/run_eval/run_eval_Qwen3_VL.py \
  --dataset gqa \
  --model_dir /path/to/model \
  --gpu 0
```

可选参数（用于启用降低幻觉的封装模块；不加则不启用）：

- `--use_vcd`
- `--use_inter`

输出文件（会按模型名与数据集分目录存放）：

- 预测结果：`POPE/result/<模型目录名>/<dataset>/{random,popular,adversarial}.json`
- 指标结果：`POPE/result/<模型目录名>/<dataset>/eval/{random,popular,adversarial}_results.json`


## 4. 常见问题

- **多卡/指定 GPU**：POPE 一键脚本支持 `--gpu`（会设置 `CUDA_VISIBLE_DEVICES`）；CHAIR 可自行在命令前加 `CUDA_VISIBLE_DEVICES=0` 并保持 `--device cuda`。
- **路径都是相对路径**：建议从 `halleval/` 目录执行命令（上面的示例均按此组织）。

