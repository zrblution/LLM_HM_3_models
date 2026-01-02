#!/usr/bin/env python3
"""
Convert an existing output JSON (same shape as coco_2014_data/output.json)
to a new JSON where each sample's `labels` is replaced with the model's
prediction normalized to "yes" or "no".

Usage:
  python convert_output.py \
    --input_json /media/ubuntu/data/xican/coco_2014_data/output.json \
    --model_dir /media/ubuntu/data/xican/hallmodel/coco_2017 \
    --image_root /media/ubuntu/data/xican/hallusion_bench \
    --output_json /media/ubuntu/data/xican/hall_eval/POPE/output_with_model_labels.json

This script re-uses the loading and generation helpers from the existing
`hall_eval/hallusion/run_inference.py`.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
import re


def load_run_inference_module(path: str):
    spec = importlib.util.spec_from_file_location("run_inference_external", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_to_yes_no(answer: str) -> str:
    """Map a model answer string to 'yes' or 'no' using simple heuristics."""
    if not answer:
        return "yes"
    
    s = answer.strip().lower()
    
    # 如果是 fallback 回答，标记为特殊情况
    if s.startswith("[fallback]") or s.startswith("[error"):
        return "yes"  # 默认返回 yes
    
    # 首先检查是否以 yes/no 开头（最常见的情况）
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    
    # 检查第一个单词
    words = s.split()
    first_word = words[0] if words else ""
    # 清理标点
    first_word_clean = first_word.strip(",.!?;:'\"")
    
    if first_word_clean in ["yes", "y", "yeah", "yep", "true", "correct", "right", "affirmative"]:
        return "yes"
    if first_word_clean in ["no", "n", "nope", "false", "incorrect", "wrong", "not"]:
        return "no"
    
    # 检查中文
    if s.startswith("是") or s.startswith("对") or s.startswith("有") or s.startswith("正确"):
        return "yes"
    if s.startswith("不") or s.startswith("没") or s.startswith("否") or s.startswith("错"):
        return "no"
    
    # 检查完整的 yes/no 单词（用空格分隔），只检查前几个词
    for word in words[:5]:
        clean_word = word.strip(",.!?;:'\"").lower()
        if clean_word == "yes":
            return "yes"
        if clean_word == "no":
            return "no"
    
    # 检查否定表达
    negative_patterns = [
        "there is no", "there are no", "there isn't", "there aren't",
        "i don't see", "i cannot see", "not visible", "cannot find",
        "does not", "do not", "doesn't", "don't",
        "is not", "are not", "isn't", "aren't",
        "no,", "not present", "absent"
    ]
    for pattern in negative_patterns:
        if pattern in s:
            return "no"
    
    # 检查肯定表达
    positive_patterns = [
        "there is a", "there are", "i can see", "visible",
        "yes,", "present", "appears", "shows"
    ]
    for pattern in positive_patterns:
        if pattern in s:
            return "yes"
    
    # 默认返回 yes（因为模型倾向于肯定回答）
    return "yes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference (default: 64)")
    parser.add_argument("--use_vcd", action="store_true", help="Wrap model with VCD integration")
    parser.add_argument("--use_inter", action="store_true", help="Wrap model with INTER integration")
    args = parser.parse_args()

    # Path to the existing inference helpers
    repo_root = Path(__file__).resolve().parent.parent
    run_inference_path = repo_root / "hallusion" / "run_inference.py"
    if not run_inference_path.exists():
        raise FileNotFoundError(f"Expected run_inference helpers at {run_inference_path}")

    run_inf = load_run_inference_module(str(run_inference_path))
    # Load tokenizer/processor/model via run_inference helpers (support integration wrappers)
    tokenizer, processor, model = run_inf.load_model_and_tools(args.model_dir, args.device, use_vcd=args.use_vcd, use_inter=args.use_inter)
    
    # 检查模型是否正确加载
    if model is None:
        print("ERROR: Model failed to load!")
        sys.exit(1)
    else:
        print(f"Model loaded successfully from {args.model_dir}")
        print(f"Model type: {type(model)}")
    
    if processor is None:
        print("WARNING: Processor is None, will use fallback generation")
        sys.exit(1)
    if tokenizer is None:
        print("WARNING: Tokenizer is None, will use fallback generation")
        sys.exit(1)
    
    # 设置padding_side为left以支持批量生成
    processor.tokenizer.padding_side = 'left'
    
    # 启用GPU优化
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    from PIL import Image

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples is not None and args.max_samples >= 0:
        total_to_process = min(len(data), args.max_samples)
        data_to_iterate = data[:total_to_process]
    else:
        data_to_iterate = data

    results = []
    yes_count = 0
    no_count = 0
    batch_size = args.batch_size
    
    print(f"Processing {len(data_to_iterate)} samples with batch_size={batch_size}...")
    
    # 批量处理
    for batch_start in range(0, len(data_to_iterate), batch_size):
        batch_end = min(batch_start + batch_size, len(data_to_iterate))
        batch_data = data_to_iterate[batch_start:batch_end]
        
        try:
            # 准备批量数据
            messages_list = []
            images_list = []
            valid_indices = []
            
            for idx, sample in enumerate(batch_data):
                filename = sample.get("img") or sample.get("filename")
                question = sample.get("text") or sample.get("question") or ""
                
                if filename:
                    img_path = os.path.join(args.image_root, filename)
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).convert("RGB")
                            images_list.append(img)
                            messages = [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": question}
                                ]}
                            ]
                            messages_list.append(messages)
                            valid_indices.append(idx)
                        except Exception as e:
                            print(f"WARNING: Failed to load image {img_path}: {e}")
                    else:
                        print(f"WARNING: Image not found: {img_path}")
            
            if not valid_indices:
                # 如果没有有效样本，跳过这个批次
                for sample in batch_data:
                    out_sample = dict(sample)
                    out_sample["labels"] = "yes"
                    out_sample["test_answer"] = "[ERROR: No valid image]"
                    results.append(out_sample)
                    yes_count += 1
                continue
            
            # 批量推理
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
            inputs = processor(text=texts, images=images_list, return_tensors="pt", padding=True)
            
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1)
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 提取assistant回答
            pred_texts = []
            for text in decoded:
                if "assistant" in text:
                    ans = text.split("assistant")[-1].strip()
                else:
                    ans = text.strip()
                pred_texts.append(ans)
            
            # 处理结果
            pred_idx = 0
            for idx, sample in enumerate(batch_data):
                if idx in valid_indices:
                    pred_text = pred_texts[pred_idx]
                    pred_idx += 1
                else:
                    pred_text = "[ERROR: Invalid sample]"
                
                normalized_label = normalize_to_yes_no(pred_text or "")
                
                if normalized_label == "yes":
                    yes_count += 1
                else:
                    no_count += 1
                
                out_sample = dict(sample)
                out_sample["labels"] = normalized_label
                out_sample["test_answer"] = pred_text or ""
                results.append(out_sample)
                
                # 打印前几个样本
                if len(results) <= 5:
                    print(f"\n--- Sample {len(results)} ---")
                    print(f"Question: {sample.get('text', sample.get('question', ''))[:80]}")
                    print(f"Raw output: {pred_text}")
                    print(f"Normalized: {normalized_label}")
                    print(f"Ground truth: {sample.get('labels', 'N/A')}")
                    
        except Exception as e:
            print(f"ERROR in batch {batch_start}-{batch_end}: {e}")
            import traceback
            traceback.print_exc()
            # 回退到逐个处理
            for sample in batch_data:
                out_sample = dict(sample)
                out_sample["labels"] = "yes"
                out_sample["test_answer"] = f"[ERROR: {e}]"
                results.append(out_sample)
                yes_count += 1
        
        # 打印进度
        if (batch_end) % 100 == 0 or batch_end == len(data_to_iterate):
            print(f"Processed {batch_end}/{len(data_to_iterate)} samples (yes: {yes_count}, no: {no_count})")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Yes predictions: {yes_count} ({100*yes_count/len(results):.1f}%)")
    print(f"No predictions: {no_count} ({100*no_count/len(results):.1f}%)")
    print(f"Wrote {len(results)} results to {args.output_json}")


if __name__ == "__main__":
    main()
