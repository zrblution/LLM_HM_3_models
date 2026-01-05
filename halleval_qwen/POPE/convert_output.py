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
import multiprocessing as mp
from tqdm import tqdm


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


def worker_process(gpu_id, data_chunk, start_idx, model_dir, image_root, batch_size, use_vcd, use_inter, result_queue):
    """Worker process for multi-GPU inference."""
    import torch
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from PIL import Image
    
    # Path to the existing inference helpers
    repo_root = Path(__file__).resolve().parent.parent
    run_inference_path = repo_root / "hallusion" / "run_inference.py"
    
    run_inf = load_run_inference_module(str(run_inference_path))
    
    print(f"[GPU {gpu_id}] Loading model...")
    tokenizer, processor, model = run_inf.load_model_and_tools(model_dir, "cuda", use_vcd=use_vcd, use_inter=use_inter)
    
    if model is None or processor is None or tokenizer is None:
        print(f"[GPU {gpu_id}] ERROR: Failed to load model/processor/tokenizer!")
        result_queue.put((gpu_id, start_idx, [], 0, 0))
        return
    
    processor.tokenizer.padding_side = 'left'
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    print(f"[GPU {gpu_id}] Model loaded, processing {len(data_chunk)} samples...")
    
    results = []
    yes_count = 0
    no_count = 0
    
    pbar = tqdm(total=len(data_chunk), desc=f"GPU {gpu_id}", position=gpu_id, leave=True)
    
    for i in range(0, len(data_chunk), batch_size):
        batch_end = min(i + batch_size, len(data_chunk))
        batch_data = data_chunk[i:batch_end]
        
        try:
            messages_list = []
            images_list = []
            valid_indices = []
            
            for idx, sample in enumerate(batch_data):
                filename = sample.get("img") or sample.get("filename")
                question = sample.get("text") or sample.get("question") or ""
                
                if filename:
                    img_path = os.path.join(image_root, filename)
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
                        except Exception:
                            pass
            
            if not valid_indices:
                for sample in batch_data:
                    out_sample = dict(sample)
                    out_sample["labels"] = "yes"
                    out_sample["test_answer"] = "[ERROR: No valid image]"
                    results.append(out_sample)
                    yes_count += 1
                pbar.update(len(batch_data))
                pbar.set_postfix(yes=yes_count, no=no_count)
                continue
            
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
            inputs = processor(text=texts, images=images_list, return_tensors="pt", padding=True)
            
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1,
                                        temperature=None, top_p=None, top_k=None)
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            pred_texts = []
            for text in decoded:
                if "assistant" in text:
                    ans = text.split("assistant")[-1].strip()
                else:
                    ans = text.strip()
                pred_texts.append(ans)
            
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
                
        except Exception as e:
            for sample in batch_data:
                out_sample = dict(sample)
                out_sample["labels"] = "yes"
                out_sample["test_answer"] = f"[ERROR: {e}]"
                results.append(out_sample)
                yes_count += 1
        
        pbar.update(len(batch_data))
        pbar.set_postfix(yes=yes_count, no=no_count)
    
    pbar.close()
    print(f"[GPU {gpu_id}] Completed!")
    result_queue.put((gpu_id, start_idx, results, yes_count, no_count))


def multi_gpu_inference(data, model_dir, image_root, batch_size, gpus, use_vcd, use_inter):
    """Run inference across multiple GPUs in parallel."""
    gpu_list = [int(g.strip()) for g in gpus.split(",")]
    num_gpus = len(gpu_list)
    
    # Split data across GPUs
    chunk_size = len(data) // num_gpus
    chunks = []
    start_indices = []
    
    for i, gpu_id in enumerate(gpu_list):
        start_idx = i * chunk_size
        if i == num_gpus - 1:
            # Last GPU gets remaining samples
            chunk = data[start_idx:]
        else:
            chunk = data[start_idx:start_idx + chunk_size]
        chunks.append(chunk)
        start_indices.append(start_idx)
    
    print(f"=== Multi-GPU Inference ===")
    print(f"Total samples: {len(data)}")
    print(f"GPUs: {gpu_list}")
    for i, gpu_id in enumerate(gpu_list):
        print(f"  GPU {gpu_id}: {len(chunks[i])} samples (starting at index {start_indices[i]})")
    print()
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for i, gpu_id in enumerate(gpu_list):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, chunks[i], start_indices[i], model_dir, image_root, 
                  batch_size, use_vcd, use_inter, result_queue)
        )
        p.start()
        processes.append(p)
    
    print(f"\nWaiting for {num_gpus} workers to complete...")
    
    # Collect results
    all_results = {}
    total_yes = 0
    total_no = 0
    
    for _ in range(num_gpus):
        gpu_id, start_idx, results, yes_count, no_count = result_queue.get()
        all_results[start_idx] = results
        total_yes += yes_count
        total_no += no_count
        print(f"\n[INFO] Worker {gpu_id} completed: {len(results)} samples, yes={yes_count}, no={no_count}")
    
    print("\nAll workers completed, joining processes...")
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Merge results in order
    merged_results = []
    for start_idx in sorted(all_results.keys()):
        merged_results.extend(all_results[start_idx])
    
    print(f"\nMerged {len(merged_results)} results from {num_gpus} workers")
    
    return merged_results, total_yes, total_no


def single_gpu_inference(data, model_dir, image_root, batch_size, device, use_vcd, use_inter):
    """Run inference on a single GPU."""
    import torch
    from PIL import Image
    
    repo_root = Path(__file__).resolve().parent.parent
    run_inference_path = repo_root / "hallusion" / "run_inference.py"
    
    run_inf = load_run_inference_module(str(run_inference_path))
    tokenizer, processor, model = run_inf.load_model_and_tools(model_dir, device, use_vcd=use_vcd, use_inter=use_inter)
    
    if model is None:
        print("ERROR: Model failed to load!")
        sys.exit(1)
    
    if processor is None or tokenizer is None:
        print("ERROR: Processor or tokenizer failed to load!")
        sys.exit(1)
    
    processor.tokenizer.padding_side = 'left'
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    results = []
    yes_count = 0
    no_count = 0
    
    print(f"Processing {len(data)} samples with batch_size={batch_size}...")
    
    for batch_start in tqdm(range(0, len(data), batch_size), desc="Processing"):
        batch_end = min(batch_start + batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        try:
            messages_list = []
            images_list = []
            valid_indices = []
            
            for idx, sample in enumerate(batch_data):
                filename = sample.get("img") or sample.get("filename")
                question = sample.get("text") or sample.get("question") or ""
                
                if filename:
                    img_path = os.path.join(image_root, filename)
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
                        except Exception:
                            pass
            
            if not valid_indices:
                for sample in batch_data:
                    out_sample = dict(sample)
                    out_sample["labels"] = "yes"
                    out_sample["test_answer"] = "[ERROR: No valid image]"
                    results.append(out_sample)
                    yes_count += 1
                continue
            
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
            inputs = processor(text=texts, images=images_list, return_tensors="pt", padding=True)
            
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1)
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            pred_texts = []
            for text in decoded:
                if "assistant" in text:
                    ans = text.split("assistant")[-1].strip()
                else:
                    ans = text.strip()
                pred_texts.append(ans)
            
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
                
        except Exception as e:
            for sample in batch_data:
                out_sample = dict(sample)
                out_sample["labels"] = "yes"
                out_sample["test_answer"] = f"[ERROR: {e}]"
                results.append(out_sample)
                yes_count += 1
    
    return results, yes_count, no_count


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
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU data parallel inference")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU ids for multi-GPU mode")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples is not None and args.max_samples >= 0:
        total_to_process = min(len(data), args.max_samples)
        data_to_iterate = data[:total_to_process]
    else:
        data_to_iterate = data

    if args.multi_gpu:
        results, yes_count, no_count = multi_gpu_inference(
            data_to_iterate, args.model_dir, args.image_root, 
            args.batch_size, args.gpus, args.use_vcd, args.use_inter
        )
    else:
        results, yes_count, no_count = single_gpu_inference(
            data_to_iterate, args.model_dir, args.image_root,
            args.batch_size, args.device, args.use_vcd, args.use_inter
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Yes predictions: {yes_count} ({100*yes_count/len(results):.1f}%)")
    print(f"No predictions: {no_count} ({100*no_count/len(results):.1f}%)")
    print(f"Wrote {len(results)} results to {args.output_json}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
