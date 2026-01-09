#!/usr/bin/env python3
"""
Convert an existing POPE input JSON to a new JSON where each sample's `labels`
is replaced with the model's prediction normalized to "yes" or "no".

Key fixes:
1) bf16 dtype mismatch fix:
   - Cast floating tensors in processor outputs to model.dtype before generate().

2) Safer decoding:
   - Decode only the newly generated tokens (exclude the prompt).

3) Error handling:
   - Default: fail-fast (non-zero exit) if generation errors happen.
   - Optional: --allow_errors to keep going (not recommended for official eval).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
import multiprocessing as mp
from typing import Any, Dict, List, Optional

from tqdm import tqdm


def _is_cuda_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "cuda out of memory" in msg or "out of memory" in msg


def load_run_inference_module(path: str):
    spec = importlib.util.spec_from_file_location("run_inference_external", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def normalize_to_yes_no(answer: str) -> str:
    """Map a model answer string to 'yes' or 'no' using simple heuristics."""
    if not answer:
        return "yes"

    s = answer.strip().lower()

    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"

    words = s.split()
    first = words[0].strip(",.!?;:'\"") if words else ""
    if first in ["yes", "y", "yeah", "yep", "true", "correct", "right", "affirmative"]:
        return "yes"
    if first in ["no", "n", "nope", "false", "incorrect", "wrong"]:
        return "no"

    if s.startswith(("是", "对", "有", "正确")):
        return "yes"
    if s.startswith(("不", "没", "否", "错")):
        return "no"

    for w in words[:6]:
        w = w.strip(",.!?;:'\"").lower()
        if w == "yes":
            return "yes"
        if w == "no":
            return "no"

    negative_patterns = [
        "there is no", "there are no", "there isn't", "there aren't",
        "i don't see", "i cannot see", "not visible", "cannot find",
        "does not", "do not", "doesn't", "don't",
        "is not", "are not", "isn't", "aren't",
        "not present", "absent",
    ]
    for p in negative_patterns:
        if p in s:
            return "no"

    positive_patterns = [
        "there is a", "there are", "i can see", "visible",
        "present", "appears", "shows",
    ]
    for p in positive_patterns:
        if p in s:
            return "yes"

    return "yes"


def _move_inputs_to_model_device_and_dtype(inputs: Dict[str, Any], model) -> Dict[str, Any]:
    """Move tensors to model device, and cast floating tensors to model dtype."""
    import torch

    try:
        p = next(model.parameters())
        device = p.device
        dtype = p.dtype
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype, non_blocking=True)
            else:
                out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _decode_generated_only(tokenizer, output_ids, input_ids) -> List[str]:
    """Decode only the continuation part (exclude prompt)."""
    if input_ids is None:
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    prompt_len = input_ids.shape[1]
    gen_ids = output_ids[:, prompt_len:]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


def worker_process(
    local_rank: int,
    gpu_id: int,
    data_chunk: list,
    start_idx: int,
    model_dir: str,
    image_root: str,
    batch_size: int,
    use_vcd: bool,
    use_inter: bool,
    allow_errors: bool,
    max_new_tokens: int,
    result_queue,
):
    """Worker process for multi-GPU inference."""
    fatal_error: Optional[str] = None
    results: list = []
    yes_count = 0
    no_count = 0
    error_count = 0

    try:
        import torch
        from PIL import Image

        if not torch.cuda.is_available():
            fatal_error = "CUDA is not available in this worker process."
            result_queue.put((gpu_id, start_idx, results, yes_count, no_count, error_count, fatal_error))
            return

        if local_rank >= torch.cuda.device_count():
            fatal_error = (
                f"Requested local_rank={local_rank} but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
            result_queue.put((gpu_id, start_idx, results, yes_count, no_count, error_count, fatal_error))
            return

        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

        repo_root = Path(__file__).resolve().parent.parent
        run_inference_path = repo_root / "hallusion" / "run_inference.py"
        run_inf = load_run_inference_module(str(run_inference_path))

        print(f"[GPU {gpu_id}] Loading model...")
        tokenizer, processor, model = run_inf.load_model_and_tools(
            model_dir, device, use_vcd=use_vcd, use_inter=use_inter
        )

        if model is None or processor is None or tokenizer is None:
            fatal_error = f"Failed to load model/processor/tokenizer on GPU {gpu_id}"
            result_queue.put((gpu_id, start_idx, results, yes_count, no_count, error_count, fatal_error))
            return

        try:
            processor.tokenizer.padding_side = "left"
        except Exception:
            pass

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        print(f"[GPU {gpu_id}] Model loaded, processing {len(data_chunk)} samples...")

        pbar = tqdm(total=len(data_chunk), desc=f"GPU {gpu_id}", position=local_rank, leave=True)

        with torch.inference_mode():
            def process_batch(batch_data: list):
                nonlocal yes_count, no_count, error_count, fatal_error

                messages_list = []
                images_list = []
                valid_indices = []

                for idx, sample in enumerate(batch_data):
                    filename = sample.get("img") or sample.get("filename")
                    question = (sample.get("text") or sample.get("question") or "").strip()

                    if not filename or not question:
                        continue

                    img_path = os.path.join(image_root, filename)
                    if not os.path.exists(img_path):
                        continue

                    try:
                        img = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue

                    images_list.append(img)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": f"{question}\nPlease answer with yes or no."},
                            ],
                        }
                    ]
                    messages_list.append(messages)
                    valid_indices.append(idx)

                if not valid_indices:
                    for sample in batch_data:
                        out_sample = dict(sample)
                        out_sample["labels"] = "yes"
                        out_sample["test_answer"] = "[ERROR: No valid image]"
                        results.append(out_sample)
                        error_count += 1
                    pbar.update(len(batch_data))
                    pbar.set_postfix(yes=yes_count, no=no_count, err=error_count)
                    if not allow_errors:
                        fatal_error = "No valid samples in a batch (image load failed)."
                    return

                try:
                    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
                    inputs = processor(text=texts, images=images_list, return_tensors="pt", padding=True)

                    inputs = _move_inputs_to_model_device_and_dtype(inputs, model)

                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=getattr(tokenizer, "pad_token_id", None),
                        eos_token_id=getattr(tokenizer, "eos_token_id", None),
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )

                    decoded = _decode_generated_only(tokenizer, output_ids, inputs.get("input_ids"))

                    pred_idx = 0
                    for idx, sample in enumerate(batch_data):
                        if idx in valid_indices:
                            pred_text = (decoded[pred_idx] or "").strip()
                            pred_idx += 1
                        else:
                            pred_text = "[ERROR: Invalid sample]"

                        normalized = normalize_to_yes_no(pred_text)

                        if normalized == "yes":
                            yes_count += 1
                        else:
                            no_count += 1

                        out_sample = dict(sample)
                        out_sample["labels"] = normalized
                        out_sample["test_answer"] = pred_text
                        results.append(out_sample)

                except Exception as e:
                    if _is_cuda_oom(e) and len(batch_data) > 1:
                        # Retry with smaller batches on CUDA OOM.
                        try:
                            del inputs
                            del output_ids
                        except Exception:
                            pass
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        try:
                            import gc

                            gc.collect()
                        except Exception:
                            pass

                        mid = len(batch_data) // 2
                        process_batch(batch_data[:mid])
                        if fatal_error and not allow_errors:
                            return
                        process_batch(batch_data[mid:])
                        return

                    error_count += len(batch_data)
                    for sample in batch_data:
                        out_sample = dict(sample)
                        out_sample["labels"] = "yes"
                        out_sample["test_answer"] = f"[ERROR: {e}]"
                        results.append(out_sample)
                    if not allow_errors:
                        fatal_error = f"Generation failed on GPU {gpu_id}: {e}"
                    pbar.update(len(batch_data))
                    pbar.set_postfix(yes=yes_count, no=no_count, err=error_count)
                    return

                pbar.update(len(batch_data))
                pbar.set_postfix(yes=yes_count, no=no_count, err=error_count)

            for i in range(0, len(data_chunk), batch_size):
                if fatal_error:
                    break
                batch_end = min(i + batch_size, len(data_chunk))
                process_batch(data_chunk[i:batch_end])
                if fatal_error and not allow_errors:
                    break

        pbar.close()
        if fatal_error:
            print(f"[GPU {gpu_id}] Stopped early due to error: {fatal_error}")
        else:
            print(f"[GPU {gpu_id}] Completed!")

    except Exception as e:
        fatal_error = f"Worker crash on GPU {gpu_id}: {e}"

    finally:
        result_queue.put((gpu_id, start_idx, results, yes_count, no_count, error_count, fatal_error))


def multi_gpu_inference(
    data: list,
    model_dir: str,
    image_root: str,
    batch_size: int,
    gpus: str,
    use_vcd: bool,
    use_inter: bool,
    allow_errors: bool,
    max_new_tokens: int,
):
    """Run inference across multiple GPUs in parallel."""
    # Ensure the visible devices map matches the requested GPU list.
    # We use local_rank indices (cuda:0..N-1) inside workers, so this provides a stable mapping even when
    # the job scheduler already remaps devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    import torch

    gpu_list = [int(g.strip()) for g in gpus.split(",") if g.strip() != ""]
    num_gpus = len(gpu_list)
    if num_gpus <= 0:
        raise ValueError("--gpus is empty")

    visible = torch.cuda.device_count()
    if visible <= 0:
        raise RuntimeError(f"No CUDA GPUs are available (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    if visible < num_gpus:
        print(
            f"Warning: requested {num_gpus} GPU(s) {gpu_list}, but only {visible} device(s) are visible; "
            f"using first {visible}."
        )
        gpu_list = gpu_list[:visible]
        num_gpus = len(gpu_list)

    chunk_size = len(data) // num_gpus
    chunks = []
    start_indices = []

    for i, gpu_id in enumerate(gpu_list):
        start_idx = i * chunk_size
        if i == num_gpus - 1:
            chunk = data[start_idx:]
        else:
            chunk = data[start_idx:start_idx + chunk_size]
        chunks.append(chunk)
        start_indices.append(start_idx)

    print("=== Multi-GPU Inference ===")
    print(f"Total samples: {len(data)}")
    print(f"GPUs: {gpu_list}")
    for i, gpu_id in enumerate(gpu_list):
        print(f"  GPU {gpu_id}: {len(chunks[i])} samples (starting at index {start_indices[i]})")
    print()

    result_queue = mp.Queue()

    processes = []
    for local_rank, gpu_id in enumerate(gpu_list):
        p = mp.Process(
            target=worker_process,
            args=(
                local_rank,
                gpu_id,
                chunks[local_rank],
                start_indices[local_rank],
                model_dir,
                image_root,
                batch_size,
                use_vcd,
                use_inter,
                allow_errors,
                max_new_tokens,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    print(f"\nWaiting for {num_gpus} workers to complete...")

    all_results: Dict[int, list] = {}
    total_yes = 0
    total_no = 0
    total_err = 0
    fatal_errors: List[str] = []

    for _ in range(num_gpus):
        gpu_id, start_idx, results, yes_count, no_count, err_count, fatal = result_queue.get()
        all_results[start_idx] = results
        total_yes += yes_count
        total_no += no_count
        total_err += err_count
        if fatal:
            fatal_errors.append(f"[GPU {gpu_id}] {fatal}")
        print(f"\n[INFO] Worker {gpu_id} completed: {len(results)} samples, yes={yes_count}, no={no_count}, err={err_count}")

    print("\nAll workers completed, joining processes...")
    for p in processes:
        p.join()

    merged_results: list = []
    for start_idx in sorted(all_results.keys()):
        merged_results.extend(all_results[start_idx])

    print(f"\nMerged {len(merged_results)} results from {num_gpus} workers")

    return merged_results, total_yes, total_no, total_err, fatal_errors


def single_gpu_inference(
    data: list,
    model_dir: str,
    image_root: str,
    batch_size: int,
    device: str,
    use_vcd: bool,
    use_inter: bool,
    allow_errors: bool,
    max_new_tokens: int,
):
    """Run inference on a single GPU."""
    import torch
    from PIL import Image

    repo_root = Path(__file__).resolve().parent.parent
    run_inference_path = repo_root / "hallusion" / "run_inference.py"
    run_inf = load_run_inference_module(str(run_inference_path))

    tokenizer, processor, model = run_inf.load_model_and_tools(model_dir, device, use_vcd=use_vcd, use_inter=use_inter)

    if model is None or processor is None or tokenizer is None:
        raise RuntimeError("Model / processor / tokenizer failed to load")

    try:
        processor.tokenizer.padding_side = "left"
    except Exception:
        pass

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    results = []
    yes_count = 0
    no_count = 0
    error_count = 0
    fatal_errors: List[str] = []

    print(f"Processing {len(data)} samples with batch_size={batch_size}...")

    with torch.inference_mode():
        pbar = tqdm(total=len(data), desc="Processing")

        def process_batch(batch_data: list):
            nonlocal yes_count, no_count, error_count

            messages_list = []
            images_list = []
            valid_indices = []

            for idx, sample in enumerate(batch_data):
                filename = sample.get("img") or sample.get("filename")
                question = (sample.get("text") or sample.get("question") or "").strip()

                if not filename or not question:
                    continue

                img_path = os.path.join(image_root, filename)
                if not os.path.exists(img_path):
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                images_list.append(img)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": f"{question}\nPlease answer with yes or no."},
                        ],
                    }
                ]
                messages_list.append(messages)
                valid_indices.append(idx)

            if not valid_indices:
                error_count += len(batch_data)
                for sample in batch_data:
                    out_sample = dict(sample)
                    out_sample["labels"] = "yes"
                    out_sample["test_answer"] = "[ERROR: No valid image]"
                    results.append(out_sample)
                pbar.update(len(batch_data))
                if not allow_errors:
                    fatal_errors.append("No valid samples in a batch (image load failed).")
                return

            try:
                texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
                inputs = processor(text=texts, images=images_list, return_tensors="pt", padding=True)
                inputs = _move_inputs_to_model_device_and_dtype(inputs, model)

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )

                decoded = _decode_generated_only(tokenizer, output_ids, inputs.get("input_ids"))

                pred_idx = 0
                for idx, sample in enumerate(batch_data):
                    if idx in valid_indices:
                        pred_text = (decoded[pred_idx] or "").strip()
                        pred_idx += 1
                    else:
                        pred_text = "[ERROR: Invalid sample]"

                    normalized = normalize_to_yes_no(pred_text)

                    if normalized == "yes":
                        yes_count += 1
                    else:
                        no_count += 1

                    out_sample = dict(sample)
                    out_sample["labels"] = normalized
                    out_sample["test_answer"] = pred_text
                    results.append(out_sample)
                pbar.update(len(batch_data))

            except Exception as e:
                if _is_cuda_oom(e) and len(batch_data) > 1:
                    try:
                        del inputs
                        del output_ids
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        import gc

                        gc.collect()
                    except Exception:
                        pass

                    mid = len(batch_data) // 2
                    process_batch(batch_data[:mid])
                    if fatal_errors and not allow_errors:
                        return
                    process_batch(batch_data[mid:])
                    return

                error_count += len(batch_data)
                for sample in batch_data:
                    out_sample = dict(sample)
                    out_sample["labels"] = "yes"
                    out_sample["test_answer"] = f"[ERROR: {e}]"
                    results.append(out_sample)
                pbar.update(len(batch_data))
                if not allow_errors:
                    fatal_errors.append(str(e))
                return

        for batch_start in range(0, len(data), batch_size):
            if fatal_errors and not allow_errors:
                break
            batch_end = min(batch_start + batch_size, len(data))
            process_batch(data[batch_start:batch_end])
            if fatal_errors and not allow_errors:
                break

        pbar.close()

    return results, yes_count, no_count, error_count, fatal_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference (default: 64)")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Max new tokens to generate (default: 16)")
    parser.add_argument("--use_vcd", action="store_true", help="Wrap model with VCD integration")
    parser.add_argument("--use_inter", action="store_true", help="Wrap model with INTER integration")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU data parallel inference")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU ids for multi-GPU mode")
    parser.add_argument("--allow_errors", action="store_true", help="Do not stop on generation errors (NOT recommended for official eval)")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples is not None and args.max_samples >= 0:
        total_to_process = min(len(data), args.max_samples)
        data_to_iterate = data[:total_to_process]
    else:
        data_to_iterate = data

    if args.multi_gpu:
        results, yes_count, no_count, err_count, fatal_errors = multi_gpu_inference(
            data_to_iterate,
            args.model_dir,
            args.image_root,
            args.batch_size,
            args.gpus,
            args.use_vcd,
            args.use_inter,
            args.allow_errors,
            args.max_new_tokens,
        )
    else:
        results, yes_count, no_count, err_count, fatal_errors = single_gpu_inference(
            data_to_iterate,
            args.model_dir,
            args.image_root,
            args.batch_size,
            args.device,
            args.use_vcd,
            args.use_inter,
            args.allow_errors,
            args.max_new_tokens,
        )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = len(results) if results else 0
    print("\n=== Summary ===")
    print(f"Total samples: {total}")
    if total > 0:
        print(f"Yes predictions: {yes_count} ({100*yes_count/total:.1f}%)")
        print(f"No predictions: {no_count} ({100*no_count/total:.1f}%)")
        print(f"Error samples : {err_count} ({100*err_count/total:.1f}%)")
    print(f"Wrote {total} results to {args.output_json}")

    if fatal_errors and not args.allow_errors:
        print("\n=== FATAL ERRORS ===")
        for msg in fatal_errors[:10]:
            print(msg)
        print(f"(showing {min(10, len(fatal_errors))}/{len(fatal_errors)})")
        sys.exit(3)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
