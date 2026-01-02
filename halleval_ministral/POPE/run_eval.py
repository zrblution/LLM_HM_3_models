#!/usr/bin/env python3
"""
Evaluation runner for POPE.

Usage examples:
  python run_eval.py --dataset coco --model_dir /media/ubuntu/data/xican/hallmodel/coco_2017
  CUDA_VISIBLE_DEVICES=3 python run_eval.py --dataset coco --model_dir /media/ubuntu/data/xican/hallmodel/coco_2017 --gpu 3

This script implements the COCO workflow described in workflow.md:
  1) run convert_output.py for random/popular/adversarial variants
  2) run pope.py to evaluate each generated json against the GT

For other datasets (akvqa, gqa) it currently attempts the same directory conventions;
adjust paths or extend mapping as needed.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    "random",
    "popular",
    "adversarial",
]

def run_cmd(cmd, env=None, dry_run=False):
    print("Running:", " ".join(cmd))
    if dry_run:
        return 0
    try:
        completed = subprocess.run(cmd, check=True, env=env)
        return completed.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with returncode={e.returncode}", file=sys.stderr)
        raise

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def eval_dataset(dataset: str, model_dir: str, model_name: str = None, gpu: str = None, dry_run: bool = False, use_vcd: bool = False, use_inter: bool = False, batch_size: int = 64):
    """
    Unified evaluation for any dataset.
    - For 'coco' uses VARIANTS = ['random','popular','adversarial'] with input files
      named '{dataset}_pope_{variant}.json'.
    - For other datasets, detects json files under ROOT/{dataset}/output and
      extracts variant names. If filenames follow the '{dataset}_pope_{variant}.json'
      pattern the variant is parsed; otherwise the file stem is used as variant.
    """
    dataset = dataset.lower()
    model_dir = str(Path(model_dir).resolve())
    if model_name is None:
        model_name = Path(model_dir).name

    # Centralized result layout: POPE/result/{model_name}/{dataset}/...
    result_dir = ROOT / "result" / model_name / dataset
    ensure_dir(result_dir)

    output_root = ROOT / dataset / "output"
    image_root_base = ROOT / dataset
    # Put evaluation outputs under centralized result dir as well
    eval_out_root = ROOT / "result" / model_name / dataset / "eval"
    ensure_dir(eval_out_root)

    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    variants_pairs = []
    # For all datasets, use the standard VARIANTS with files named '{dataset}_pope_{variant}.json'
    for variant in VARIANTS:
        json_filename = f"{dataset}_pope_{variant}.json"
        json_path = output_root / json_filename
        if json_path.exists():
            variants_pairs.append((variant, json_filename))
        else:
            print(f"Warning: {json_path} not found, skipping variant '{variant}'")

    if not variants_pairs:
        raise RuntimeError(f"No variants detected for dataset '{dataset}' at {output_root}. Please extend script mapping.")

    # Step 1: run convert_output.py to produce predictions
    for variant, input_filename in variants_pairs:
        input_json = output_root / input_filename
        image_root = image_root_base / variant
        pred_json = result_dir / f"{variant}.json"
        ensure_dir(pred_json.parent)

        convert_cmd = [
            sys.executable,
            str(ROOT / "convert_output.py"),
            "--input_json", str(input_json),
            "--model_dir", model_dir,
            "--image_root", str(image_root),
            "--output_json", str(pred_json),
            "--batch_size", str(batch_size),
        ]
        if use_vcd:
            convert_cmd.append("--use_vcd")
        if use_inter:
            convert_cmd.append("--use_inter")
        run_cmd(convert_cmd, env=env, dry_run=dry_run)

    # Step 2: run pope.py for each generated prediction
    for variant, input_filename in variants_pairs:
        gt_file = output_root / input_filename
        pred_file = result_dir / f"{variant}.json"
        out_eval = eval_out_root / f"{variant}_results.json"
        ensure_dir(out_eval.parent)

        pope_cmd = [
            sys.executable,
            str(ROOT / "pope.py"),
            "--gt_file", str(gt_file),
            "--pred_file", str(pred_file),
            "--output_json", str(out_eval),
        ]
        run_cmd(pope_cmd, env=env, dry_run=dry_run)

def eval_generic(dataset: str, model_dir: str, model_name: str = None, gpu: str = None, dry_run: bool = False, use_vcd: bool = False, use_inter: bool = False, batch_size: int = 64):
    dataset = dataset.lower()
    return eval_dataset(dataset=dataset, model_dir=model_dir, model_name=model_name, gpu=gpu, dry_run=dry_run, use_vcd=use_vcd, use_inter=use_inter, batch_size=batch_size)

def parse_args():
    p = argparse.ArgumentParser(description="Run POPE evaluation workflow for a model and dataset.")
    p.add_argument("--dataset", choices=["coco","akvqa","gqa"], default="coco", help="Dataset to evaluate")
    p.add_argument("--model_dir", required=True, help="Path to model directory (e.g. /media/.../coco_2017)")
    p.add_argument("--model_name", required=False, help="Optional name for model (defaults to basename of model_dir)")
    p.add_argument("--gpu", required=False, help="GPU id to set in CUDA_VISIBLE_DEVICES")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for inference (default: 64)")
    p.add_argument("--dry_run", action="store_true", help="Print commands but do not execute them")
    p.add_argument("--use_vcd", action="store_true", help="Wrap model with VCD integration (forward to convert_output.py)")
    p.add_argument("--use_inter", action="store_true", help="Wrap model with INTER integration (forward to convert_output.py)")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        eval_generic(dataset=args.dataset, model_dir=args.model_dir, model_name=args.model_name, gpu=args.gpu, dry_run=args.dry_run, use_vcd=getattr(args, "use_vcd", False), use_inter=getattr(args, "use_inter", False), batch_size=args.batch_size)
    except Exception as e:
        print("Error during evaluation:", e, file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()

