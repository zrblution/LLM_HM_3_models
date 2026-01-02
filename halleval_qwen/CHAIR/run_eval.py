#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def run_command(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")

def main():
    chair_root = os.path.dirname(os.path.dirname(__file__))  # .../CHAIR
    parser = argparse.ArgumentParser(description="Orchestrate CHAIR evaluation: generate captions then run chair.py")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument(
        "--model_type",
        default="qwen3_vl",
        choices=["auto", "qwen3_vl", "ministral_vl"],
        help="Forwarded to generate_captions.py. For this script the default is qwen3_vl.",
    )
    parser.add_argument("--image_dir", default=os.path.join(chair_root, "val2014_1000"))
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="Please describe this image in detail.")
    parser.add_argument("--annotation_path", required=True, help="Path to annotations dir (contains instances_val2014.json)")
    parser.add_argument("--synonyms_file", default=os.path.join(chair_root, "synonyms.txt"))
    parser.add_argument("--result_root", default=os.path.join(chair_root, "result"))
    parser.add_argument("--use_vcd", action="store_true", help="Wrap model with VCD integration (forwarded to generate_captions.py)")
    parser.add_argument("--use_inter", action="store_true", help="Wrap model with INTER integration (forwarded to generate_captions.py)")
    args = parser.parse_args()

    model_name = os.path.basename(os.path.normpath(args.model_dir))
    result_dir = os.path.join(args.result_root, model_name)
    os.makedirs(result_dir, exist_ok=True)

    generated_caps_path = os.path.join(result_dir, "generated_captions.json")

    # 1) run generate_captions.py
    gen_script = os.path.join(chair_root, "generate_captions.py")
    gen_cmd = [
        sys.executable, gen_script,
        "--image_dir", args.image_dir,
        "--model_dir", args.model_dir,
        "--model_type", args.model_type,
        "--output_json", generated_caps_path,
        "--num_samples", str(args.num_samples),
        "--device", args.device,
        "--prompt", args.prompt,
    ]
    if args.use_vcd:
        gen_cmd.append("--use_vcd")
    if args.use_inter:
        gen_cmd.append("--use_inter")
    run_command(gen_cmd)

    # 2) run chair.py with cwd set to result_dir so output file is written there
    chair_script = os.path.join(chair_root, "chair.py")
    chair_cmd = [
        sys.executable, chair_script,
        "--cap_file", generated_caps_path,
        "--annotation_path", args.annotation_path,
        "--synonyms_file", args.synonyms_file,
    ]
    run_command(chair_cmd, cwd=result_dir)

    print("All done. Results saved in:", result_dir)

if __name__ == "__main__":
    main()
