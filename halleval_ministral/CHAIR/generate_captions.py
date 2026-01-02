import os
import sys
import json
import random
import argparse
import re
from pathlib import Path

import torch
from PIL import Image

# Ensure we can import custom model code (halltrain) and evaluation integrations.
BASE_DIR = Path(__file__).resolve().parent  # .../halleval/CHAIR
HALLEVAL_DIR = BASE_DIR.parent             # .../halleval
PROJECT_DIR = HALLEVAL_DIR.parent          # .../LLM_HM_3_model
HALLTRAIN_DIR = PROJECT_DIR / "halltrain"  # .../LLM_HM_3_model/halltrain

for p in (HALLEVAL_DIR, HALLTRAIN_DIR):
    if p.exists():
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

# 默认模型输出路径
DEFAULT_MODEL_PATH = "/home/tos_data/LLM-Disentanglement-Hallucination-Mitigation/output_model_Qwen3-VL-2B"

from transformers import AutoTokenizer, AutoProcessor


def _read_model_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _detect_model_type(model_dir: str) -> str:
    cfg = _read_model_config(model_dir)
    model_type = str(cfg.get("model_type", "")).lower()
    architectures = cfg.get("architectures", []) or []
    architectures_l = [str(a) for a in architectures]

    if "mistral" in model_type or any("mistral" in a.lower() for a in architectures_l):
        return "ministral_vl"
    if "qwen" in model_type or any("qwen" in a.lower() for a in architectures_l):
        return "qwen3_vl"

    return "qwen3_vl"


def load_model_and_tools(
    model_dir: str,
    device: str,
    model_type: str = "auto",
    use_vcd: bool = False,
    use_inter: bool = False,
):
    """
    Load tokenizer/processor and either the base model or an integration wrapper.
    If use_vcd/use_inter is set, attempt to load the corresponding loader from the
    `integration` directory and return its wrapper as `model`.
    """
    tokenizer = None
    processor = None
    model = None

    from importlib import util as _util
    base_dir = os.path.dirname(os.path.dirname(__file__))  # hall_eval/
    integ_dir = os.path.join(base_dir, "integration")

    try:
        # VCD/INTER wrappers are currently implemented for Qwen-style models.
        if (use_vcd or use_inter) and (model_type == "auto" or model_type == "qwen3_vl"):
            if use_vcd:
                loader_path = os.path.join(integ_dir, "vcd_integration", "loader.py")
                spec = _util.spec_from_file_location("vcd_loader", loader_path)
                mod = _util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                wrapper = mod.load_model_with_vcd(model_dir, tokenizer=None, processor=None, device=device)
                tokenizer = getattr(wrapper, "tokenizer", None)
                processor = getattr(wrapper, "processor", None)
                model = wrapper
                return tokenizer, processor, model
            if use_inter:
                loader_path = os.path.join(integ_dir, "inter_integration", "loader.py")
                spec = _util.spec_from_file_location("inter_loader", loader_path)
                mod = _util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                wrapper = mod.load_model_with_inter(model_dir, tokenizer=None, processor=None, device=device)
                tokenizer = getattr(wrapper, "tokenizer", None)
                processor = getattr(wrapper, "processor", None)
                model = wrapper
                return tokenizer, processor, model
        elif use_vcd or use_inter:
            print(f"WARNING: --use_vcd/--use_inter currently only supports Qwen-style models; got model_type={model_type}. Loading base model instead.")
    except Exception:
        # Fall back to standard loading below
        pass

    effective_model_type = model_type
    if effective_model_type == "auto":
        effective_model_type = _detect_model_type(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_dir)

    # Prefer using the repo's custom model classes if the checkpoint expects them.
    if effective_model_type == "ministral_vl":
        model_class = None
        try:
            from model.ministral_vl_model import MinistralCustomVLForConditionalGeneration as model_class  # type: ignore
        except Exception:
            try:
                from model.ministral_vl_model import Qwen2_5_CustomVLForConditionalGeneration as model_class  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Failed to import Ministral custom model class from halltrain. "
                    "Please ensure `/home/tos_data/LLM_HM_3_model/halltrain` is accessible and contains `model/ministral_vl_model.py`."
                ) from e

        print(f"Using custom Ministral loader: {model_class.__name__}")
        model = model_class.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.to(device)
        model.eval()
        return tokenizer, processor, model

    # Qwen3-VL path (default)
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except Exception as e:
        raise ImportError("Failed to import Qwen3VLForConditionalGeneration from transformers.") from e

    Qwen2_5_CustomVLForConditionalGeneration = None
    try:
        from model.qwen_vl_model import Qwen2_5_CustomVLForConditionalGeneration
    except Exception:
        Qwen2_5_CustomVLForConditionalGeneration = None

    model_class = Qwen3VLForConditionalGeneration
    cfg = _read_model_config(model_dir)
    architectures = cfg.get("architectures", []) or []
    if "Qwen2_5_CustomVLForConditionalGeneration" in architectures:
        if Qwen2_5_CustomVLForConditionalGeneration is not None:
            model_class = Qwen2_5_CustomVLForConditionalGeneration
            print("✅ Using custom Qwen model class: Qwen2_5_CustomVLForConditionalGeneration")
            print("   Injection modules will be loaded and used during inference.")
        else:
            print("⚠️  Checkpoint expects custom Qwen class but it is not importable; falling back to Qwen3VLForConditionalGeneration (injection modules will be ignored).")
    else:
        print("Using standard Qwen model class: Qwen3VLForConditionalGeneration")

    model = model_class.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    return tokenizer, processor, model



def prepare_inputs_for_sample(sample: dict, processor, image_root: str):
    question = sample.get("question", "")
    user_text = question

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [],
        },
    ]

    filename = sample.get("filename")
    if sample.get("visual_input") in [1, "1", True] and filename:
        rel_path = filename.lstrip("./")
        img_path = os.path.join(image_root, rel_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_pil = Image.open(img_path).convert("RGB")
        messages[1]["content"].append({
            "type": "image",
            "image": image_pil,
            "resize_height": 224,
            "resize_width": 224,
        })

    messages[1]["content"].append({"type": "text", "text": user_text})

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_list = [m["image"] for m in messages[1]["content"] if isinstance(m, dict) and m.get("type") == "image"]
    images_arg = image_list if len(image_list) > 0 else None

    inputs = processor(
        text=[text_input],
        images=images_arg,
        videos=None,
        return_tensors="pt",
        do_resize=True,
        padding=True,
    )
    return inputs


def generate_answer_for_sample(sample: dict, tokenizer, processor, model, device: str, image_root: str):
    # If model is an integration wrapper, prefer calling its API which expects
    # a messages list and a PIL image. Otherwise fall back to standard .generate.
    filename = sample.get("filename")
    image_pil = None
    question = sample.get("question", "")

    # Build messages in the Qwen style
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": []},
    ]
    if sample.get("visual_input") in [1, "1", True] and filename:
        rel_path = filename.lstrip("./")
        img_path = os.path.join(image_root, rel_path)
        try:
            image_pil = Image.open(img_path).convert("RGB")
            messages[1]["content"].append({
                "type": "image",
                "image": image_pil,
                "resize_height": 224,
                "resize_width": 224,
            })
        except Exception:
            image_pil = None
    messages[1]["content"].append({"type": "text", "text": question})

    # Integration wrappers
    if model is not None:
        if hasattr(model, "generate_vcd"):
            try:
                out = model.generate_vcd(messages=messages, image=image_pil, max_length=64)
                return out
            except Exception as e:
                print(f"VCD generation failed: {e}")
        if hasattr(model, "generate_inter"):
            try:
                out = model.generate_inter(messages=messages, image=image_pil, max_length=64)
                return out
            except Exception as e:
                print(f"INTER generation failed: {e}")

    # Fallback to original tensor-based generation
    inputs = prepare_inputs_for_sample(sample, processor, image_root)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    with torch.no_grad():
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = image_grid_thw

        outputs = model.generate(**gen_kwargs)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    assistant_part = decoded
    if "assistant\n" in decoded:
        assistant_part = decoded.split("assistant\n", 1)[1]
    elif "assistant:" in decoded:
        assistant_part = decoded.split("assistant:", 1)[1]
    assistant_part = assistant_part.strip()

    m = re.search(r'(.+?[。\.!?？])', assistant_part)
    if m:
        first_sentence = m.group(1).strip()
    else:
        lines = [ln.strip() for ln in assistant_part.splitlines() if ln.strip()]
        first_sentence = lines[0] if lines else assistant_part

    return first_sentence


def extract_image_id_from_filename(fname: str):
    # Extract the last continuous group of digits from the filename and return as int.
    # Examples:
    #  - COCO_val2014_000000000073.jpg -> 73
    #  - image_12345.png -> 12345
    bn = os.path.basename(fname)
    matches = re.findall(r'(\d+)', bn)
    if matches:
        return int(matches[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description="Sample images from val2014 and generate captions using the model.")
    parser.add_argument("--image_dir", default="/media/ubuntu/data/xican/coco_2014_data/val2014")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--model_type",
        default="auto",
        choices=["auto", "qwen3_vl", "ministral_vl"],
        help="Model family selector. Use 'auto' to infer from config.json (recommended).",
    )
    parser.add_argument("--output_json", default="/media/ubuntu/data/xican/hall_eval/CHAIR/generated_captions.json")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="Please describe this image in detail.")
    parser.add_argument("--use_vcd", action="store_true", help="Wrap model with VCD integration")
    parser.add_argument("--use_inter", action="store_true", help="Wrap model with INTER integration")
    args = parser.parse_args()

    # pick image files
    all_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(all_files) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")
    num = min(args.num_samples, len(all_files))
    sampled = random.sample(all_files, num)

    tokenizer, processor, model = load_model_and_tools(
        args.model_dir,
        args.device,
        model_type=args.model_type,
        use_vcd=args.use_vcd,
        use_inter=args.use_inter,
    )

    results = []
    for fname in sampled:
        sample = {
            "filename": fname,
            "visual_input": 1,
            "question": args.prompt,
        }
        try:
            caption = generate_answer_for_sample(sample, tokenizer, processor, model, args.device, args.image_dir)
        except Exception as e:
            caption = f"[ERROR during generation: {e}]"
        image_id = extract_image_id_from_filename(fname)
        results.append({"image_id": image_id, "caption": caption})

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} captions to {args.output_json}")


if __name__ == "__main__":
    main()
