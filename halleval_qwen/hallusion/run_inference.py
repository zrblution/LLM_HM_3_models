#!/usr/bin/env python3
"""Helpers for loading a model and generating answers used by evaluation scripts.

Provides:
- load_model_and_tools(model_dir, device) -> (tokenizer, processor, model)
- generate_answer_for_sample(sample, tokenizer, processor, model, device, image_root)

The implementation aims to be robust and fallback-friendly: it tries to use
`transformers` and `torch` when available, but still returns a readable
fallback string if generation fails.
"""
from typing import Any, Optional, Tuple
import os
import sys
from pathlib import Path
import logging
import re

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TRAIN_ROOT = PROJECT_ROOT.parent.parent / "halltrain" / "LLM-HM-Qwen3-VL-2B-Instruct"
OUTPUT_MODEL_PATH = "/home/tos_data/LLM-Disentanglement-Hallucination-Mitigation/output_model_Qwen3-VL-2B"

# Ensure local project roots are importable (for model.qwen_vl_model, integrations, etc.)
for p in (PROJECT_ROOT, TRAIN_ROOT):
    if p.exists():
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoProcessor,
    )
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoProcessor = None

try:
    from PIL import Image
except Exception:
    Image = None

logger = logging.getLogger(__name__)


def load_model_and_tools(model_dir: str, device: str = "cuda", use_vcd: bool = False, use_inter: bool = False) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Load tokenizer, processor (optional), and model from `model_dir`.

    If `use_vcd` or `use_inter` is set, attempt to load the corresponding wrapper
    from the `integration` utilities and return the wrapper as the `model`
    object. Default behaviour (both flags False) preserves the original loader.
    """
    tokenizer = None
    processor = None
    model = None

    try:
        # Import here to avoid top-level import errors in environments lacking these deps
        import torch
        from transformers import AutoTokenizer, AutoProcessor
        from transformers import Qwen3VLForConditionalGeneration
    except Exception as e:
        logger.warning("Required imports for Qwen3-VL loader not available: %s", e)
        return tokenizer, processor, model
    
    # Try to import custom model class (for fine-tuned models with injection modules)
    Qwen2_5_CustomVLForConditionalGeneration = None
    try:
        from model.qwen_vl_model import Qwen2_5_CustomVLForConditionalGeneration
        logger.info("Custom model class (Qwen2_5_CustomVLForConditionalGeneration) imported successfully")
    except Exception as e:
        logger.info("Custom model class not available, will use standard model: %s", e)

    # If the caller requested VCD or INTER wrappers, try to load from integration loaders.
    try:
        if use_vcd or use_inter:
            # Load integration loader module by path to avoid import path issues.
            import importlib.util
            base_dir = os.path.dirname(os.path.dirname(__file__))  # hall_eval/
            integ_dir = os.path.join(base_dir, "integration")
            if use_vcd:
                loader_path = os.path.join(integ_dir, "vcd_integration", "loader.py")
                spec = importlib.util.spec_from_file_location("vcd_loader", loader_path)
                vcd_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(vcd_mod)
                # loader provides load_model_with_vcd(pretrained_model_name_or_path, tokenizer=None, processor=None, device="cuda")
                wrapper = vcd_mod.load_model_with_vcd(model_dir, tokenizer=None, processor=None, device=device)
                # loader may have created tokenizer/processor; try to extract if present
                tokenizer = getattr(wrapper, "tokenizer", tokenizer)
                processor = getattr(wrapper, "processor", processor)
                model = wrapper
                return tokenizer, processor, model
            if use_inter:
                loader_path = os.path.join(integ_dir, "inter_integration", "loader.py")
                spec = importlib.util.spec_from_file_location("inter_loader", loader_path)
                inter_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(inter_mod)
                wrapper = inter_mod.load_model_with_inter(model_dir, tokenizer=None, processor=None, device=device)
                tokenizer = getattr(wrapper, "tokenizer", tokenizer)
                processor = getattr(wrapper, "processor", processor)
                model = wrapper
                return tokenizer, processor, model
    except Exception as e:
        logger.warning("Failed to load integration wrapper (vcd/inter): %s", e)
        # fall back to regular loading below

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    except Exception as e:
        logger.warning("Failed to load tokenizer from %s: %s", model_dir, e)
        tokenizer = None

    try:
        processor = AutoProcessor.from_pretrained(model_dir)
    except Exception:
        processor = None

    # Auto-detect model architecture from config.json
    model_class = Qwen3VLForConditionalGeneration  # default
    try:
        import json
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            architectures = config.get('architectures', [])
            if 'Qwen2_5_CustomVLForConditionalGeneration' in architectures:
                if Qwen2_5_CustomVLForConditionalGeneration is not None:
                    model_class = Qwen2_5_CustomVLForConditionalGeneration
                    print("✅ Using custom model class: Qwen2_5_CustomVLForConditionalGeneration")
                    print("   Injection modules will be loaded and used during inference!")
                else:
                    print("⚠️  Model requires custom class but it's not available, using standard model")
                    print("   Injection modules will be IGNORED!")
            else:
                print("Using standard model class: Qwen3VLForConditionalGeneration")
    except Exception as e:
        logger.warning("Could not detect model architecture: %s", e)

    try:
        model = model_class.from_pretrained(
            model_dir, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully: %s", model_class.__name__)
    except Exception as e:
        logger.warning("Failed to load model from %s: %s", model_dir, e)
        model = None

    return tokenizer, processor, model


def prepare_inputs_for_sample(sample: dict, processor, image_root: str):
    """Prepare model inputs using the processor (Qwen-style chat template + image handling)."""
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


def generate_answer_for_sample(
    sample: dict,
    tokenizer: Any,
    processor: Any,
    model: Any,
    device: str,
    image_root: str,
) -> str:
    """Generate a textual answer for `sample`.

    sample is expected to contain:
    - "question": str
    - "filename": optional path relative to image_root
    - "visual_input": int flag where 1 indicates an image is present

    The function tries to use the provided tokenizer/model. On any failure
    it falls back to producing a readable placeholder string.
    
    Returns the raw model output (assistant part) without further processing.
    """
    question = (sample.get("question") or "").strip()
    visual_input = int(sample.get("visual_input") or 0)
    filename = sample.get("filename")

    prompt = question
    if visual_input and filename:
        # Try to load image to ensure file exists; we do not assume a specific
        # multimodal model API, so we simply indicate image presence in the prompt.
        image_path = filename if os.path.isabs(filename) else os.path.join(image_root or "", filename)
        try:
            if Image is not None and os.path.exists(image_path):
                # Open and immediately close to validate the file
                with Image.open(image_path) as _img:
                    pass
            prompt = "[IMAGE] " + prompt
        except Exception:
            # If image read fails, still proceed with the text-only prompt.
            prompt = "[IMAGE-UNREADABLE] " + prompt

    # If model provides special multimodal wrappers (VCD or INTER), prefer their APIs.
    try:
        # Build messages and load image if visual input present (wrappers expect messages + PIL image)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": []},
        ]
        image_pil = None
        if visual_input and filename:
            image_path = filename if os.path.isabs(filename) else os.path.join(image_root or "", filename)
            try:
                if Image is not None and os.path.exists(image_path):
                    from PIL import Image as PILImage
                    image_pil = PILImage.open(image_path).convert("RGB")
                    messages[1]["content"].append({
                        "type": "image",
                        "image": image_pil,
                        "resize_height": 224,
                        "resize_width": 224,
                    })
            except Exception:
                # proceed without image if loading fails
                image_pil = None
        messages[1]["content"].append({"type": "text", "text": question})

        # If wrapper exposes generate_vcd or generate_inter, call them.
        if model is not None:
            if hasattr(model, "generate_vcd"):
                try:
                    return model.generate_vcd(messages=messages, image=image_pil, max_length=64)
                except Exception as e:
                    logger.warning("VCD generation failed: %s", e)
            if hasattr(model, "generate_inter"):
                try:
                    return model.generate_inter(messages=messages, image=image_pil, max_length=64)
                except Exception as e:
                    logger.warning("INTER generation failed: %s", e)

    except Exception as e:
        logger.warning("Multimodal wrapper generation attempt failed: %s", e)

    # If we have a model and tokenizer that support text generation, prefer that.
    if tokenizer is not None and model is not None and torch is not None:
        try:
            # Prefer processor-based (chat template + image tensors) generation when processor exists.
            if processor is not None:
                try:
                    inputs = prepare_inputs_for_sample(
                        {"question": question, "filename": filename, "visual_input": visual_input},
                        processor,
                        image_root,
                    )
                    model_device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device(device if torch.cuda.is_available() else "cpu")
                    input_ids = inputs["input_ids"].to(model_device)
                    attention_mask = inputs["attention_mask"].to(model_device)
                    pixel_values = inputs.get("pixel_values")
                    if pixel_values is not None:
                        pixel_values = pixel_values.to(model_device)
                    image_grid_thw = inputs.get("image_grid_thw")
                    if image_grid_thw is not None:
                        image_grid_thw = image_grid_thw.to(model_device)

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
                    
                    # 提取 assistant 部分
                    assistant_part = decoded
                    if "assistant\n" in decoded:
                        assistant_part = decoded.split("assistant\n", 1)[1]
                    elif "assistant:" in decoded:
                        assistant_part = decoded.split("assistant:", 1)[1]
                    elif "Assistant:" in decoded:
                        assistant_part = decoded.split("Assistant:", 1)[1]
                    elif "Assistant\n" in decoded:
                        assistant_part = decoded.split("Assistant\n", 1)[1]
                    assistant_part = assistant_part.strip()

                    # 返回完整的 assistant 回答，不做进一步截取
                    return assistant_part
                except Exception as e:
                    logger.warning("Processor-based generation failed: %s", e)
                    import traceback
                    traceback.print_exc()

            # Fallback to tokenizer prompt-based generation (text-only)
            model_device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device(device if torch.cuda.is_available() else "cpu")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            # Move inputs to model device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Some models return the full prompt+answer; try to strip the prompt if present.
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text or ""
        except Exception as e:
            logger.warning("Generation with model failed: %s", e)
            import traceback
            traceback.print_exc()

    # Last-resort fallback: echo the question with a prefix
    try:
        return f"[FALLBACK] 回答: {prompt}" if prompt else "[FALLBACK] 回答: 是"
    except Exception as e:
        return f"[ERROR during generation fallback: {e}]"


__all__ = ["load_model_and_tools", "generate_answer_for_sample"]
