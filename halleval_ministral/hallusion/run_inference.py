#!/usr/bin/env python3
"""Helpers for loading Ministral model and generating answers used by evaluation scripts.

Provides:
- load_model_and_tools(model_dir, device) -> (tokenizer, processor, model)
- generate_answer_for_sample(sample, tokenizer, processor, model, device, image_root)

This version is adapted for Ministral-3-8B-Instruct-2512 (multimodal model with vision capabilities).
"""
from typing import Any, Optional, Tuple
import os
import sys
from pathlib import Path
import logging
import re

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TRAIN_ROOT = PROJECT_ROOT.parent.parent / "halltrain" / "LLM-HM-Ministral-3-8B-Instruct"
OUTPUT_MODEL_PATH = "/home/tos_data/LLM-Disentanglement-Hallucination-Mitigation/output_model_Ministral-3-8B"

# Ensure local project roots are importable
for p in (PROJECT_ROOT, TRAIN_ROOT):
    if p.exists():
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForVision2Seq,
        AutoModelForCausalLM,
        AutoProcessor,
    )
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForVision2Seq = None
    AutoModelForCausalLM = None
    AutoProcessor = None

try:
    from PIL import Image
except Exception:
    Image = None

logger = logging.getLogger(__name__)


def load_model_and_tools(model_dir: str, device: str = "cuda", use_vcd: bool = False, use_inter: bool = False) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Load tokenizer, processor, and model from `model_dir`.
    
    Ministral-3-8B-Instruct-2512 is a multimodal model with vision capabilities.
    严格使用 Mistral3ForConditionalGeneration，不使用回退机制。
    """
    tokenizer = None
    processor = None
    model = None

    try:
        import torch
        from transformers import AutoTokenizer, AutoProcessor
        from transformers import Mistral3ForConditionalGeneration
    except Exception as e:
        logger.warning("Required imports for Ministral loader not available: %s", e)
        return tokenizer, processor, model
    
    # Try to import custom model class (for fine-tuned models with injection modules)
    # Note: The class is named Qwen2_5_CustomVLForConditionalGeneration for compatibility
    Qwen2_5_CustomVLForConditionalGeneration = None
    try:
        from model.ministral_vl_model import Qwen2_5_CustomVLForConditionalGeneration
        logger.info("Custom model class (Qwen2_5_CustomVLForConditionalGeneration for Ministral) imported successfully")
    except Exception as e:
        logger.info("Custom model class not available, will use standard model: %s", e)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
        # 确保 tokenizer 有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        logger.warning("Failed to load tokenizer from %s: %s", model_dir, e)
        tokenizer = None

    # 加载 processor（Ministral 使用 PixtralProcessor）
    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        logger.info("Processor loaded successfully (PixtralProcessor).")
    except Exception as e:
        logger.warning("Could not load processor: %s", e)
        processor = None

    # Auto-detect model architecture from config.json
    model_class = Mistral3ForConditionalGeneration  # default
    try:
        import json
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            architectures = config.get('architectures', [])
            # Check for custom model class (named Qwen2_5_CustomVLForConditionalGeneration for compatibility)
            if 'Qwen2_5_CustomVLForConditionalGeneration' in architectures:
                if Qwen2_5_CustomVLForConditionalGeneration is not None:
                    model_class = Qwen2_5_CustomVLForConditionalGeneration
                    print("✅ Using custom model class: Qwen2_5_CustomVLForConditionalGeneration (Ministral)")
                    print("   Injection modules will be loaded and used during inference!")
                else:
                    print("⚠️  Model requires custom class but it's not available, using standard model")
                    print("   Injection modules will be IGNORED!")
            else:
                print("Using standard model class: Mistral3ForConditionalGeneration")
    except Exception as e:
        logger.warning("Could not detect model architecture: %s", e)

    try:
        model = model_class.from_pretrained(
            model_dir, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Loaded as %s (multimodal model).", model_class.__name__)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.warning("Failed to load Ministral model: %s", e)
        model = None

    return tokenizer, processor, model


def prepare_inputs_for_sample(sample: dict, processor, tokenizer, image_root: str, device: str):
    """Prepare model inputs for Ministral multimodal model.
    使用 Ministral 官方聊天模板格式。
    """
    question = sample.get("question", "")
    filename = sample.get("filename")
    visual_input = sample.get("visual_input") in [1, "1", True]
    
    # 加载图像
    image_pil = None
    if visual_input and filename:
        rel_path = filename.lstrip("./")
        img_path = os.path.join(image_root, rel_path)
        if os.path.exists(img_path) and Image is not None:
            try:
                image_pil = Image.open(img_path).convert("RGB")
            except Exception:
                image_pil = None
    
    # 构建 Ministral 官方格式的消息（使用 processor.apply_chat_template）
    messages = [
        {"role": "user", "content": []}
    ]
    
    # 如果有图像，添加图像内容
    if image_pil is not None:
        messages[0]["content"].append({"type": "image", "image": image_pil})
    
    # 添加文本内容
    messages[0]["content"].append({"type": "text", "text": question})
    
    # 使用 processor 的 apply_chat_template 生成正确格式的输入
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 准备输入
    if image_pil is not None:
        inputs = processor(
            text=[text_input],
            images=[image_pil],
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text_input],
            return_tensors="pt",
        )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
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

    Ministral-3-8B-Instruct-2512 supports both text and image inputs.
    """
    question = (sample.get("question") or "").strip()
    visual_input = sample.get("visual_input") in [1, "1", True]
    filename = sample.get("filename")
    
    if not question:
        return "[FALLBACK] 回答: 是"

    # 如果有 tokenizer 和 model，使用它们生成回答
    if tokenizer is not None and model is not None and torch is not None:
        try:
            inputs = prepare_inputs_for_sample(sample, processor, tokenizer, image_root, device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 解码输出
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的回答部分
            if "[/INST]" in decoded:
                assistant_part = decoded.split("[/INST]")[-1].strip()
            elif "assistant" in decoded.lower():
                for sep in ["assistant\n", "assistant:", "Assistant:", "Assistant\n"]:
                    if sep in decoded:
                        assistant_part = decoded.split(sep, 1)[1].strip()
                        break
                else:
                    assistant_part = decoded
            else:
                assistant_part = decoded
            
            return assistant_part.strip() or "yes"
            
        except Exception as e:
            logger.warning("Generation with Ministral model failed: %s", e)
            import traceback
            traceback.print_exc()

    # Last-resort fallback
    return f"[FALLBACK] 回答: {question}" if question else "[FALLBACK] 回答: 是"


__all__ = ["load_model_and_tools", "generate_answer_for_sample"]
