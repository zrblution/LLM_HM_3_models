#!/usr/bin/env python3
"""Helpers for loading Ministral model and generating answers used by evaluation scripts.

Key fixes:
1) Robust checkpoint loading:
   - Some checkpoints are saved with an extra prefix like `_model.` in state_dict keys.
   - We detect and strip common wrapper prefixes before loading weights.

2) Robust dtype handling (bf16):
   - Processor returns image tensors in fp32 by default; bf16 model will crash without casting.
   - We cast floating tensors to model dtype before generate().

This version is adapted for Ministral multimodal model (vision + language).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import glob
import json
import logging
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
# halltrain 目录在 LLM_HM_3_models 下
HALLTRAIN_ROOT = PROJECT_ROOT.parent / "halltrain"

# Ensure local project roots are importable (for model.ministral_vl_model, integrations, etc.)
for p in (PROJECT_ROOT, HALLTRAIN_ROOT):
    if p.exists():
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

logger = logging.getLogger(__name__)


def _torch_load_state_dict(path: str) -> Dict[str, Any]:
    """Load a torch state_dict safely (weights only if supported)."""
    import torch

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint object type from {path}: {type(obj)}")


def _safetensors_load_state_dict(path: str) -> Dict[str, Any]:
    from safetensors.torch import load_file
    return load_file(path, device="cpu")


def _load_state_dict_from_model_dir(model_dir: str) -> Optional[Dict[str, Any]]:
    """Load a HF-style checkpoint from model_dir into a single state_dict."""
    model_dir = str(model_dir)
    if not os.path.isdir(model_dir):
        return None

    # 1) Prefer index.json if present (official sharded format)
    index_candidates = [
        os.path.join(model_dir, "model.safetensors.index.json"),
        os.path.join(model_dir, "pytorch_model.bin.index.json"),
        os.path.join(model_dir, "pytorch_model.safetensors.index.json"),
    ]
    for index_path in index_candidates:
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            shard_files = sorted({os.path.join(model_dir, fn) for fn in weight_map.values()})
            if not shard_files:
                continue
            sd: Dict[str, Any] = {}
            for sf in shard_files:
                if sf.endswith(".safetensors"):
                    part = _safetensors_load_state_dict(sf)
                else:
                    part = _torch_load_state_dict(sf)
                sd.update(part)
            return sd

    # 2) Single-file checkpoints
    single_candidates = [
        os.path.join(model_dir, "model.safetensors"),
        os.path.join(model_dir, "pytorch_model.safetensors"),
        os.path.join(model_dir, "pytorch_model.bin"),
    ]
    for p in single_candidates:
        if os.path.exists(p):
            if p.endswith(".safetensors"):
                return _safetensors_load_state_dict(p)
            return _torch_load_state_dict(p)

    # 3) Simple shards without index.json
    shard_patterns = [
        os.path.join(model_dir, "model-*.safetensors"),
        os.path.join(model_dir, "pytorch_model-*.bin"),
        os.path.join(model_dir, "pytorch_model-*.safetensors"),
    ]
    shard_files: list[str] = []
    for pat in shard_patterns:
        shard_files.extend(glob.glob(pat))
    shard_files = sorted(set(shard_files))
    if shard_files:
        sd = {}
        for sf in shard_files:
            if sf.endswith(".safetensors"):
                part = _safetensors_load_state_dict(sf)
            else:
                part = _torch_load_state_dict(sf)
            sd.update(part)
        return sd

    return None


def _strip_state_dict_prefixes(state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
    """Strip common wrapper prefixes from state_dict keys."""
    if not state_dict:
        return state_dict, []

    prefixes = [
        "_model.",            # your training wrapper
        "module.",            # DDP/DeepSpeed wrapper
        "_orig_mod.",         # torch.compile wrapper
        "base_model.model.",  # PEFT wrapper (common)
    ]

    stripped: list[str] = []
    sd = state_dict

    # Strip iteratively: sometimes we have nested wrappers like module._model.
    for pref in prefixes:
        keys = list(sd.keys())
        total = len(keys)
        if total == 0:
            break
        r = sum(1 for k in keys if k.startswith(pref)) / total
        if r >= 0.80:
            sd = {(k[len(pref):] if k.startswith(pref) else k): v for k, v in sd.items()}
            stripped.append(pref)

    # Also strip `_model.` even if not global (some checkpoints only wrap part of keys)
    if any(k.startswith("_model.") for k in sd.keys()) and "_model." not in stripped:
        sd = {(k[len("_model."):] if k.startswith("_model.") else k): v for k, v in sd.items()}
        stripped.append("_model.(partial)")

    return sd, stripped


def _remap_mistral3_vl_flat_keys(state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Some training code saves only the inner multi-modal module weights, with keys like:
      - language_model.model.layers....
      - language_model.lm_head.weight
      - vision_tower....
      - multi_modal_projector....

    HF Mistral3ForConditionalGeneration expects:
      - model.language_model.layers....
      - lm_head.weight
      - model.vision_tower....
      - model.multi_modal_projector....
    """
    if not state_dict:
        return state_dict, False

    keys = list(state_dict.keys())
    if any(k.startswith("model.") for k in keys):
        return state_dict, False

    has_language_model = any(k.startswith("language_model.") for k in keys)
    has_vision = any(k.startswith("vision_tower.") for k in keys)
    has_projector = any(k.startswith("multi_modal_projector.") for k in keys)
    if not (has_language_model and (has_vision or has_projector)):
        return state_dict, False

    remapped: Dict[str, Any] = {}

    def remap_key(k: str) -> str:
        if k.startswith("language_model.model."):
            return "model.language_model." + k[len("language_model.model.") :]
        if k.startswith("language_model.lm_head."):
            return "lm_head." + k[len("language_model.lm_head.") :]
        if k.startswith("vision_tower."):
            return "model.vision_tower." + k[len("vision_tower.") :]
        if k.startswith("multi_modal_projector."):
            return "model.multi_modal_projector." + k[len("multi_modal_projector.") :]
        return k

    changed = False
    for k, v in state_dict.items():
        nk = remap_key(k)
        if nk != k:
            changed = True
        remapped[nk] = v

    return remapped, changed


def _move_inputs_to_device_and_dtype(inputs: Dict[str, Any], device: str, dtype) -> Dict[str, Any]:
    """Move processor outputs to device and cast floating tensors to dtype."""
    import torch
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


def _load_state_dict_best_effort(model: Any, state_dict: Dict[str, Any]):
    """Load `state_dict` into the most compatible module (model vs model.model)."""
    import torch

    sd_keys = set(state_dict.keys())

    candidates: list[tuple[str, torch.nn.Module]] = []
    if isinstance(model, torch.nn.Module):
        candidates.append(("model", model))
    inner = getattr(model, "model", None)
    if isinstance(inner, torch.nn.Module):
        candidates.append(("model.model", inner))
    inner2 = getattr(model, "_model", None)
    if isinstance(inner2, torch.nn.Module):
        candidates.append(("model._model", inner2))

    if not candidates:
        raise TypeError("Provided model is not a torch.nn.Module")

    best_name: str = candidates[0][0]
    best_module: torch.nn.Module = candidates[0][1]
    best_overlap = -1
    for name, module in candidates:
        try:
            module_keys = set(module.state_dict().keys())
        except Exception:
            continue
        overlap = len(sd_keys & module_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
            best_module = module

    incompatible = best_module.load_state_dict(state_dict, strict=False)
    return incompatible, best_name


def load_model_and_tools(
    model_dir: str,
    device: str = "cuda",
    use_vcd: bool = False,
    use_inter: bool = False,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Load tokenizer, processor, and model from `model_dir`."""
    tokenizer = None
    processor = None
    model = None

    try:
        import torch
        from transformers import AutoTokenizer, AutoProcessor, AutoConfig
        from transformers import Mistral3ForConditionalGeneration
    except Exception as e:
        logger.warning("Required imports for Ministral loader not available: %s", e)
        return tokenizer, processor, model

    # Try to import custom model class (for fine-tuned models with injection modules)
    Qwen2_5_CustomVLForConditionalGeneration = None
    try:
        from model.ministral_vl_model import Qwen2_5_CustomVLForConditionalGeneration
        logger.info("Custom model class imported: Qwen2_5_CustomVLForConditionalGeneration")
    except Exception as e:
        logger.info("Custom model class not available from model.ministral_vl_model, trying alternative paths: %s", e)
        try:
            import importlib.util
            model_path = PROJECT_ROOT / "model" / "ministral_vl_model.py"
            if model_path.exists():
                spec = importlib.util.spec_from_file_location("ministral_vl_model", str(model_path))
                ministral_mod = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(ministral_mod)
                Qwen2_5_CustomVLForConditionalGeneration = ministral_mod.Qwen2_5_CustomVLForConditionalGeneration
                logger.info("Custom model class imported from %s", model_path)
        except Exception as e2:
            logger.warning("Failed to import custom model class from alternative path: %s", e2)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        logger.warning("Failed to load tokenizer from %s: %s", model_dir, e)
        tokenizer = None

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        logger.info("Processor loaded successfully.")
    except Exception as e:
        logger.warning("Could not load processor: %s", e)
        processor = None

    # Detect which class to use
    model_class = Mistral3ForConditionalGeneration
    try:
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_json = json.load(f)
            architectures = cfg_json.get("architectures", []) or []
            has_inject_op = cfg_json.get("inject_op") is not None
            is_custom_arch = "Qwen2_5_CustomVLForConditionalGeneration" in architectures
            if (is_custom_arch or has_inject_op) and (Qwen2_5_CustomVLForConditionalGeneration is not None):
                model_class = Qwen2_5_CustomVLForConditionalGeneration
                print("✅ Using custom model class: Qwen2_5_CustomVLForConditionalGeneration (Ministral)")
                print(f"   Detected: inject_op={cfg_json.get('inject_op')}, architectures={architectures}")
                print("   Injection modules will be loaded and used during inference!")
            elif (is_custom_arch or has_inject_op) and (Qwen2_5_CustomVLForConditionalGeneration is None):
                print("⚠️  Model config indicates custom architecture/injection, but custom class is not available.")
                print("   Falling back to standard Mistral3ForConditionalGeneration; injection modules will be IGNORED.")
    except Exception as e:
        logger.warning("Could not detect model architecture: %s", e)

    # Load config
    torch_dtype = torch.bfloat16
    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        logger.warning("AutoConfig.from_pretrained failed (%s). Will rely on from_pretrained.", e)
        config = None

    # Preload state_dict and strip key prefixes if needed
    state_dict = None
    stripped_prefixes: list[str] = []
    try:
        state_dict = _load_state_dict_from_model_dir(model_dir)
        if state_dict is not None:
            state_dict, stripped_prefixes = _strip_state_dict_prefixes(state_dict)
            state_dict, remapped = _remap_mistral3_vl_flat_keys(state_dict)
            if remapped:
                stripped_prefixes.append("mistral3_vl_flat_keys")
            if stripped_prefixes:
                logger.warning("Detected checkpoint key transforms: %s", stripped_prefixes)
    except Exception as e:
        logger.warning("Failed to pre-load checkpoint state_dict for key-fix: %s", e)
        state_dict = None

    # Build + load model
    try:
        # Custom Ministral wrapper (not a PreTrainedModel): build base model, patch layers, then load weights.
        if model_class is Qwen2_5_CustomVLForConditionalGeneration:
            if config is None:
                config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

            base_model = Mistral3ForConditionalGeneration(config)
            model = model_class(base_model)

            if state_dict is not None:
                incompatible, load_target = _load_state_dict_best_effort(model, state_dict)
                missing = getattr(incompatible, "missing_keys", [])
                unexpected = getattr(incompatible, "unexpected_keys", [])

                if missing:
                    logger.warning(
                        "State dict load (%s): %d missing keys (first 20): %s", load_target, len(missing), missing[:20]
                    )
                if unexpected:
                    logger.warning(
                        "State dict load (%s): %d unexpected keys (first 20): %s",
                        load_target,
                        len(unexpected),
                        unexpected[:20],
                    )
            else:
                # Fallback: let the wrapper handle loading from disk (best-effort).
                model = model_class.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                )

            model = model.to(device=device, dtype=torch_dtype)
            model.eval()
            logger.info("Loaded model via custom wrapper as %s", model_class.__name__)

        elif config is not None and state_dict is not None:
            # Instantiate model from config, then load remapped state_dict
            try:
                model = model_class(config)
            except Exception:
                if hasattr(model_class, "from_config"):
                    model = model_class.from_config(config)  # type: ignore[attr-defined]
                else:
                    raise

            incompatible, load_target = _load_state_dict_best_effort(model, state_dict)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])

            if missing:
                logger.warning(
                    "State dict load (%s): %d missing keys (first 20): %s", load_target, len(missing), missing[:20]
                )
            if unexpected:
                logger.warning(
                    "State dict load (%s): %d unexpected keys (first 20): %s",
                    load_target,
                    len(unexpected),
                    unexpected[:20],
                )
            if load_target != "model":
                logger.warning(
                    "Checkpoint keys did not match the top-level module; loaded weights into `%s` instead.", load_target
                )

            if hasattr(model, "tie_weights"):
                try:
                    model.tie_weights()
                except Exception:
                    pass

            model = model.to(device=device, dtype=torch_dtype)
            model.eval()
            logger.info("Loaded model via manual state_dict load as %s", model_class.__name__)
        else:
            model = model_class.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            model.to(device)
            model.eval()
            logger.info("Loaded model via from_pretrained as %s", model_class.__name__)

    except Exception as e:
        logger.warning("Failed to load Ministral model: %s", e)
        model = None

    # Optional wrappers (best-effort)
    if model is not None and use_vcd:
        try:
            from integrations.vcd import VCDModel  # type: ignore
            model = VCDModel(model)
            logger.info("Wrapped model with VCDModel")
        except Exception as e:
            logger.warning("use_vcd requested but VCDModel not available: %s", e)

    if model is not None and use_inter:
        try:
            from integrations.inter import INTERModel  # type: ignore
            model = INTERModel(model)
            logger.info("Wrapped model with INTERModel")
        except Exception as e:
            logger.warning("use_inter requested but INTERModel not available: %s", e)

    return tokenizer, processor, model


def prepare_inputs_for_sample(
    sample: dict,
    processor,
    tokenizer,
    image_root: str,
    device: str,
    dtype=None,
):
    """Prepare model inputs for a single sample (debug / non-batched use)."""
    from PIL import Image

    question = sample.get("question", "") or sample.get("text", "") or ""
    filename = sample.get("filename") or sample.get("img")
    visual_input = sample.get("visual_input") in [1, "1", True] or bool(filename)

    image_pil = None
    if visual_input and filename:
        rel_path = str(filename).lstrip("./")
        img_path = os.path.join(image_root, rel_path)
        if os.path.exists(img_path):
            try:
                image_pil = Image.open(img_path).convert("RGB")
            except Exception:
                image_pil = None

    messages = [{"role": "user", "content": []}]
    if image_pil is not None:
        messages[0]["content"].append({"type": "image", "image": image_pil})

    messages[0]["content"].append({"type": "text", "text": f"{question}\nPlease answer with yes or no."})

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if image_pil is not None:
        inputs = processor(text=[text_input], images=[image_pil], return_tensors="pt")
    else:
        inputs = processor(text=[text_input], return_tensors="pt")

    if dtype is None:
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    return _move_inputs_to_device_and_dtype(inputs, device=device, dtype=dtype)


def generate_answer_for_sample(
    sample: dict,
    tokenizer: Any,
    processor: Any,
    model: Any,
    device: str,
    image_root: str,
) -> str:
    """Generate a textual answer for `sample` (single-sample, mainly for debugging)."""
    import torch

    question = (sample.get("question") or sample.get("text") or "").strip()
    if not question:
        return "yes"

    if tokenizer is None or processor is None or model is None:
        return "yes"

    try:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        inputs = prepare_inputs_for_sample(
            sample=sample,
            processor=processor,
            tokenizer=tokenizer,
            image_root=image_root,
            device=str(model_device),
            dtype=model_dtype,
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                num_beams=1,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )

        # Decode only the generated continuation (avoid echoing the prompt)
        if "input_ids" in inputs:
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[:, prompt_len:]
        else:
            gen_ids = output_ids

        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return (text or "").strip() or "yes"

    except Exception as e:
        logger.warning("Generation failed: %s", e)
        return "yes"


__all__ = ["load_model_and_tools", "generate_answer_for_sample"]
