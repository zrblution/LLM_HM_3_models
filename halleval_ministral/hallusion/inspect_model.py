#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _ensure_import_paths():
    base_dir = Path(__file__).resolve().parent  # .../halleval_ministral/hallusion
    project_root = base_dir.parent              # .../halleval_ministral
    for p in (project_root, base_dir):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _find_decoder_layers(model: Any):
    import torch.nn as nn

    candidates = [
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("decoder", "layers"),
        ("transformer", "h"),
        ("transformer", "layers"),
    ]

    for chain in candidates:
        cur: Any = model
        ok = True
        for name in chain:
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok and isinstance(cur, nn.ModuleList):
            return cur, ".".join(chain)

    return None, None


def _iter_weight_files(model_dir: str) -> List[str]:
    model_dir = str(model_dir)
    if not os.path.isdir(model_dir):
        return []

    # Prefer HF index.json (sharded) if present
    index_candidates = [
        os.path.join(model_dir, "model.safetensors.index.json"),
        os.path.join(model_dir, "pytorch_model.safetensors.index.json"),
        os.path.join(model_dir, "pytorch_model.bin.index.json"),
    ]
    for idx in index_candidates:
        if os.path.exists(idx):
            import json

            with open(idx, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {}) or {}
            shard_files = sorted({os.path.join(model_dir, fn) for fn in weight_map.values()})
            return [p for p in shard_files if os.path.exists(p)]

    # Single-file
    single_candidates = [
        os.path.join(model_dir, "model.safetensors"),
        os.path.join(model_dir, "pytorch_model.safetensors"),
        os.path.join(model_dir, "pytorch_model.bin"),
    ]
    for p in single_candidates:
        if os.path.exists(p):
            return [p]

    # Fallback patterns
    out: List[str] = []
    for pat in ("model-*.safetensors", "pytorch_model-*.safetensors", "pytorch_model-*.bin"):
        out.extend(sorted(Path(model_dir).glob(pat)))
    return [str(p) for p in out]


def _list_checkpoint_keys(model_dir: str) -> Set[str]:
    weight_files = _iter_weight_files(model_dir)
    if not weight_files:
        return set()

    keys: Set[str] = set()
    for wf in weight_files:
        if wf.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(wf, framework="pt", device="cpu") as f:
                keys.update(list(f.keys()))
        else:
            import torch

            obj = torch.load(wf, map_location="cpu", weights_only=True)
            if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
                obj = obj["state_dict"]
            if isinstance(obj, dict):
                keys.update(list(obj.keys()))
    return keys


def _normalize_checkpoint_keys(keys: Iterable[str], run_inf) -> Set[str]:
    # Reuse the exact normalization logic from hallusion/run_inference.py
    dummy: Dict[str, Any] = {k: None for k in keys}
    dummy, _ = run_inf._strip_state_dict_prefixes(dummy)  # type: ignore[attr-defined]
    dummy, _ = run_inf._remap_mistral3_vl_flat_keys(dummy)  # type: ignore[attr-defined]
    return set(dummy.keys())


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    _ensure_import_paths()

    p = argparse.ArgumentParser(description="Inspect Ministral checkpoint + architecture for (mis)match.")
    p.add_argument("--model_dir", required=True, help="Path to model directory (with config.json + weights)")
    p.add_argument("--device", default=None, help="Device for loading model (e.g. cuda:0, cpu). Default: cuda:0 if available.")
    p.add_argument("--max_layer_print", type=int, default=3, help="How many decoder layers to print (default: 3)")
    p.add_argument("--check_keys", action="store_true", help="Compare checkpoint keys vs model.state_dict keys")
    args = p.parse_args()

    import torch
    import transformers

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    run_inf = importlib.import_module("hallusion.run_inference")

    _print_header("Environment")
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device_count:", torch.cuda.device_count())
        print("selected device:", device)

    _print_header("Load Model")
    tokenizer, processor, model = run_inf.load_model_and_tools(args.model_dir, device=device)
    if model is None:
        raise SystemExit("Failed to load model (model=None)")

    base = getattr(model, "_model", model)
    print("model class:", type(model))
    if base is not model:
        print("base  class:", type(base))

    cfg = getattr(model, "config", None) or getattr(base, "config", None)
    if cfg is not None:
        keys = [
            "model_type",
            "architectures",
            "enable_vision_gate",
            "inject_position",
            "inject_op",
            "use_utilization",
            "evidence_source",
            "gate_layers",
        ]
        print("\nconfig (selected):")
        for k in keys:
            if hasattr(cfg, k):
                print(f"  {k}: {getattr(cfg, k)}")

    layers, layers_path = _find_decoder_layers(base)
    if layers is None:
        print("\ndecoder layers: <NOT FOUND>")
    else:
        print(f"\ndecoder layers: {len(layers)} (path={layers_path})")
        n = min(len(layers), max(0, int(args.max_layer_print)))
        for i in range(n):
            layer = layers[i]
            flags = {
                "retriever": hasattr(layer, "retriever"),
                "analyzer": hasattr(layer, "analyzer"),
                "util": hasattr(layer, "util"),
                "corrector": hasattr(layer, "corrector"),
                "concat_proj": hasattr(layer, "concat_proj"),
            }
            present = [k for k, v in flags.items() if v]
            print(f"  layer[{i}] injected: {present if present else 'NO'}")

    _print_header("Param Counts")
    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print("total params:", total)
    print("trainable params:", trainable)

    # Rough heuristic for injected modules
    injected = 0
    injected_names = ("retriever", "analyzer", "util", "corrector", "concat_proj")
    for name, param in base.named_parameters():
        if any(f".{n}." in name for n in injected_names):
            injected += param.numel()
    if injected:
        print("injected params:", injected)

    if args.check_keys:
        _print_header("Key Match Check")
        ckpt_keys = _list_checkpoint_keys(args.model_dir)
        print("checkpoint keys:", len(ckpt_keys))
        if not ckpt_keys:
            print("No checkpoint keys found.")
            return

        ckpt_norm = _normalize_checkpoint_keys(ckpt_keys, run_inf)
        model_keys = set(base.state_dict().keys())

        missing_in_model = sorted(ckpt_norm - model_keys)
        missing_in_ckpt = sorted(model_keys - ckpt_norm)

        print("normalized checkpoint keys:", len(ckpt_norm))
        print("model state_dict keys:", len(model_keys))
        print("in checkpoint but not in model:", len(missing_in_model))
        print("in model but not in checkpoint:", len(missing_in_ckpt))

        if missing_in_model:
            print("\nfirst 30 checkpoint-only keys:")
            for k in missing_in_model[:30]:
                print("  ", k)
        if missing_in_ckpt:
            print("\nfirst 30 model-only keys:")
            for k in missing_in_ckpt[:30]:
                print("  ", k)


if __name__ == "__main__":
    main()

