#!/usr/bin/env python3
"""
Compute per-layer utilization strength (u_t) statistics on POPE dataset for Qwen3-VL-2B.

This script computes the mean u_t for each of the 28 decoder layers across the entire
POPE test set. The u_t values are computed using masked mean over tokens (excluding padding),
then averaged across all samples.

Usage:
    python compute_u_stats_pope_qwen3vl.py \
        --dataset coco \
        --pope_json /path/to/coco_pope_random.json \
        --image_root /path/to/images \
        --model_name_or_path /path/to/qwen3-vl-2b-checkpoint \
        --output /path/to/output.json

Output JSON format:
{
    "n_layers": 28,
    "mean_u_per_layer": [0.xxx, 0.xxx, ...],  // 28 floats
    "count_per_layer": [N, N, ...],           // 28 ints, should all equal dataset size
    "aggregation_method": "masked_mean_over_tokens -> per_sample_scalar -> dataset_mean",
    "dataset": "coco",
    "total_samples": N,
    "model_path": "/path/to/model"
}
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add parent directory to path for model imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-layer u_t statistics on POPE dataset for Qwen3-VL-2B"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["coco", "gqa"],
        required=True,
        help="Dataset name: coco or gqa"
    )
    parser.add_argument(
        "--pope_json",
        type=str,
        required=True,
        help="Path to POPE annotation JSON file"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory containing images"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to Qwen3-VL-2B checkpoint"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (currently only batch_size=1 is fully supported)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path for statistics"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    return parser.parse_args()


def load_pope_data(pope_json: str) -> List[Dict[str, Any]]:
    """Load POPE dataset from JSON file."""
    with open(pope_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _parse_gate_layers(spec: str) -> Optional[List[int]]:
    """Parse gate_layers specification string."""
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return None  # None means all layers
    if spec == "none":
        return []
    # comma-separated ints
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def sync_config_to_model(model, config_keys: List[str]):
    """
    Sync experiment config keys from model.config to text_config and language_model.config.
    This is critical for decoder layers to read the correct configuration.
    
    Replicates the logic from training script's _sync_config_to_text_config.
    """
    # Sync to text_config
    if hasattr(model.config, 'text_config') and model.config.text_config is not None:
        for key in config_keys:
            if hasattr(model.config, key):
                setattr(model.config.text_config, key, getattr(model.config, key))
    
    # Sync to language_model.config
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        if hasattr(model.model.language_model, 'config'):
            for key in config_keys:
                if hasattr(model.config, key):
                    setattr(model.model.language_model.config, key, getattr(model.config, key))


def load_model_and_processor(model_path: str, device: str):
    """
    Load Qwen3-VL model with custom injection modules and processor.
    
    Returns:
        model, processor, tokenizer
    """
    from transformers import AutoTokenizer, AutoProcessor
    
    # Try to import custom model class
    try:
        from model.qwen_vl_model import Qwen2_5_CustomVLForConditionalGeneration
        print("✅ Custom model class imported successfully")
    except ImportError as e:
        raise ImportError(
            f"Failed to import custom model class. Make sure model/qwen_vl_model.py exists: {e}"
        )
    
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load model
    model = Qwen2_5_CustomVLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # ===== Critical: Set up config for u_t computation =====
    # These settings ensure that:
    # 1. Evidence modules are enabled (enable_vision_gate=True)
    # 2. All layers compute u_t (gate_layers=None means all)
    # 3. Utilization strength is computed (use_utilization=True)
    
    model.config.enable_vision_gate = True
    model.config.gate_layers = None  # None = all layers
    model.config.inject_position = "per_layer"
    model.config.inject_op = "ours"
    model.config.use_utilization = True
    model.config.evidence_source = "aligned"
    
    # Config keys that need to be synced
    config_keys = [
        "enable_vision_gate",
        "gate_layers",
        "inject_position",
        "inject_op",
        "use_utilization",
        "evidence_source",
    ]
    
    # Sync config to text_config and language_model.config
    sync_config_to_model(model, config_keys)
    
    print(f"Config synced: enable_vision_gate={model.config.enable_vision_gate}, "
          f"gate_layers={model.config.gate_layers}, use_utilization={model.config.use_utilization}")
    
    model.to(device)
    model.eval()
    
    # Get number of layers
    n_layers = model.config.text_config.num_hidden_layers
    print(f"Model loaded with {n_layers} decoder layers")
    
    return model, processor, tokenizer, n_layers


def build_qwen_inputs(
    sample: Dict[str, Any],
    processor,
    image_root: str,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Build model inputs for a single POPE sample.
    
    Uses the same prompt format as the existing POPE evaluation scripts:
    - System prompt: "You are a helpful assistant."
    - User prompt: question text with image
    
    Args:
        sample: POPE sample dict with keys: id, img, text, labels
        processor: Qwen processor
        image_root: Root directory for images
        device: Target device
    
    Returns:
        Dict with input tensors ready for model forward
    """
    # Get image path and question
    img_filename = sample.get("img") or sample.get("filename")
    question = sample.get("text") or sample.get("question", "")
    
    # Load image
    img_path = os.path.join(image_root, img_filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    image = Image.open(img_path).convert("RGB")
    
    # Build messages in Qwen format (same as convert_output.py)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    return inputs


class AuxCaptureHook:
    """
    Hook to capture aux outputs from CustomQwen3VLTextModel.forward().
    
    Since Qwen3VLModel.forward() doesn't propagate aux to its return value,
    we use a forward hook to capture it directly from the language_model.
    """
    def __init__(self):
        self.aux = None
    
    def __call__(self, module, input, output):
        # output is BaseModelOutputWithPast with dynamically added aux attribute
        if hasattr(output, 'aux'):
            self.aux = output.aux
        else:
            self.aux = None


def compute_masked_mean_u(
    aux_list: List[Dict[str, Any]],
    attention_mask: torch.Tensor
) -> List[float]:
    """
    Compute masked mean of u_t for each layer.
    
    Args:
        aux_list: List of aux dicts from each layer, each containing 'u' tensor of shape [B, T, 1]
        attention_mask: Attention mask of shape [B, T], 1 for valid tokens, 0 for padding
    
    Returns:
        List of per-layer mean u values (one float per layer)
    """
    per_layer_means = []
    
    for aux in aux_list:
        if aux is None or 'u' not in aux:
            # Layer didn't compute u (shouldn't happen if config is correct)
            per_layer_means.append(None)
            continue
        
        u = aux['u']  # Shape: [B, T, 1]
        
        # Squeeze the last dimension
        u = u.squeeze(-1)  # Shape: [B, T]
        
        # Apply mask: only consider non-padding tokens
        # attention_mask: [B, T], 1 for valid, 0 for padding
        mask = attention_mask.to(u.device).float()
        
        # Masked mean over tokens
        # Sum of u values for valid tokens / count of valid tokens
        masked_u = u * mask
        sum_u = masked_u.sum(dim=-1)  # [B]
        count = mask.sum(dim=-1)  # [B]
        
        # Avoid division by zero
        count = count.clamp(min=1)
        
        # Per-sample mean
        sample_mean = sum_u / count  # [B]
        
        # Mean across batch (for batch_size=1, this is just the single value)
        layer_mean = sample_mean.mean().item()
        
        per_layer_means.append(layer_mean)
    
    return per_layer_means


def main():
    args = parse_args()
    
    print(f"=== Compute u_t Statistics for Qwen3-VL on POPE ===")
    print(f"Dataset: {args.dataset}")
    print(f"POPE JSON: {args.pope_json}")
    print(f"Image root: {args.image_root}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Output: {args.output}")
    print()
    
    # Load POPE data
    print("Loading POPE data...")
    data = load_pope_data(args.pope_json)
    total_samples = len(data)
    print(f"Loaded {total_samples} samples")
    
    # Apply max_samples limit
    if args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Limited to {len(data)} samples (--max_samples={args.max_samples})")
    
    # Load model
    model, processor, tokenizer, n_layers = load_model_and_processor(
        args.model_name_or_path, args.device
    )
    
    print(f"\nExpected n_layers: 28, Actual: {n_layers}")
    if n_layers != 28:
        print(f"⚠️  Warning: Expected 28 layers for Qwen3-VL-2B, got {n_layers}")
    
    # Register hook to capture aux from language_model
    hook = AuxCaptureHook()
    hook_handle = model.model.language_model.register_forward_hook(hook)
    
    # Initialize accumulators for per-layer statistics
    # We accumulate sum and count separately to compute mean at the end
    layer_u_sums = [0.0] * n_layers
    layer_counts = [0] * n_layers
    
    # Process samples
    print(f"\nProcessing {len(data)} samples...")
    
    with torch.no_grad():
        for sample in tqdm(data, desc="Computing u_t stats"):
            try:
                # Build inputs
                inputs = build_qwen_inputs(sample, processor, args.image_root, args.device)
                
                # Get attention mask for masked mean computation
                attention_mask = inputs.get('attention_mask')
                
                # Forward pass (not generate, we just need one forward to get aux)
                # We don't need the output, just the aux captured by the hook
                _ = model(**inputs, use_cache=False, return_dict=True)
                
                # Get captured aux
                aux_list = hook.aux
                
                if aux_list is None or len(aux_list) == 0:
                    print(f"⚠️  Warning: No aux captured for sample {sample.get('id', '?')}")
                    continue
                
                # Compute per-layer masked mean u
                per_layer_means = compute_masked_mean_u(aux_list, attention_mask)
                
                # Accumulate
                for layer_idx, mean_u in enumerate(per_layer_means):
                    if mean_u is not None:
                        layer_u_sums[layer_idx] += mean_u
                        layer_counts[layer_idx] += 1
                
            except Exception as e:
                print(f"⚠️  Error processing sample {sample.get('id', '?')}: {e}")
                continue
    
    # Remove hook
    hook_handle.remove()
    
    # Compute final per-layer means
    mean_u_per_layer = []
    for layer_idx in range(n_layers):
        if layer_counts[layer_idx] > 0:
            mean_u_per_layer.append(layer_u_sums[layer_idx] / layer_counts[layer_idx])
        else:
            mean_u_per_layer.append(None)
    
    # Prepare output
    output_data = {
        "n_layers": n_layers,
        "mean_u_per_layer": mean_u_per_layer,
        "count_per_layer": layer_counts,
        "aggregation_method": "masked_mean_over_tokens -> per_sample_scalar -> dataset_mean",
        "dataset": args.dataset,
        "pope_json": args.pope_json,
        "total_samples": len(data),
        "model_path": args.model_name_or_path,
    }
    
    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Results ===")
    print(f"n_layers: {n_layers}")
    print(f"Samples processed per layer: {layer_counts}")
    print(f"\nPer-layer mean u_t:")
    for i, mean_u in enumerate(mean_u_per_layer):
        if mean_u is not None:
            print(f"  Layer {i:2d}: {mean_u:.6f}")
        else:
            print(f"  Layer {i:2d}: N/A (no data)")
    
    print(f"\nResults saved to {args.output}")
    
    # Validation
    print(f"\n=== Validation ===")
    print(f"✓ n_layers == 28: {n_layers == 28}")
    print(f"✓ len(mean_u_per_layer) == 28: {len(mean_u_per_layer) == 28}")
    all_counts_equal = all(c == layer_counts[0] for c in layer_counts if c > 0)
    print(f"✓ All layer counts equal: {all_counts_equal}")
    if all_counts_equal and layer_counts[0] > 0:
        print(f"  (Each layer processed {layer_counts[0]} samples)")


if __name__ == "__main__":
    main()
