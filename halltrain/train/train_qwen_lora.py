"""
Qwen3-VL LoRA 微调训练脚本
基于 train_qwen.py，添加 LoRA 支持以减少显存占用
"""
import torch
from datasets import Dataset
from dataclasses import dataclass, field
from transformers import AutoTokenizer, HfArgumentParser
from qwen_vl_utils import process_vision_info
from typing import Optional
from transformers import (
    TrainingArguments,
    Trainer,
    AutoProcessor
)
from PIL import Image
import json
import os
from model.qwen_vl_model import Qwen2_5_CustomVLForConditionalGeneration
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict
from transformers import PreTrainedTokenizerBase
from datetime import timedelta

# LoRA 相关导入
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./basemodel")
    lambda_orth: float = field(default=0.0, metadata={"help": "Weight for L_orth regularizer."})
    lambda_ctr: float = field(default=0.0, metadata={"help": "Weight for L_ctr regularizer."})
    tau: float = field(default=0.07, metadata={"help": "Temperature for contrastive loss."})
    aux_layers: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated layer indices to regularize; None means all layers returned."},
    )

    # ===== Experiment control (must be CLI-driven) =====
    enable_evidence: bool = field(
        default=True,
        metadata={"help": "Enable evidence modules (retrieval/analysis/util/correction). False -> Base."},
    )

    # Layer selection for per-layer injection: "all" / "none" / "0,1,2"
    gate_layers: str = field(
        default="all",
        metadata={"help": "Which decoder layers apply evidence injection: all|none|comma-separated indices."},
    )

    # Injection position: per_layer -> +M+A/Ours; first_layer_input -> +M; none -> Base
    inject_position: str = field(
        default="per_layer",
        metadata={"help": "Evidence injection position: none|first_layer_input|per_layer."},
    )

    # Injection operator for per-layer (and first-layer input): ours|add|concat
    inject_op: str = field(
        default="ours",
        metadata={"help": "Evidence injection operator: ours|add|concat."},
    )

    # Whether to use utilization strength u (sigmoid MLP). +M requires False.
    use_utilization: bool = field(
        default=True,
        metadata={"help": "Use utilization strength u. False -> unweighted injection."},
    )

    # Evidence source for injection: candidate(e) or aligned(a)
    evidence_source: str = field(
        default="aligned",
        metadata={"help": "Which evidence vector to inject: candidate|aligned."},
    )

    # Export case-study stats (layer-wise mean u)
    export_u_stats: bool = field(
        default=False,
        metadata={"help": "Export layer-wise mean utilization strength u during training."},
    )
    export_u_stats_path: Optional[str] = field(
        default=None,
        metadata={"help": "Output path (json) for exported u stats. Default: <output_dir>/u_stats.json"},
    )

    # ===== LoRA 参数 =====
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension (rank)."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter for scaling."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability."},
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA. Default targets attention and MLP layers."},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias type: none, all, or lora_only."},
    )


@dataclass
class DataArguments:
    training_data_path: str = field(default=None,
                                    metadata={"help": "Path to the training data."})
    training_image_dir: str = field(default=None,
                                    metadata={"help": "Path to the image directory."})


class MultiModalCollator:
    """实时处理图像的 Collator，避免预保存大量 pixel_values"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, processor, data_args: DataArguments):
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = data_args

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Separate out text features that we want to tokenize/pad:
        text_features = []
        for i, f in enumerate(features):
            if not isinstance(f.get("input_ids"), list):
                print(f"[DEBUG] feature index={i} has non-list input_ids:", f["input_ids"])
            if not isinstance(f.get("attention_mask"), list):
                print(f"[DEBUG] feature index={i} has non-list attention_mask:", f["attention_mask"])
            if not isinstance(f.get("labels"), list):
                print(f"[DEBUG] feature index={i} has non-list labels:", f["labels"])

            text_features.append({
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
                "labels": f["labels"]
            })

        # 2. 用 try/except 捕获 tokenizer.pad(...) 的报错
        try:
            batch_text = self.tokenizer.pad(
                text_features,
                padding=True,
                return_tensors="pt"
            )
        except Exception as e:
            print("\n[ERROR] tokenizer.pad(...) failed. Below is the text_features content:\n")
            for i, tf in enumerate(text_features):
                print(f"  === Sample {i} ===")
                print("  input_ids:", tf["input_ids"])
                print("  attention_mask:", tf["attention_mask"])
                print("  labels:", tf["labels"])
                print("  ----------------")
            raise e

        # 3. 实时处理图像：从图片路径加载并处理图像
        pixel_values_list = []
        image_grid_thw_list = []
        
        for f in features:
            img_path = os.path.join(self.data_args.training_image_dir, f["img"])
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            image_pil = Image.open(img_path).convert("RGB")
            fixed_size = (224, 224)
            image_pil = image_pil.resize(fixed_size, Image.BICUBIC)
            
            user_text = f["user_text"]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_pil,
                            "resize_height": 224,
                            "resize_width": 224,
                        },
                        {
                            "type": "text",
                            "text": user_text
                        },
                    ],
                },
            ]
            
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                do_resize=True,
                padding=True,
            )
            
            pv = inputs["pixel_values"]
            if pv.dim() == 4 and pv.shape[0] == 1:
                pv = pv.squeeze(0)
            pixel_values_list.append(pv)
            
            gthw = inputs["image_grid_thw"]
            if gthw.dim() == 2 and gthw.shape[0] == 1:
                gthw = gthw.squeeze(0)
            elif gthw.dim() == 1:
                pass
            else:
                gthw = gthw.squeeze()
            image_grid_thw_list.append(gthw)

        # 4. Stack 图像数据
        pixel_values = torch.stack(pixel_values_list, dim=0)
        image_grid_thw = torch.stack(image_grid_thw_list, dim=0)

        # 5. Merge
        batch = {
            "input_ids": batch_text["input_ids"],
            "attention_mask": batch_text["attention_mask"],
            "labels": batch_text["labels"],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

        return batch


def process_func(example, data_args, tokenizer, processor, max_length=32000):
    """
    轻量级预处理：只保存图片路径和文本信息，不处理图像。
    """
    user_text = example["text"] + '\n'
    label_text = example["labels"]
    
    img_path = os.path.join(data_args.training_image_dir, example["img"])
    image_pil = Image.open(img_path).convert("RGB")
    fixed_size = (224, 224)
    image_pil = image_pil.resize(fixed_size, Image.BICUBIC)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_pil,
                    "resize_height": 224,
                    "resize_width": 224,
                },
                {
                    "type": "text",
                    "text": user_text
                },
            ],
        },
    ]
    
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        do_resize=True,
        padding=True,
    )
    text_input_ids = inputs["input_ids"][0].tolist()
    
    response = tokenizer([label_text], add_special_tokens=False)

    input_ids = text_input_ids + response["input_ids"][0] + [tokenizer.pad_token_id]
    attention_mask = [1] * len(input_ids)
    labels = (
            [-100] * len(text_input_ids)
            + response["input_ids"][0]
            + [tokenizer.pad_token_id]
    )

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "img": example["img"],
        "user_text": user_text,
    }


def print_trainable_parameters(model):
    """打印可训练参数数量"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}%"
    )


parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Attach evidence-regularization hyperparameters to training_args
training_args.lambda_orth = getattr(model_args, "lambda_orth", 0.0)
training_args.lambda_ctr = getattr(model_args, "lambda_ctr", 0.0)
training_args.tau = getattr(model_args, "tau", 0.07)
training_args.aux_layers = getattr(model_args, "aux_layers", None)

# 检查是否在分布式环境中
import torch.distributed as dist
is_distributed = dist.is_initialized() if hasattr(dist, 'is_initialized') else False
local_rank = int(os.environ.get('LOCAL_RANK', -1))
rank = int(os.environ.get('RANK', -1)) if is_distributed else 0

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

# 预处理数据缓存路径
cache_dir = os.path.join(training_args.output_dir, "preprocessed_cache")
os.makedirs(cache_dir, exist_ok=True)
done_file = os.path.join(cache_dir, ".preprocessing_done")

cache_exists = os.path.exists(done_file) and os.path.exists(os.path.join(cache_dir, "dataset_info.json"))

# 数据预处理逻辑（与原版相同）
if is_distributed:
    if rank == 0:
        if cache_exists:
            print(f"[Rank {rank}] Preprocessed cache found, loading from {cache_dir}...")
            train_dataset = Dataset.load_from_disk(cache_dir)
            print(f"[Rank {rank}] Dataset loaded from cache.")
            dist.barrier()
        else:
            print(f"[Rank {rank}] No cache found, starting data preprocessing...")
            with open(data_args.training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            train_ds = Dataset.from_list(training_data)
            
            import threading
            import time
            heartbeat_interval = 30
            heartbeat_stop = threading.Event()
            
            def heartbeat_worker():
                while not heartbeat_stop.is_set():
                    try:
                        dummy_tensor = torch.tensor([1.0], device='cpu')
                        dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM, async_op=False)
                        time.sleep(heartbeat_interval)
                    except Exception as e:
                        time.sleep(heartbeat_interval)
            
            heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
            heartbeat_thread.start()
            
            try:
                train_dataset = train_ds.map(
                    lambda ex: process_func(ex, data_args, tokenizer, processor),
                    num_proc=1,
                    desc="Preprocessing dataset"
                )
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=5)
            
            print(f"[Rank {rank}] Saving preprocessed dataset to {cache_dir}...")
            train_dataset.save_to_disk(cache_dir)
            del train_ds
            import gc
            gc.collect()
            
            with open(done_file, 'w') as f:
                f.write(str(time.time()))
            print(f"[Rank {rank}] Preprocessing completed and saved.")
            
            dist.barrier()
    else:
        if cache_exists:
            print(f"[Rank {rank}] Preprocessed cache found, loading from {cache_dir}...")
            dist.barrier()
            train_dataset = Dataset.load_from_disk(cache_dir)
            print(f"[Rank {rank}] Dataset loaded from cache.")
        else:
            print(f"[Rank {rank}] Waiting for rank 0 to finish preprocessing...")
            import time
            max_wait_time = 3600 * 2
            start_time = time.time()
            check_interval = 30
            
            while True:
                try:
                    if os.path.exists(done_file):
                        break
                    
                    if time.time() - start_time > max_wait_time:
                        raise RuntimeError(f"Timeout waiting for rank 0 to finish preprocessing (waited {max_wait_time}s)")
                    
                    try:
                        dummy_tensor = torch.tensor([1.0], device='cpu')
                        dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM, async_op=False)
                    except:
                        pass
                    
                    time.sleep(check_interval)
                except Exception as e:
                    if "barrier" in str(e).lower() or "timeout" in str(e).lower():
                        if time.time() - start_time > max_wait_time:
                            raise
                        time.sleep(check_interval)
                    else:
                        raise
            
            dist.barrier()
            
            print(f"[Rank {rank}] Loading preprocessed dataset from {cache_dir}...")
            train_dataset = Dataset.load_from_disk(cache_dir)
            print(f"[Rank {rank}] Dataset loaded.")
else:
    if cache_exists:
        print(f"Preprocessed cache found, loading from {cache_dir}...")
        train_dataset = Dataset.load_from_disk(cache_dir)
        print("Dataset loaded from cache.")
    else:
        print("No cache found, starting data preprocessing...")
        with open(data_args.training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        train_ds = Dataset.from_list(training_data)
        
        train_dataset = train_ds.map(
            lambda ex: process_func(ex, data_args, tokenizer, processor),
            num_proc=1,
            desc="Preprocessing dataset"
        )
        
        print(f"Saving preprocessed dataset to {cache_dir}...")
        train_dataset.save_to_disk(cache_dir)
        import time
        with open(done_file, 'w') as f:
            f.write(str(time.time()))
        print("Preprocessing completed and saved.")

# 加载基础模型
print("Loading base model...")
model = Qwen2_5_CustomVLForConditionalGeneration.from_pretrained(
    model_args.model_name_or_path, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 使用 bf16 减少显存
)

def _parse_gate_layers(spec: str):
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return None
    if spec == "none":
        return []
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(int(x))
    return out

# 设置模型配置
model.config.enable_vision_gate = bool(getattr(model_args, "enable_evidence", True))
model.config.gate_layers = _parse_gate_layers(getattr(model_args, "gate_layers", "all"))
model.config.inject_position = getattr(model_args, "inject_position", "per_layer").strip().lower()
model.config.inject_op = getattr(model_args, "inject_op", "ours").strip().lower()
model.config.use_utilization = bool(getattr(model_args, "use_utilization", True))
model.config.evidence_source = getattr(model_args, "evidence_source", "aligned").strip().lower()
model.config.export_u_stats = bool(getattr(model_args, "export_u_stats", False))
model.config.export_u_stats_path = getattr(model_args, "export_u_stats_path", None)

# 验证配置
valid_pos = {"none", "first_layer_input", "per_layer"}
valid_op = {"ours", "add", "concat"}
valid_src = {"candidate", "aligned"}

if model.config.inject_position not in valid_pos:
    raise ValueError(f"inject_position must be one of {valid_pos}, got {model.config.inject_position}")
if model.config.inject_op not in valid_op:
    raise ValueError(f"inject_op must be one of {valid_op}, got {model.config.inject_op}")
if model.config.evidence_source not in valid_src:
    raise ValueError(f"evidence_source must be one of {valid_src}, got {model.config.evidence_source}")

# ===== 配置 LoRA =====
print("Configuring LoRA...")

# 解析 target_modules
target_modules = model_args.lora_target_modules.split(",") if model_args.lora_target_modules else None
target_modules = [m.strip() for m in target_modules] if target_modules else None

lora_config = LoraConfig(
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    target_modules=target_modules,
    lora_dropout=model_args.lora_dropout,
    bias=model_args.lora_bias,
    task_type=TaskType.CAUSAL_LM,
    # 对于自定义模型，可能需要指定 modules_to_save 来保存 Evidence 模块
    modules_to_save=["retriever", "analyzer", "util", "corrector", "concat_proj"],
)

print(f"LoRA config: r={model_args.lora_r}, alpha={model_args.lora_alpha}, "
      f"dropout={model_args.lora_dropout}, target_modules={target_modules}")

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 打印可训练参数信息
print("\n" + "="*50)
print("LoRA Model Parameters:")
print_trainable_parameters(model)
print("="*50 + "\n")

# 创建 collator
data_collator = MultiModalCollator(tokenizer=tokenizer, processor=processor, data_args=data_args)

if not hasattr(training_args, 'dataloader_num_workers') or training_args.dataloader_num_workers is None:
    training_args.dataloader_num_workers = 0


class EvidenceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        aux = getattr(outputs, "aux", None)
        if aux is None:
            return (loss, outputs) if return_outputs else loss

        aux_layers = None
        if getattr(self.args, "aux_layers", None) is not None:
            try:
                aux_layers = set(
                    int(x.strip()) for x in self.args.aux_layers.split(",") if x.strip() != ""
                )
            except Exception:
                aux_layers = None

        # L_orth
        lambda_orth = getattr(self.args, "lambda_orth", 0.0)
        if lambda_orth and lambda_orth > 0:
            l_orth = 0.0
            count = 0
            for item in aux:
                if aux_layers is not None and item.get("layer_idx", None) not in aux_layers:
                    continue
                a = item.get("a", None)
                r = item.get("r", None)
                if a is None or r is None:
                    continue

                eps = 1e-6
                num = (a * r).sum(dim=-1)
                den = (a.norm(p=2, dim=-1) * r.norm(p=2, dim=-1) + eps)
                cos = (num / den).abs().mean()

                l_orth = l_orth + cos
                count += 1

            if count > 0:
                loss = loss + lambda_orth * (l_orth / count)

        # L_ctr
        lambda_ctr = getattr(self.args, "lambda_ctr", 0.0)
        tau = getattr(self.args, "tau", 0.07)
        if lambda_ctr and lambda_ctr > 0:
            pooled = []
            for item in aux:
                if aux_layers is not None and item.get("layer_idx", None) not in aux_layers:
                    continue
                a = item.get("a", None)
                if a is None:
                    continue
                pooled.append(a.mean(dim=1))

            if len(pooled) > 0:
                z = torch.stack(pooled, dim=0).mean(dim=0)
                z = torch.nn.functional.normalize(z, p=2, dim=-1)

                sim = torch.matmul(z, z.transpose(0, 1)) / tau
                labels = torch.arange(sim.size(0), device=sim.device)
                l_ctr = torch.nn.functional.cross_entropy(sim, labels)

                loss = loss + lambda_ctr * l_ctr

        return (loss, outputs) if return_outputs else loss


trainer = EvidenceTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# 开启模型训练
print("Starting LoRA training...")
trainer.train()

# 保存 LoRA 适配器权重
print(f"Saving LoRA adapter to {training_args.output_dir}...")
model.save_pretrained(training_args.output_dir)

# 保存 tokenizer 和 processor
tokenizer.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

# 保存配置信息
config_info = {
    "base_model": model_args.model_name_or_path,
    "lora_r": model_args.lora_r,
    "lora_alpha": model_args.lora_alpha,
    "lora_dropout": model_args.lora_dropout,
    "lora_target_modules": target_modules,
    "lora_bias": model_args.lora_bias,
}
import json as json_module
with open(os.path.join(training_args.output_dir, "lora_config_info.json"), "w") as f:
    json_module.dump(config_info, f, indent=2)

print("LoRA training completed!")
