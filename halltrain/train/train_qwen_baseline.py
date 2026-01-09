"""
Qwen3-VL 原始模型基线训练脚本
不添加任何新模块，直接使用原始 Qwen3VLForConditionalGeneration 进行微调
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
    AutoProcessor,
    Qwen3VLForConditionalGeneration  # 直接使用原始模型
)
from PIL import Image
import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict
from transformers import PreTrainedTokenizerBase
from datetime import timedelta

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./basemodel")
    # 以下参数与 train_qwen.py 保持一致，基线模型不使用但保留以兼容命令行
    lambda_orth: float = field(default=0.0, metadata={"help": "Weight for L_orth regularizer."})
    lambda_ctr: float = field(default=0.0, metadata={"help": "Weight for L_ctr regularizer."})
    tau: float = field(default=0.07, metadata={"help": "Temperature for contrastive loss."})
    aux_layers: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated layer indices to regularize; None means all layers returned."},
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


parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Attach evidence-regularization hyperparameters to training_args so the custom Trainer can access them
# 与 train_qwen.py 保持一致
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

# 预处理数据缓存路径 - 使用不同的缓存目录避免与自定义模型冲突
cache_dir = os.path.join(training_args.output_dir, "preprocessed_cache_baseline")
os.makedirs(cache_dir, exist_ok=True)
done_file = os.path.join(cache_dir, ".preprocessing_done")

cache_exists = os.path.exists(done_file) and os.path.exists(os.path.join(cache_dir, "dataset_info.json"))

# 只在主进程（rank 0）进行预处理，其他进程等待并加载缓存
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

# ========== 关键修改：直接使用原始 Qwen3VLForConditionalGeneration ==========
print("Loading original Qwen3VLForConditionalGeneration model (no custom modules)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_args.model_name_or_path, 
    trust_remote_code=True
)
print("Model loaded successfully!")

# 创建 collator
data_collator = MultiModalCollator(tokenizer=tokenizer, processor=processor, data_args=data_args)

if not hasattr(training_args, 'dataloader_num_workers') or training_args.dataloader_num_workers is None:
    training_args.dataloader_num_workers = 0


class EvidenceTrainer(Trainer):
    """
    自定义 Trainer，与 train_qwen.py 保持一致
    对于基线模型，由于没有 aux 输出，正则化损失部分会被跳过
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        # 基线模型不会有 aux 输出，所以下面的正则化损失不会被应用
        # 但保留这部分代码以确保训练逻辑与 train_qwen.py 完全一致
        aux = getattr(outputs, "aux", None)
        if aux is None:
            return (loss, outputs) if return_outputs else loss

        # Optional layer filtering
        aux_layers = None
        if getattr(self.args, "aux_layers", None) is not None:
            try:
                aux_layers = set(
                    int(x.strip()) for x in self.args.aux_layers.split(",") if x.strip() != ""
                )
            except Exception:
                aux_layers = None

        # -------- L_orth: mean over layers of mean over tokens of |cos(a,r)| --------
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

        # -------- L_ctr: minimal stable in-batch contrastive on pooled a (placeholder) --------
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
                pooled.append(a.mean(dim=1))  # [B,T,D] -> [B,D]

            if len(pooled) > 0:
                z = torch.stack(pooled, dim=0).mean(dim=0)  # average over layers -> [B,D]
                z = torch.nn.functional.normalize(z, p=2, dim=-1)

                sim = torch.matmul(z, z.transpose(0, 1)) / tau  # [B,B]
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
trainer.train()
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
model.config.save_pretrained(training_args.output_dir)
