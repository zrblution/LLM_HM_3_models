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
            # 可以先简单检查一下 f 的结构
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
            # 如果这里报错，就打印出出问题的 text_features
            print("\n[ERROR] tokenizer.pad(...) failed. Below is the text_features content:\n")
            for i, tf in enumerate(text_features):
                print(f"  === Sample {i} ===")
                print("  input_ids:", tf["input_ids"])
                print("  attention_mask:", tf["attention_mask"])
                print("  labels:", tf["labels"])
                print("  ----------------")
            raise e  # 再把错误抛出

        # 3. 实时处理图像：从图片路径加载并处理图像
        pixel_values_list = []
        image_grid_thw_list = []
        
        for f in features:
            # 从保存的图片路径加载图像
            img_path = os.path.join(self.data_args.training_image_dir, f["img"])
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            image_pil = Image.open(img_path).convert("RGB")
            fixed_size = (224, 224)
            image_pil = image_pil.resize(fixed_size, Image.BICUBIC)
            
            # 使用保存的文本信息构造 messages
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
            
            # 处理图像
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
            
            # 提取 pixel_values 和 image_grid_thw
            # processor 返回的 pixel_values 形状可能是 [1, C, H, W] 或 [C, H, W]
            pv = inputs["pixel_values"]
            if pv.dim() == 4 and pv.shape[0] == 1:
                # 如果有 batch 维度且为 1，移除它
                pv = pv.squeeze(0)
            pixel_values_list.append(pv)
            
            # 处理 image_grid_thw
            gthw = inputs["image_grid_thw"]
            if gthw.dim() == 2 and gthw.shape[0] == 1:
                # 如果是 [1, 3]，squeeze 成 [3]
                gthw = gthw.squeeze(0)
            elif gthw.dim() == 1:
                # 如果已经是 [3]，保持不变
                pass
            else:
                # 其他情况，尝试 squeeze
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
    图像处理将在训练时的 DataCollator 中实时进行，避免预保存大量 pixel_values。
    
    example: {
      "id": 42953,
      "img": "img/42953.png",
      "label": "not-hateful",
      "text": "its their character not their color that matters",
    }
    """

    # ---------- 1. 准备构造输入文本 & 输出文本 ----------
    #    这里示例做法：把用户的文本 + 图片视作输入，label 视作模型需要生成的输出。
    #    也可以根据自己需要改成其它prompt风格。
    user_text = example["text"] + '\n'
    label_text = example["labels"]  # hmc是分类标签，如 "not-hateful"
    
    # ---------- 2. 使用真实图像计算 input_ids（确保 token 数量准确） ----------
    #    注意：我们处理图像只是为了计算正确的 token 数量，但不保存 pixel_values
    #    这样可以确保预处理和 collator 中的 token 数量一致
    
    # 读取图像（用于计算 token，但不保存处理结果）
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
    
    # 获取文本模板（包含图像 token）
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 使用 processor 处理，获取 input_ids（与原始代码保持一致）
    # 注意：我们只使用 input_ids，不保存 pixel_values
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        do_resize=True,
        padding=True,
    )
    # 提取 input_ids（与原始代码逻辑一致）
    text_input_ids = inputs["input_ids"][0].tolist()
    
    # assistant 的文字（即 label_text）也要拼进最终 tokens
    response = tokenizer([label_text], add_special_tokens=False)

    # ---------- 3. 拼接 input_ids、attention_mask、labels ----------
    input_ids = text_input_ids + response["input_ids"][0] + [tokenizer.pad_token_id]
    attention_mask = [1] * len(input_ids)
    labels = (
            [-100] * len(text_input_ids)
            + response["input_ids"][0]
            + [tokenizer.pad_token_id]
    )

    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    # ---------- 4. 只返回轻量级数据：文本 tokens 和图片路径 ----------
    #    不保存 pixel_values 和 image_grid_thw，这些将在 collator 中实时处理
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "img": example["img"],  # 保存图片路径
        "user_text": user_text,  # 保存用户文本，用于在 collator 中重新构造 messages
    }

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Attach evidence-regularization hyperparameters to training_args so the custom Trainer can access them
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
# 注意：如果之前有旧版本的缓存（包含 pixel_values），建议删除后重新预处理
# 新版本只保存轻量级数据（图片路径和文本 tokens），大幅减少存储空间
cache_dir = os.path.join(training_args.output_dir, "preprocessed_cache")
os.makedirs(cache_dir, exist_ok=True)
done_file = os.path.join(cache_dir, ".preprocessing_done")

# 检查缓存是否存在
cache_exists = os.path.exists(done_file) and os.path.exists(os.path.join(cache_dir, "dataset_info.json"))

# 只在主进程（rank 0）进行预处理，其他进程等待并加载缓存
if is_distributed:
    if rank == 0:
        if cache_exists:
            # 缓存已存在，直接加载
            print(f"[Rank {rank}] Preprocessed cache found, loading from {cache_dir}...")
            train_dataset = Dataset.load_from_disk(cache_dir)
            print(f"[Rank {rank}] Dataset loaded from cache.")
            # 通知其他进程
            dist.barrier()
        else:
            # 主进程：预处理数据
            print(f"[Rank {rank}] No cache found, starting data preprocessing...")
            with open(data_args.training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            train_ds = Dataset.from_list(training_data)
            
            # 创建一个包装函数，定期发送心跳保持连接活跃
            import threading
            import time
            import torch
            heartbeat_interval = 30  # 每30秒发送一次心跳
            heartbeat_stop = threading.Event()
            
            def heartbeat_worker():
                """后台线程定期发送心跳，保持分布式连接活跃"""
                while not heartbeat_stop.is_set():
                    try:
                        # 使用 all_reduce 操作保持连接活跃（使用公开 API）
                        # 创建一个小的 tensor 进行 all_reduce，保持连接活跃
                        dummy_tensor = torch.tensor([1.0], device='cpu')
                        dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM, async_op=False)
                        time.sleep(heartbeat_interval)
                    except Exception as e:
                        # 如果发送心跳失败，继续尝试
                        time.sleep(heartbeat_interval)
            
            # 启动心跳线程
            heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
            heartbeat_thread.start()
            
            try:
                train_dataset = train_ds.map(
                    lambda ex: process_func(ex, data_args, tokenizer, processor),
                    num_proc=1,  # 单进程预处理，减少内存占用
                    desc="Preprocessing dataset"
                )
            finally:
                # 停止心跳线程
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=5)
            
            # 保存预处理后的数据
            print(f"[Rank {rank}] Saving preprocessed dataset to {cache_dir}...")
            train_dataset.save_to_disk(cache_dir)
            del train_ds  # 释放原始数据内存
            import gc
            gc.collect()
            
            # 创建完成标志文件，通知其他进程
            with open(done_file, 'w') as f:
                f.write(str(time.time()))
            print(f"[Rank {rank}] Preprocessing completed and saved.")
            
            # 通知其他进程
            dist.barrier()
    else:
        # 其他进程：等待主进程完成
        if cache_exists:
            # 缓存已存在，直接加载
            print(f"[Rank {rank}] Preprocessed cache found, loading from {cache_dir}...")
            # 等待主进程同步
            dist.barrier()
            train_dataset = Dataset.load_from_disk(cache_dir)
            print(f"[Rank {rank}] Dataset loaded from cache.")
        else:
            # 等待主进程完成预处理
            print(f"[Rank {rank}] Waiting for rank 0 to finish preprocessing...")
            # 在等待期间定期检查心跳和文件，避免超时
            import time
            max_wait_time = 3600 * 2  # 最多等待2小时
            start_time = time.time()
            check_interval = 30  # 每30秒检查一次
            
            while True:
                try:
                    # 检查完成标志文件（主要同步机制）
                    if os.path.exists(done_file):
                        # rank0 已完成，可以执行 barrier
                        break
                    
                    # 检查是否超时
                    if time.time() - start_time > max_wait_time:
                        raise RuntimeError(f"Timeout waiting for rank 0 to finish preprocessing (waited {max_wait_time}s)")
                    
                    # 在等待期间，定期执行轻量级的分布式操作来保持连接活跃
                    # 这样可以避免在等待期间连接超时
                    try:
                        dummy_tensor = torch.tensor([1.0], device='cpu')
                        dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM, async_op=False)
                    except:
                        # 如果 all_reduce 失败（可能 rank0 还没准备好），继续等待
                        pass
                    
                    # 等待一段时间后再次检查
                    time.sleep(check_interval)
                except Exception as e:
                    if "barrier" in str(e).lower() or "timeout" in str(e).lower():
                        # 如果是 barrier 相关的错误，继续等待
                        if time.time() - start_time > max_wait_time:
                            raise
                        time.sleep(check_interval)
                    else:
                        raise
            
            # 执行 barrier 同步
            dist.barrier()
            
            # 加载预处理后的数据
            print(f"[Rank {rank}] Loading preprocessed dataset from {cache_dir}...")
            train_dataset = Dataset.load_from_disk(cache_dir)
            print(f"[Rank {rank}] Dataset loaded.")
else:
    # 非分布式环境：检查缓存或直接预处理
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
        
        # 保存预处理后的数据
        print(f"Saving preprocessed dataset to {cache_dir}...")
        train_dataset.save_to_disk(cache_dir)
        import time
        with open(done_file, 'w') as f:
            f.write(str(time.time()))
        print("Preprocessing completed and saved.")

model = Qwen2_5_CustomVLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

def _parse_gate_layers(spec: str):
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return None
    if spec == "none":
        return []
    # comma-separated ints
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(int(x))
    return out

# ---- Write CLI controls into model.config (single source of truth) ----
model.config.enable_vision_gate = bool(getattr(model_args, "enable_evidence", True))
model.config.gate_layers = _parse_gate_layers(getattr(model_args, "gate_layers", "all"))

model.config.inject_position = getattr(model_args, "inject_position", "per_layer").strip().lower()
model.config.inject_op = getattr(model_args, "inject_op", "ours").strip().lower()
model.config.use_utilization = bool(getattr(model_args, "use_utilization", True))
model.config.evidence_source = getattr(model_args, "evidence_source", "aligned").strip().lower()

model.config.export_u_stats = bool(getattr(model_args, "export_u_stats", False))
model.config.export_u_stats_path = getattr(model_args, "export_u_stats_path", None)

# Basic validation
valid_pos = {"none", "first_layer_input", "per_layer"}
valid_op = {"ours", "add", "concat"}
valid_src = {"candidate", "aligned"}

if model.config.inject_position not in valid_pos:
    raise ValueError(f"inject_position must be one of {valid_pos}, got {model.config.inject_position}")
if model.config.inject_op not in valid_op:
    raise ValueError(f"inject_op must be one of {valid_op}, got {model.config.inject_op}")
if model.config.evidence_source not in valid_src:
    raise ValueError(f"evidence_source must be one of {valid_src}, got {model.config.evidence_source}")

# 创建 collator，传入 processor 和 data_args 以便实时处理图像
data_collator = MultiModalCollator(tokenizer=tokenizer, processor=processor, data_args=data_args)

# 确保DataLoader不使用额外的worker进程，减少内存占用
# 注意：由于需要在 collator 中实时处理图像，建议使用单进程加载数据
if not hasattr(training_args, 'dataloader_num_workers') or training_args.dataloader_num_workers is None:
    training_args.dataloader_num_workers = 0  # 设置为0，避免额外的worker进程占用内存


class EvidenceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

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
    data_collator=data_collator,  # custom collator
)

# 开启模型训练
trainer.train()
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
model.config.save_pretrained(training_args.output_dir)


