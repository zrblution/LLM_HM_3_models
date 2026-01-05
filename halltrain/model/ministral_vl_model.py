"""
Ministral-3-8B-Instruct-2512 多模态模型适配
用于视觉证据记忆（VEM）训练，与 Qwen-VL 训练代码保持一致的逻辑

根据 HuggingFace 官方文档：
https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512

使用 Mistral3ForConditionalGeneration 加载模型
需要安装：
- transformers (from main branch for FP8 support)
- mistral-common >= 1.8.6
"""
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
import os

logger = logging.get_logger(__name__)


# ============== 动态导入 Mistral3 模型 ==============
def _get_mistral3_model_class():
    """
    动态获取 Mistral3ForConditionalGeneration 类
    兼容不同版本的 transformers
    """
    try:
        # 首先尝试官方推荐的导入方式
        from transformers import Mistral3ForConditionalGeneration
        return Mistral3ForConditionalGeneration, "Mistral3ForConditionalGeneration"
    except ImportError:
        pass
    
    try:
        # 尝试从 mistral 模块导入
        from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
        return Mistral3ForConditionalGeneration, "Mistral3ForConditionalGeneration"
    except ImportError:
        pass
    
    try:
        # 回退到 AutoModelForVision2Seq
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq, "AutoModelForVision2Seq"
    except ImportError:
        pass
    
    try:
        # 最后回退到 AutoModelForCausalLM
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM, "AutoModelForCausalLM"
    except ImportError:
        raise ImportError(
            "无法导入任何模型类。请确保安装了 transformers 库。\n"
            "推荐安装方式：pip install git+https://github.com/huggingface/transformers"
        )


# ============== VEM 核心模块（与 Qwen-VL 完全一致） ==============

class EvidenceRetriever(nn.Module):
    """Cross-attention retrieval from visual evidence memory."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_mem: torch.Tensor,
        v_mem_mask: Optional[torch.Tensor] = None,
    ):
        q = self.w_q(hidden_states)
        k = self.w_k(v_mem)
        v = self.w_v(v_mem)

        attn_logits = torch.matmul(q, k.transpose(-1, -2))

        if v_mem_mask is not None:
            mask = v_mem_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(
                mask == 0, torch.finfo(attn_logits.dtype).min
            )

        alpha = torch.softmax(attn_logits, dim=-1)
        e = torch.matmul(alpha, v)
        return e, alpha


class EvidenceAnalyzer(nn.Module):
    """Query-conditioned analysis: split candidate evidence into aligned (a) and residual (r)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, e: torch.Tensor, h: torch.Tensor):
        x = torch.cat([e, h], dim=-1)
        a = self.mlp(x)
        r = e - a
        return a, r


class EvidenceUtilization(nn.Module):
    """Estimate utilization strength u in (0,1) for each token."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, a: torch.Tensor, h: torch.Tensor):
        x = torch.cat([a, h], dim=-1)
        u = torch.sigmoid(self.mlp(x))
        return u


class EvidenceCorrector(nn.Module):
    """Residual decoding correction: h_tilde = h + u * W_c(a)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w_c = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h: torch.Tensor, a: torch.Tensor, u: torch.Tensor):
        return h + u * self.w_c(a)


# ============== 自定义输出类 ==============

class MinistralVLCausalLMOutputWithPast(CausalLMOutputWithPast):
    """扩展的输出类，包含 aux 和 rope_deltas"""
    def __init__(
        self,
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        aux=None,
        rope_deltas=None,
    ):
        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )
        self.aux = aux
        self.rope_deltas = rope_deltas


# ============== 自定义解码器层（添加 VEM 模块） ==============

class CustomMinistralDecoderLayer(nn.Module):
    """
    自定义的 Ministral 解码器层，添加视觉证据推理模块
    
    注意：original_layer 不注册为子模块，避免参数重复
    """
    def __init__(self, original_layer, hidden_size: int, layer_idx: int, config):
        super().__init__()
        # 使用 object.__setattr__ 避免将 original_layer 注册为子模块
        # 这样 original_layer 的参数不会被重复计算
        object.__setattr__(self, '_original_layer', original_layer)
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.config = config

        # 注意：enable_vision_gate 和 gate_layers 不再缓存为实例变量
        # 而是在 forward 中动态从 self.config 读取，以支持配置的后期更新
        # 保留这两个属性是为了兼容训练脚本中的同步逻辑
        self._enable_vision_gate = getattr(config, "enable_vision_gate", True)
        self._gate_layers = getattr(config, "gate_layers", None)

        # 视觉证据推理模块 (只有这些是新增的可训练参数)
        self.retriever = EvidenceRetriever(hidden_size)
        self.analyzer = EvidenceAnalyzer(hidden_size)
        self.util = EvidenceUtilization(hidden_size)
        self.corrector = EvidenceCorrector(hidden_size)
        self.concat_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
    
    @property
    def original_layer(self):
        """访问原始层（不作为子模块）"""
        return self._original_layer
    
    @property
    def enable_vision_gate(self):
        """动态获取 enable_vision_gate，优先从 config 读取"""
        return getattr(self.config, "enable_vision_gate", self._enable_vision_gate)
    
    @enable_vision_gate.setter
    def enable_vision_gate(self, value):
        """设置 enable_vision_gate"""
        self._enable_vision_gate = value
    
    @property
    def gate_layers(self):
        """动态获取 gate_layers，优先从 config 读取"""
        return getattr(self.config, "gate_layers", self._gate_layers)
    
    @gate_layers.setter
    def gate_layers(self, value):
        """设置 gate_layers"""
        self._gate_layers = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        v_mem: Optional[torch.Tensor] = None,
        v_mem_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        # 调用原始层的 forward
        layer_kwargs = {
            'hidden_states': hidden_states,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_value': past_key_value,
            'output_attentions': output_attentions,
            'use_cache': use_cache,
        }
        
        # 添加可选参数
        if cache_position is not None:
            layer_kwargs['cache_position'] = cache_position
        if position_embeddings is not None:
            layer_kwargs['position_embeddings'] = position_embeddings
            
        try:
            layer_outputs = self.original_layer(**layer_kwargs)
        except TypeError:
            # 如果某些参数不支持，尝试简化调用
            layer_outputs = self.original_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        # 获取 hidden_states
        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        # 视觉证据推理
        # 动态从 config 读取配置，支持配置的后期更新（如训练脚本中设置后同步）
        enable_vision_gate = getattr(self.config, "enable_vision_gate", self._enable_vision_gate)
        gate_layers = getattr(self.config, "gate_layers", self._gate_layers)
        
        aux = None
        if (
            enable_vision_gate
            and v_mem is not None
            and (gate_layers is None or self.layer_idx in gate_layers)
        ):
            inject_op = getattr(self.config, "inject_op", "ours").lower()
            use_u = getattr(self.config, "use_utilization", True)
            evidence_source = getattr(self.config, "evidence_source", "aligned").lower()

            e_t, alpha = self.retriever(hidden_states, v_mem, v_mem_mask)
            a_t, r_t = self.analyzer(e_t, hidden_states)

            src = e_t if evidence_source == "candidate" else a_t

            if use_u:
                u_t = self.util(a_t, hidden_states)
            else:
                u_t = None

            if inject_op == "ours":
                if u_t is None:
                    hidden_states = hidden_states + self.corrector.w_c(src)
                else:
                    hidden_states = self.corrector(hidden_states, src, u_t)
            elif inject_op == "add":
                delta = self.corrector.w_c(src)
                if u_t is None:
                    hidden_states = hidden_states + delta
                else:
                    hidden_states = hidden_states + u_t * delta
            elif inject_op == "concat":
                cat = torch.cat([hidden_states, src], dim=-1)
                delta = self.concat_proj(cat)
                if u_t is None:
                    hidden_states = hidden_states + delta
                else:
                    hidden_states = hidden_states + u_t * delta
            else:
                raise ValueError(f"Unknown inject_op: {inject_op}")

            aux = {
                "e": e_t,
                "a": a_t,
                "r": r_t,
                "u": u_t if u_t is not None else hidden_states.new_ones(hidden_states.size(0), hidden_states.size(1), 1),
                "alpha": alpha,
            }

        # 构造输出
        outputs = (hidden_states, aux)
        
        # 添加其他输出（attention weights, present_key_value 等）
        if isinstance(layer_outputs, tuple) and len(layer_outputs) > 1:
            outputs = outputs + layer_outputs[1:]

        return outputs


# ============== 主模型类 ==============

class Qwen2_5_CustomVLForConditionalGeneration(nn.Module):
    """
    Ministral-3-8B-Instruct-2512 多模态模型的自定义包装类，添加 VEM 训练模块
    
    根据 HuggingFace 官方文档使用 Mistral3ForConditionalGeneration 加载
    类名保持为 Qwen2_5_CustomVLForConditionalGeneration 以兼容现有训练代码
    """
    
    def __init__(self, config=None, model=None, hidden_size=None):
        super().__init__()
        self.config = config
        self._model = model
        self._hidden_size = hidden_size
        self.rope_deltas = None
        self.custom_layers = None
        self._model_class_name = None
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        加载 Ministral 多模态模型并添加 VEM 模块
        
        根据官方文档：
        - 使用 Mistral3ForConditionalGeneration 加载
        - 支持 FP8 和 BF16 两种模式
        """
        print(f"Loading Ministral-3-8B model from {pretrained_model_name_or_path}...")
        
        # 获取模型类
        ModelClass, class_name = _get_mistral3_model_class()
        print(f"Using model class: {class_name}")
        
        # 检查是否需要 BF16 转换（根据官方文档）
        # 如果用户没有指定 torch_dtype，默认使用 bfloat16
        if 'torch_dtype' not in kwargs:
            kwargs['torch_dtype'] = torch.bfloat16
            print("Using default dtype: bfloat16")
        
        # 从 kwargs 中移除 trust_remote_code 以避免重复传递
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('trust_remote_code', None)
        
        # 尝试加载模型
        try:
            # 首先尝试使用官方推荐的方式
            if class_name == "Mistral3ForConditionalGeneration":
                # 检查是否需要 FP8 -> BF16 转换
                try:
                    from transformers import FineGrainedFP8Config
                    # 如果模型是 FP8 格式，使用 dequantize 转换为 BF16
                    quantization_config = FineGrainedFP8Config(dequantize=True)
                    model = ModelClass.from_pretrained(
                        pretrained_model_name_or_path,
                        trust_remote_code=True,
                        quantization_config=quantization_config,
                        **kwargs_copy
                    )
                    print("Loaded with FP8 -> BF16 dequantization.")
                except (ImportError, Exception) as e:
                    print(f"FP8 config not available or failed: {e}")
                    # 直接加载
                    model = ModelClass.from_pretrained(
                        pretrained_model_name_or_path,
                        trust_remote_code=True,
                        **kwargs_copy
                    )
                    print(f"Loaded as {class_name}.")
            else:
                # 使用 Auto 类加载
                model = ModelClass.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    **kwargs_copy
                )
                print(f"Loaded as {class_name}.")
                
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
        
        # 获取 hidden_size（Mistral3 config nests text/vision configs）
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
                hidden_size = model.config.text_config.hidden_size
            elif hasattr(model.config, "vision_config") and hasattr(model.config.vision_config, "hidden_size"):
                hidden_size = model.config.vision_config.hidden_size
            elif hasattr(model.config, "d_model"):
                hidden_size = model.config.d_model
        if hidden_size is None:
            raise AttributeError("Config missing hidden size (expected hidden_size or text_config.hidden_size).")
        print(f"Model hidden_size: {hidden_size}")
        
        instance = cls(config=model.config, model=model, hidden_size=hidden_size)
        instance._model_class_name = class_name
        
        # 包装解码器层，添加 VEM 模块
        instance._wrap_decoder_layers()
        
        # 加载 VEM 模块的权重（如果存在）
        instance._load_custom_layers_weights(pretrained_model_name_or_path)
        
        # 打印 LOAD REPORT（类似 Qwen 模型的输出）
        instance._print_load_report(pretrained_model_name_or_path)
        
        return instance
    
    def _print_load_report(self, model_path):
        """打印 VEM 模块加载报告，类似 Qwen 模型的 LOAD REPORT"""
        if self.custom_layers is None:
            print(f"\nMinistralCustomVLForConditionalGeneration LOAD REPORT from: {model_path}")
            print("WARNING: No custom layers were created. VEM modules not added.")
            return
        
        num_layers = len(self.custom_layers)
        
        # 收集新增的 VEM 模块参数名
        vem_params = []
        for idx, layer in enumerate(self.custom_layers):
            for module_name in ['retriever', 'analyzer', 'util', 'corrector', 'concat_proj']:
                if hasattr(layer, module_name):
                    module = getattr(layer, module_name)
                    for name, _ in module.named_parameters():
                        param_name = f"custom_layers.{idx}.{module_name}.{name}"
                        vem_params.append(param_name)
        
        # 按模块类型分组
        param_groups = {}
        for param in vem_params:
            # 提取模块类型和参数名
            parts = param.split('.')
            if len(parts) >= 4:
                module_type = parts[2]  # retriever, analyzer, etc.
                param_type = '.'.join(parts[3:])  # w_q.weight, mlp.0.weight, etc.
                key = f"custom_layers.{{0...{num_layers-1}}}.{module_type}.{param_type}"
                param_groups[key] = "MISSING (newly initialized)"
        
        # 打印报告
        print(f"\nMinistralCustomVLForConditionalGeneration LOAD REPORT from: {model_path}")
        print("-" * 80)
        print(f"{'Key':<65} | {'Status':<12} |")
        print("-" * 80)
        
        for key, status in sorted(param_groups.items()):
            print(f"{key:<65} | {status:<12} |")
        
        print("-" * 80)
        print(f"\nNotes:")
        print(f"- MISSING (newly initialized): VEM modules were newly initialized for training.")
        print(f"- Total VEM parameters: {len(vem_params)} across {num_layers} layers.")
        print(f"- These parameters will be trained during fine-tuning.\n")
    
    def _load_custom_layers_weights(self, model_path):
        """从 checkpoint 加载 VEM 模块的权重"""
        if self.custom_layers is None:
            return
        
        import os
        from safetensors.torch import load_file
        
        # 查找 safetensors 文件
        safetensors_files = []
        if os.path.isdir(model_path):
            for f in os.listdir(model_path):
                if f.endswith('.safetensors'):
                    safetensors_files.append(os.path.join(model_path, f))
        
        if not safetensors_files:
            print(f"No safetensors files found in {model_path}, VEM modules will use random initialization.")
            return
        
        # 加载所有 safetensors 文件中的 custom_layers 权重
        loaded_count = 0
        for sf_path in safetensors_files:
            try:
                state_dict = load_file(sf_path)
                # 筛选出 custom_layers 相关的权重
                custom_weights = {k: v for k, v in state_dict.items() if k.startswith('custom_layers.')}
                
                if custom_weights:
                    # 加载权重到 custom_layers
                    missing, unexpected = self.custom_layers.load_state_dict(custom_weights, strict=False)
                    loaded_count += len(custom_weights) - len(missing)
                    
                    if loaded_count > 0:
                        print(f"✅ Loaded {loaded_count} VEM parameters from {os.path.basename(sf_path)}")
                        
            except Exception as e:
                print(f"Warning: Failed to load weights from {sf_path}: {e}")
        
        if loaded_count == 0:
            print("No VEM weights found in checkpoint, using random initialization.")
        else:
            # 确保数据类型一致
            model_dtype = next(self._model.parameters()).dtype
            for layer in self.custom_layers:
                for module_name in ['retriever', 'analyzer', 'util', 'corrector', 'concat_proj']:
                    if hasattr(layer, module_name):
                        module = getattr(layer, module_name)
                        module.to(dtype=model_dtype)
    
    def _wrap_decoder_layers(self):
        """包装解码器层，添加 VEM 模块"""
        # 查找解码器层 - Mistral3 的结构
        decoder_layers = None
        
        # Mistral3ForConditionalGeneration 的实际结构:
        # model.model.language_model.layers (26 layers of Ministral3DecoderLayer)
        
        # 路径1: model.model.language_model.layers (Mistral3ForConditionalGeneration)
        if hasattr(self._model, 'model') and hasattr(self._model.model, 'language_model'):
            lm = self._model.model.language_model
            if hasattr(lm, 'layers'):
                decoder_layers = lm.layers
                print(f"Found decoder layers at: model.model.language_model.layers (len={len(decoder_layers)})")
        
        # 路径2: model.language_model.layers (备选)
        if decoder_layers is None and hasattr(self._model, 'language_model'):
            lm = self._model.language_model
            if hasattr(lm, 'layers'):
                decoder_layers = lm.layers
                print(f"Found decoder layers at: language_model.layers (len={len(decoder_layers)})")
            elif hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                decoder_layers = lm.model.layers
                print(f"Found decoder layers at: language_model.model.layers (len={len(decoder_layers)})")
        
        # 路径3: model.model.layers (其他模型)
        if decoder_layers is None and hasattr(self._model, 'model') and hasattr(self._model.model, 'layers'):
            decoder_layers = self._model.model.layers
            print(f"Found decoder layers at: model.layers (len={len(decoder_layers)})")
        
        # 路径4: transformer.h (GPT-style)
        if decoder_layers is None and hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'h'):
            decoder_layers = self._model.transformer.h
            print(f"Found decoder layers at: transformer.h (len={len(decoder_layers)})")
        
        # 路径5: layers (直接)
        if decoder_layers is None and hasattr(self._model, 'layers'):
            decoder_layers = self._model.layers
            print(f"Found decoder layers at: layers (len={len(decoder_layers)})")
            
        if decoder_layers is None:
            logger.warning("Could not find decoder layers to wrap. VEM modules will not be added per-layer.")
            print(f"[DEBUG] _model type: {type(self._model).__name__}")
            self.custom_layers = None
            return
            
        # 创建自定义层
        self.custom_layers = nn.ModuleList([
            CustomMinistralDecoderLayer(
                original_layer=layer,
                hidden_size=self._hidden_size,
                layer_idx=idx,
                config=self.config
            )
            for idx, layer in enumerate(decoder_layers)
        ])
        
        print(f"Wrapped {len(self.custom_layers)} decoder layers with VEM modules.")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self._model, "gradient_checkpointing_enable"):
            return self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        if hasattr(self._model, "gradient_checkpointing"):
            self._model.gradient_checkpointing = True
        if hasattr(self._model, "config") and hasattr(self._model.config, "use_cache"):
            self._model.config.use_cache = False

    def gradient_checkpointing_disable(self):
        if hasattr(self._model, "gradient_checkpointing_disable"):
            return self._model.gradient_checkpointing_disable()
        if hasattr(self._model, "gradient_checkpointing"):
            self._model.gradient_checkpointing = False
    
    @property
    def model(self):
        """获取内部模型"""
        return self._model
    
    @property
    def visual(self):
        """获取视觉编码器（如果存在）"""
        # Mistral3 的视觉编码器位置
        if hasattr(self._model, 'vision_tower'):
            return self._model.vision_tower
        elif hasattr(self._model, 'vision_model'):
            return self._model.vision_model
        elif hasattr(self._model, 'visual'):
            return self._model.visual
        elif hasattr(self._model, 'vision_encoder'):
            return self._model.vision_encoder
        return None
    
    @property
    def lm_head(self):
        """获取语言模型头"""
        if hasattr(self._model, 'lm_head'):
            return self._model.lm_head
        elif hasattr(self._model, 'language_model') and hasattr(self._model.language_model, 'lm_head'):
            return self._model.language_model.lm_head
        return None
    
    def get_input_embeddings(self):
        return self._model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self._model.set_input_embeddings(value)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        前向传播，实现 VEM 训练逻辑
        """
        return_dict = return_dict if return_dict is not None else True
        
        # 如果没有自定义层，直接调用原始模型
        if self.custom_layers is None:
            return self._forward_without_vem(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        
        # 构建 VEM
        v_mem = None
        v_mem_mask = None
        
        # 调用原始模型获取 outputs（包括图像处理）
        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': True,  # 需要 hidden_states 来构建 VEM
            'return_dict': True,
        }
        
        if pixel_values is not None:
            model_kwargs['pixel_values'] = pixel_values
        
        # 添加 image_sizes（Ministral 需要）
        if 'image_sizes' in kwargs and kwargs['image_sizes'] is not None:
            model_kwargs['image_sizes'] = kwargs['image_sizes']
            
        if position_ids is not None:
            model_kwargs['position_ids'] = position_ids
            
        if past_key_values is not None:
            model_kwargs['past_key_values'] = past_key_values
            
        if inputs_embeds is not None:
            model_kwargs['inputs_embeds'] = inputs_embeds
            model_kwargs.pop('input_ids', None)
        
        try:
            outputs = self._model(**model_kwargs)
        except TypeError as e:
            # 简化调用
            logger.warning(f"Some parameters not supported, simplifying call: {e}")
            simplified_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'return_dict': True,
            }
            if pixel_values is not None:
                simplified_kwargs['pixel_values'] = pixel_values
            # 添加 image_sizes
            if 'image_sizes' in kwargs and kwargs['image_sizes'] is not None:
                simplified_kwargs['image_sizes'] = kwargs['image_sizes']
            outputs = self._model(**simplified_kwargs)
        
        # 从 hidden_states 构建 VEM
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # 使用第一层的 hidden_states 作为 VEM
            if pixel_values is not None and image_grid_thw is not None:
                try:
                    first_hidden = outputs.hidden_states[0]  # [B, T, D]
                    bsz = first_hidden.shape[0]
                    
                    # 使用 image_grid_thw 计算每个样本的视觉 token 数
                    thw = image_grid_thw.to(first_hidden.device)
                    n_tokens_per_sample = (thw[:, 0] * thw[:, 1] * thw[:, 2]).tolist()
                    
                    max_n = max(n_tokens_per_sample) if n_tokens_per_sample else 0
                    d = first_hidden.shape[-1]
                    
                    if max_n > 0:
                        v_mem = first_hidden.new_zeros((bsz, max_n, d))
                        v_mem_mask = first_hidden.new_zeros((bsz, max_n), dtype=torch.long)
                        
                        for i, n_b in enumerate(n_tokens_per_sample):
                            n_b = int(n_b)
                            if n_b > 0 and n_b <= first_hidden.shape[1]:
                                v_mem[i, :n_b, :] = first_hidden[i, :n_b, :]
                                v_mem_mask[i, :n_b] = 1
                except Exception as e:
                    logger.warning(f"Failed to build VEM from hidden_states: {e}")
                    v_mem = None
                    v_mem_mask = None
        
        # 收集 aux 信息
        all_aux = []
        
        # ===== 支持 first_layer_input 注入模式 (+M) =====
        inject_position = getattr(self.config, "inject_position", "per_layer").lower()
        if inject_position == "first_layer_input" and v_mem is not None and self.custom_layers is not None:
            if getattr(self.config, "enable_vision_gate", True):
                # 获取最后一层的 hidden_states 作为输入
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    # 使用第一层之前的 embeddings（即 hidden_states[0]）
                    inputs_embeds = outputs.hidden_states[0]
                    
                    # 使用第一个自定义层的 VEM 模块进行注入
                    layer0 = self.custom_layers[0]
                    inject_op = getattr(self.config, "inject_op", "add").lower()
                    use_u = getattr(self.config, "use_utilization", False)
                    evidence_source = getattr(self.config, "evidence_source", "candidate").lower()
                    
                    # 检索证据
                    e_t, alpha = layer0.retriever(inputs_embeds, v_mem, v_mem_mask)
                    a_t, r_t = layer0.analyzer(e_t, inputs_embeds)
                    
                    src = e_t if evidence_source == "candidate" else a_t
                    
                    if use_u:
                        u_t = layer0.util(a_t, inputs_embeds)
                    else:
                        u_t = None
                    
                    # 应用注入操作
                    if inject_op == "ours":
                        if u_t is None:
                            modified_embeds = inputs_embeds + layer0.corrector.w_c(src)
                        else:
                            modified_embeds = layer0.corrector(inputs_embeds, src, u_t)
                    elif inject_op == "add":
                        delta = layer0.corrector.w_c(src)
                        if u_t is None:
                            modified_embeds = inputs_embeds + delta
                        else:
                            modified_embeds = inputs_embeds + u_t * delta
                    elif inject_op == "concat":
                        cat = torch.cat([inputs_embeds, src], dim=-1)
                        delta = layer0.concat_proj(cat)
                        if u_t is None:
                            modified_embeds = inputs_embeds + delta
                        else:
                            modified_embeds = inputs_embeds + u_t * delta
                    else:
                        raise ValueError(f"Unknown inject_op: {inject_op}")
                    
                    # 记录 aux 信息
                    all_aux.append({
                        "layer_idx": 0,
                        "e": e_t,
                        "a": a_t,
                        "r": r_t,
                        "u": u_t if u_t is not None else inputs_embeds.new_ones(inputs_embeds.size(0), inputs_embeds.size(1), 1),
                        "alpha": alpha,
                    })
                    
                    # first_layer_input 模式下，禁用 per_layer 注入
                    v_mem = None
                    v_mem_mask = None
        # =====================================================
        
        # 如果有 VEM，通过自定义层处理（per_layer 模式）
        if v_mem is not None and self.custom_layers is not None:
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
                
                # 通过自定义层进行 VEM 注入
                for layer in self.custom_layers:
                    layer_outputs = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        v_mem=v_mem,
                        v_mem_mask=v_mem_mask,
                    )
                    hidden_states = layer_outputs[0]
                    if layer_outputs[1] is not None:
                        all_aux.append({"layer_idx": layer.layer_idx, **layer_outputs[1]})
        
        # 构造输出
        return MinistralVLCausalLMOutputWithPast(
            loss=outputs.loss if hasattr(outputs, 'loss') else None,
            logits=outputs.logits if hasattr(outputs, 'logits') else None,
            past_key_values=getattr(outputs, 'past_key_values', None),
            hidden_states=getattr(outputs, 'hidden_states', None),
            attentions=getattr(outputs, 'attentions', None),
            aux=all_aux if all_aux else None,
            rope_deltas=self.rope_deltas,
        )
    
    def _forward_without_vem(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """不使用 VEM 的前向传播"""
        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': True,
        }
        
        if pixel_values is not None:
            model_kwargs['pixel_values'] = pixel_values
        
        # 添加 image_sizes（Ministral 需要）
        if 'image_sizes' in kwargs and kwargs['image_sizes'] is not None:
            model_kwargs['image_sizes'] = kwargs['image_sizes']
        
        try:
            outputs = self._model(**model_kwargs)
        except TypeError as e:
            logger.warning(f"Some parameters not supported, simplifying call: {e}")
            simplified_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'return_dict': True,
            }
            if pixel_values is not None:
                simplified_kwargs['pixel_values'] = pixel_values
            # 添加 image_sizes
            if 'image_sizes' in kwargs and kwargs['image_sizes'] is not None:
                simplified_kwargs['image_sizes'] = kwargs['image_sizes']
            outputs = self._model(**simplified_kwargs)
        
        return MinistralVLCausalLMOutputWithPast(
            loss=outputs.loss if hasattr(outputs, 'loss') else None,
            logits=outputs.logits if hasattr(outputs, 'logits') else None,
            past_key_values=getattr(outputs, 'past_key_values', None),
            hidden_states=getattr(outputs, 'hidden_states', None),
            attentions=getattr(outputs, 'attentions', None),
            aux=None,
            rope_deltas=self.rope_deltas,
        )
    
    def generate(self, *args, **kwargs):
        """生成方法"""
        return self._model.generate(*args, **kwargs)
    
    def save_pretrained(self, save_directory, *args, **kwargs):
        """保存模型"""
        self._model.save_pretrained(save_directory, *args, **kwargs)
        
        # 保存自定义层
        if self.custom_layers is not None:
            custom_state = {
                f"custom_layers.{k}": v 
                for k, v in self.custom_layers.state_dict().items()
            }
            torch.save(custom_state, os.path.join(save_directory, "vem_modules.pt"))
    
    def state_dict(self, *args, **kwargs):
        """获取状态字典"""
        state = self._model.state_dict(*args, **kwargs)
        if self.custom_layers is not None:
            for k, v in self.custom_layers.state_dict().items():
                state[f"custom_layers.{k}"] = v
        return state
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """加载状态字典"""
        # 分离自定义层的状态
        custom_state = {}
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith("custom_layers."):
                custom_state[k.replace("custom_layers.", "")] = v
            else:
                model_state[k] = v
        
        self._model.load_state_dict(model_state, *args, **kwargs)
        if self.custom_layers is not None and custom_state:
            self.custom_layers.load_state_dict(custom_state, *args, **kwargs)
    
    def parameters(self, recurse=True):
        """获取参数 - 只返回唯一的参数，避免重复"""
        # 返回原始模型的参数
        for p in self._model.parameters(recurse=recurse):
            yield p
        # 返回 custom_layers 中 VEM 模块的参数（不包括 original_layer）
        if self.custom_layers is not None:
            for layer in self.custom_layers:
                # 只返回 VEM 模块的参数，不返回 original_layer 的参数
                for name, p in layer.named_parameters(recurse=False):
                    # 跳过 original_layer 相关的参数（虽然现在不应该有了）
                    if not name.startswith('_original_layer'):
                        yield p
                # 返回 VEM 子模块的参数
                for module_name in ['retriever', 'analyzer', 'util', 'corrector', 'concat_proj']:
                    if hasattr(layer, module_name):
                        module = getattr(layer, module_name)
                        for p in module.parameters(recurse=recurse):
                            yield p
    
    def named_parameters(self, prefix='', recurse=True):
        """获取命名参数 - 只返回唯一的参数，避免重复"""
        # 返回原始模型的参数
        for name, p in self._model.named_parameters(prefix=prefix, recurse=recurse):
            yield name, p
        # 返回 custom_layers 中 VEM 模块的参数
        if self.custom_layers is not None:
            for idx, layer in enumerate(self.custom_layers):
                layer_prefix = f'custom_layers.{idx}' if not prefix else f'{prefix}.custom_layers.{idx}'
                # 只返回 VEM 模块的参数
                for module_name in ['retriever', 'analyzer', 'util', 'corrector', 'concat_proj']:
                    if hasattr(layer, module_name):
                        module = getattr(layer, module_name)
                        module_prefix = f'{layer_prefix}.{module_name}'
                        for name, p in module.named_parameters(prefix=module_prefix, recurse=recurse):
                            yield name, p
    
    def train(self, mode=True):
        """设置训练模式"""
        self._model.train(mode)
        if self.custom_layers is not None:
            self.custom_layers.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        self._model.eval()
        if self.custom_layers is not None:
            self.custom_layers.eval()
        return self
    
    def to(self, *args, **kwargs):
        """移动到设备"""
        self._model.to(*args, **kwargs)
        if self.custom_layers is not None:
            self.custom_layers.to(*args, **kwargs)
        return self
    
    def cuda(self, device=None):
        """移动到 CUDA"""
        self._model.cuda(device)
        if self.custom_layers is not None:
            self.custom_layers.cuda(device)
        return self
    
    def cpu(self):
        """移动到 CPU"""
        self._model.cpu()
        if self.custom_layers is not None:
            self.custom_layers.cpu()
        return self


# 为了兼容性，也导出为 MinistralCustomVLForConditionalGeneration
MinistralCustomVLForConditionalGeneration = Qwen2_5_CustomVLForConditionalGeneration


__all__ = ["MinistralCustomVLForConditionalGeneration", "Qwen2_5_CustomVLForConditionalGeneration"]
