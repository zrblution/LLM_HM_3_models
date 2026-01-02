"""
inter_integration
-----------------
提供 Interaction Guidance Sampling (INTER) 的最小实现包装器与加载器，
以便在加载 Qwen2.5-VL 系列模型时快速集成 INTER 解码能力。

用法示例请见 `inter_integration/README.md`。
"""
from .loader import load_model_with_inter, InterModelWrapper


