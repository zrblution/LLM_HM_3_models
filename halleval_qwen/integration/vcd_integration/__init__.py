"""
vcd_integration
---------------
提供一个轻量级的加载器，用于在加载 Qwen2.5-VL 系列模型时，包装出带有 Visual Contrastive Decoding (VCD)
能力的模型对象。该包实现一个 minimal 的、training-free 的 VCD 解码器接口，供替换 `train.py` 中模型加载流程使用。

注意：本实现为最小可运行版本，侧重可用性与可替换性，性能和论文中的完全实现（各种扰动、并行化、效率优化）仍可后续增强。
"""
from .loader import load_model_with_vcd, VCDModelWrapper


