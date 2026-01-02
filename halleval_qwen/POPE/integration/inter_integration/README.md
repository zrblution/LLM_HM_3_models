inter_integration
-----------------

这是一个最小化的 INTER（Interaction Guidance Sampling）集成示例包。

目的：
- 提供与 `train.py` 一致的模型加载流程（通过 `from_pretrained`），并返回一个带有 `generate_inter` 解码方法的 wrapper。
- 方便你在评测脚本中直接替换模型加载为：

```python
from inter_integration import load_model_with_inter
wrapper = load_model_with_inter("/path/to/Qwen2.5-VL-3B-Instruct", device="cuda")
model = wrapper.model  # 该 model 可用于训练（与原来接口一致）

# 推理示例
messages = [
    {"role":"system","content":"You are a helpful assistant."},
    {"role":"user","content":[{"type":"image","image":pil_image},{"type":"text","text":"Describe the image."}]}
]
out = wrapper.generate_inter(messages, pil_image, max_length=64, alpha=1.0)
print(out)
```

注意：
- 本实现为功能性最小版本，使用逐步贪心和两次前向（full / text-only）估计交互性；论文中有更细致的交互度量、与采样器的结合策略以及效率优化，可在此基础上扩展。
- 如果需要我可以把 `train.py` 中的模型加载替换示例加入到 `train.py`，或把 `--enable_inter` 参数加入训练脚本（默认为关闭）。


