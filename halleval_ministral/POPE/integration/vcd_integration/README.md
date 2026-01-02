vcd_integration
---------------

这是一个小型辅助包，用于将 Visual Contrastive Decoding（VCD）能力包装到现有 Qwen2.5-VL 模型上。

快速开始：

- 把目录放在工作区根目录（已放置于 /Users/xicanyang/Desktop/whole/vcd_integration）
- 在训练脚本中替换模型加载：

```python
from vcd_integration import load_model_with_vcd
wrapper = load_model_with_vcd("/path/to/Qwen2.5-VL-3B-Instruct", device="cuda")
model = wrapper.model  # 与原始模型兼容，可用于训练

# 推理时使用 VCD（示例）
messages = [
    {"role":"system","content":"You are a helpful assistant."},
    {"role":"user","content":[{"type":"image","image":pil_image},{"type":"text","text":"What is shown in this image?"}]}
]
out = wrapper.generate_vcd(messages, pil_image, max_length=64, distortions=["blur","mask"], alpha=1.0)
print(out)
```

注意事项：
- 本实现为最小可运行版本，逐步贪心解码，效率较低；仅作为可替换示例与起点。生产使用应按论文设计对解码器进行更高效实现（例如自定义 LogitsProcessor / 并行推理 / 缓存复用等）。
- 若仓库中存在 `qwen_vl_utils.process_vision_info`，本包会优先使用其预处理。若不存在，会直接把 PIL image 直接传给 `processor`。


