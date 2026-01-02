"""
Minimal VCD loader and wrapper.

Usage:
    from vcd_integration import load_model_with_vcd
    wrapper = load_model_with_vcd("/path/to/Qwen2.5-VL-3B-Instruct", device="cuda")
    out_text = wrapper.generate_vcd(messages, image_pil, max_length=64, distortions=["blur","mask"])

Notes:
 - 这是一个 training-free、逐步（但非高效）实现：每一步会对原图与扰动图进行前向以获得 logits 并做简单对比。
 - 性能/并行化/数值稳定性需按实际环境进一步优化；目前实现优先可替换性与可理解性。
"""
from typing import List, Optional, Tuple
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import warnings

try:
    # 当仓库有 qwen_vl_utils 时优先使用其 process_vision_info
    from qwen_vl_utils import process_vision_info  # type: ignore
    _HAS_PROCESS_VISION_INFO = True
except Exception:
    _HAS_PROCESS_VISION_INFO = False


def _apply_distortion(image: Image.Image, kind: str = "blur") -> Image.Image:
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=3))
    if kind == "mask":
        img = image.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        # 随机遮挡一个矩形
        rw = int(w * 0.3)
        rh = int(h * 0.3)
        x0 = np.random.randint(0, w - rw + 1)
        y0 = np.random.randint(0, h - rh + 1)
        draw.rectangle([x0, y0, x0 + rw, y0 + rh], fill=(0, 0, 0))
        return img
    # fallback: 返回原图
    return image


class VCDModelWrapper(torch.nn.Module):
    """
    包装原始模型，提供一个简单的 generate_vcd 接口。
    说明：此 wrapper 不改变模型参数，仅在解码时运行额外的图像扰动前向并对 logits 做简单对比。
    """
    def __init__(self, model, tokenizer, processor, device: str = "cuda"):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

    @staticmethod
    def _prepare_inputs_for_messages(processor, tokenizer, messages: List[dict], image: Image.Image):
        """
        按照 Qwen 的流程构造输入：使用 processor.apply_chat_template + processor(...)
        返回 input_ids (list), pixel_values tensor, image_grid_thw tensor
        """
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 尝试兼容 qwen_vl_utils.process_vision_info 的返回（若存在）
        if _HAS_PROCESS_VISION_INFO:
            vision_inputs, _ = process_vision_info(messages)
            images = vision_inputs
        else:
            images = [image]

        inputs = processor(
            text=[text_input],
            images=images,
            return_tensors="pt",
            do_resize=True,
            padding=True,
        )

        # processor 返回 tensor，转到 device
        inputs = {k: v.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for k, v in inputs.items()}
        return inputs

    def generate_vcd(
        self,
        messages: List[dict],
        image: Image.Image,
        max_length: int = 64,
        distortions: Optional[List[str]] = None,
        alpha: float = 1.0,
        do_sample: bool = True,
        temperature: float = 1.0,
        noise_step: int = 500,
        plausibility_lambda: float = 0.1,
        stop_token_id: Optional[int] = None,
    ) -> str:
        """
        一个最小可运行的 VCD 解码器（贪心，每步对原图与每个扰动图分别前向并对比 logits）。

        参数说明：
        - messages: Qwen 样式的 messages 列表（同 train.py 使用方式）
        - image: PIL.Image
        - distortions: 例如 ["blur","mask"]。若为 None，则使用 ["blur"] 作为默认扰动。
        - alpha: 控制对比的强度，最终使用 logits_combined = logits_orig - alpha * logits_dist
        - stop_token_id: 若提供，遇到该 token 则停止
        """
        if distortions is None:
            distortions = ["blur"]

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # 准备原图 inputs
        orig_inputs = self._prepare_inputs_for_messages(self.processor, self.tokenizer, messages, image)

        # 准备扰动图对应的 pixel_values（只做一次预处理）
        distorted_inputs_list = []
        for d in distortions:
            img_d = _apply_distortion(image, d)
            inp_d = self._prepare_inputs_for_messages(self.processor, self.tokenizer, messages, img_d)
            distorted_inputs_list.append(inp_d)

        # 拿到初始 input_ids（batch 1）
        input_ids = orig_inputs["input_ids"].tolist()[0]

        # 把 pixel_values 和 image_grid_thw 做为常量（batch 1）
        orig_pixel_values = orig_inputs.get("pixel_values", None)
        orig_image_grid_thw = orig_inputs.get("image_grid_thw", None)
        distorted_pixel_values = [d.get("pixel_values", None) for d in distorted_inputs_list]
        distorted_image_grid_thw = [d.get("image_grid_thw", None) for d in distorted_inputs_list]

        # 确保所有 tensor 在同一 device
        if orig_pixel_values is not None:
            orig_pixel_values = orig_pixel_values.to(device)
        for i in range(len(distorted_pixel_values)):
            if distorted_pixel_values[i] is not None:
                distorted_pixel_values[i] = distorted_pixel_values[i].to(device)

        # 逐步贪心解码（注意：此方式效率低，但实现简单且可理解）
        generated = []
        eos_token_id = stop_token_id or getattr(self.tokenizer, "eos_token_id", None)

        for step in range(max_length):
            # 构造当前 input_ids 张量
            cur_ids = torch.tensor([input_ids + generated], dtype=torch.long, device=device)

            # 原图前向
            with torch.no_grad():
                out_orig = self.model(input_ids=cur_ids, pixel_values=orig_pixel_values, image_grid_thw=orig_image_grid_thw)
                logits_orig = out_orig.logits  # [1, seq_len, vocab]
            last_logits = logits_orig[0, -1, :].float()

            # 对每个扰动做前向并累加对比
            agg_dist_logits = torch.zeros_like(last_logits, device=device)
            for dpv, dig in zip(distorted_pixel_values, distorted_image_grid_thw):
                if dpv is None:
                    continue
                with torch.no_grad():
                    out_dist = self.model(input_ids=cur_ids, pixel_values=dpv, image_grid_thw=dig)
                    dist_logits = out_dist.logits[0, -1, :].float()
                agg_dist_logits += dist_logits
            if len(distorted_pixel_values) > 0:
                agg_dist_logits = agg_dist_logits / max(1, len(distorted_pixel_values))

            # 简单的 contrastive reweighting：orig - alpha * dist
            combined = last_logits - alpha * agg_dist_logits

            # 选择下一个 token：支持采样或贪心。temperature 与 do_sample 对应论文里常用的解码超参数。
            if do_sample:
                # 对数值做温度缩放后采样
                scaled = combined / (temperature if temperature > 0 else 1.0)
                probs = torch.softmax(scaled, dim=0)
                # 防止数值问题
                if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() == 0:
                    next_token = int(torch.argmax(combined).item())
                else:
                    next_token = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_token = int(torch.argmax(combined).item())
            generated.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

        # 解码生成 token（去掉 BOS/EOS 清理由用户自己决定）
        out_text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return out_text


def load_model_with_vcd(pretrained_model_name_or_path: str, tokenizer=None, processor=None, device: str = "cuda"):
    """
    便捷加载函数：加载 model/tokenizer/processor（若未提供），并返回一个 VCDModelWrapper。
    目标：可直接替换原先在 `train.py` 中的模型加载处，例如把
        model = Qwen2_5_CustomVLForConditionalGeneration.from_pretrained(...)
    替换为
        wrapper = load_model_with_vcd(...); model = wrapper.model
    或者直接在训练结束后使用 wrapper.generate_vcd(...) 进行 VCD 推理测试。
    """
    from transformers import AutoTokenizer, AutoProcessor
    from model.qwen_vl_model import Qwen2_5_CustomVLForConditionalGeneration

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, trust_remote_code=True)
    if processor is None:
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

    # 加载 base model（保持原先的 from_pretrained 行为）
    model = Qwen2_5_CustomVLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
    wrapper = VCDModelWrapper(model=model, tokenizer=tokenizer, processor=processor, device=device)
    return wrapper


