"""
Minimal INTER loader and wrapper.

说明：
- 该实现为 training-free 的解码改进（与论文一致），提供一个简单的 `generate_inter` 接口。
- 实现为逐步贪心解码，并使用“全模态 vs 文本仅”两次前向的 logits 差分作为交互性估计，
  然后按比例把该交互性作用到最终 logits 上引导采样。
- 这是一个最小可运行版本，便于和 `train.py` 的加载流程替换和兼容。
"""
from typing import List, Optional
import torch
from PIL import Image
import numpy as np

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
    _HAS_PROCESS_VISION_INFO = True
except Exception:
    _HAS_PROCESS_VISION_INFO = False


class InterModelWrapper(torch.nn.Module):
    """
    包装原始模型，提供 generate_inter 接口。
    """
    def __init__(self, model, tokenizer, processor, device: str = "cuda"):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

    def _prepare_inputs(self, messages: List[dict], image: Optional[Image.Image], include_image: bool = True):
        """
        构造 processor inputs。若 include_image=False 则不传图像（或传空白）。
        返回 processor(...) 的 outputs（tensor），均置于 wrapper device。
        """
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [image] if (include_image and image is not None and not _HAS_PROCESS_VISION_INFO) else []

        # 若仓库中存在 process_vision_info，则使用它来构造 vision inputs，以保证兼容性
        if _HAS_PROCESS_VISION_INFO and include_image and image is not None:
            vision_inputs, _ = process_vision_info(messages)
            inputs = self.processor(text=[text_input], images=vision_inputs, return_tensors="pt", do_resize=True, padding=True)
        else:
            inputs = self.processor(text=[text_input], images=images, return_tensors="pt", do_resize=True, padding=True)

        # 转到 device
        inputs = {k: v.to(torch.device(self.device if torch.cuda.is_available() else "cpu")) for k, v in inputs.items()}
        return inputs

    def generate_inter(
        self,
        messages: List[dict],
        image: Image.Image,
        max_length: int = 64,
        alpha: float = 1.0,
        beam_size: int = 5,
        use_beam: bool = True,
        sample: bool = True,
        stop_token_id: Optional[int] = None,
    ) -> str:
        """
        Minimal INTER decoding:
        - 在每步，计算 full 模态 logits 与 text-only logits（不提供图像）；
        - 以 interaction = logits_full - logits_text_only 作为交互性估计；
        - 用 combined = logits_full + alpha * interaction 引导下一 token 的选择。

        说明：该实现为贪心选择（argmax），仅供快速集成与功能验证，非论文中全部细节或高效实现。
        """
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # 准备原始与 text-only 的 inputs（只需处理一次 image preprocess）
        full_inputs = self._prepare_inputs(messages, image, include_image=True)
        text_only_inputs = self._prepare_inputs(messages, image, include_image=False)

        # 初始 input_ids（list）
        input_ids = full_inputs["input_ids"].tolist()[0]

        # 固定 pixel_values
        full_pixel_values = full_inputs.get("pixel_values", None)
        full_image_grid_thw = full_inputs.get("image_grid_thw", None)
        if full_pixel_values is not None:
            full_pixel_values = full_pixel_values.to(device)
        if full_image_grid_thw is not None:
            full_image_grid_thw = full_image_grid_thw.to(device)

        # 文本仅不含 pixel_values，一般 processor 不会返回 pixel_values，当为空时前向时传 None
        generated = []
        eos_token_id = stop_token_id or getattr(self.tokenizer, "eos_token_id", None)

        # 使用 beam-search（论文中对 INTER 的评测使用了基于 beam-search 的设置），
        # 这里实现一个简单且可控的 beam-search。若不使用 beam，则退回到逐步 greedy/sample。
        if use_beam and beam_size > 1:
            beams = [([], 0.0)]  # list of (tokens_list, logprob_sum)
            finished = []

            for step in range(max_length):
                candidates = []
                for seq, score in beams:
                    cur_ids = torch.tensor([input_ids + seq], dtype=torch.long, device=device)
                    with torch.no_grad():
                        out_full = self.model(input_ids=cur_ids, pixel_values=full_pixel_values, image_grid_thw=full_image_grid_thw)
                        logits_full = out_full.logits[0, -1, :].float()

                        out_text_only = self.model(input_ids=cur_ids, pixel_values=None, image_grid_thw=None)
                        logits_text_only = out_text_only.logits[0, -1, :].float()

                    interaction = logits_full - logits_text_only
                    combined = logits_full + alpha * interaction

                    log_probs = torch.log_softmax(combined, dim=0)
                    topk = torch.topk(log_probs, k=min(beam_size, log_probs.size(0)))
                    topk_vals = topk.values
                    topk_idx = topk.indices

                    for v, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
                        new_seq = seq + [int(idx)]
                        new_score = score + float(v)
                        if eos_token_id is not None and int(idx) == eos_token_id:
                            finished.append((new_seq, new_score))
                        else:
                            candidates.append((new_seq, new_score))

                if len(candidates) == 0:
                    break
                # 选取 top beam_size 作为下一轮 beams
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
                beams = candidates

                # 如果 finished 数量达到 beam_size，可以停止
                if len(finished) >= beam_size:
                    break

            final_candidates = finished + beams
            if len(final_candidates) == 0:
                generated = []
            else:
                generated = max(final_candidates, key=lambda x: x[1])[0]
        else:
            # 逐步 greedy 或采样
            generated = []
            for step in range(max_length):
                cur_ids = torch.tensor([input_ids + generated], dtype=torch.long, device=device)
                with torch.no_grad():
                    out_full = self.model(input_ids=cur_ids, pixel_values=full_pixel_values, image_grid_thw=full_image_grid_thw)
                    logits_full = out_full.logits[0, -1, :].float()

                    out_text_only = self.model(input_ids=cur_ids, pixel_values=None, image_grid_thw=None)
                    logits_text_only = out_text_only.logits[0, -1, :].float()

                interaction = logits_full - logits_text_only
                combined = logits_full + alpha * interaction

                if sample:
                    probs = torch.softmax(combined, dim=0)
                    if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() == 0:
                        next_token = int(torch.argmax(combined).item())
                    else:
                        next_token = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_token = int(torch.argmax(combined).item())

                generated.append(next_token)
                if eos_token_id is not None and next_token == eos_token_id:
                    break

        out_text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return out_text


def load_model_with_inter(pretrained_model_name_or_path: str, tokenizer=None, processor=None, device: str = "cuda"):
    """
    加载模型并返回 InterModelWrapper，加载流程兼容 train.py 的加载方式。
    """
    from transformers import AutoTokenizer, AutoProcessor
    from model.qwen_vl_model import Qwen2_5_CustomVLForConditionalGeneration

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, trust_remote_code=True)
    if processor is None:
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

    model = Qwen2_5_CustomVLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
    wrapper = InterModelWrapper(model=model, tokenizer=tokenizer, processor=processor, device=device)
    return wrapper


