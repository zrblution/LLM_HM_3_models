import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import List

# 默认输入路径（可被命令行覆盖）
DEFAULT_INPUT = '/media/ubuntu/data/xican/coco_2014_data/coco_ground_truth_segmentation.json'
DEFAULT_OUTPUT_DIR = './output'

# 所有 COCO 类别（同 POPE）
# 不再预定义 all_categories；后续从输入数据统计所有出现的类别


def build_stats(lines: List[str]):
    """统计类别频率和共现矩阵（写入 ground_truth 和 co_occur 文件用）。"""
    freq = Counter()
    cooccur = defaultdict(Counter)
    for raw in lines:
        item = json.loads(raw)
        objs = set(item.get("objects", []))
        for a in objs:
            freq[a] += 1
        for a in objs:
            for b in objs:
                if a == b:
                    continue
                cooccur[a][b] += 1
    return freq, cooccur


def select_negatives_random(absent: List[str], k: int, rnd: random.Random):
    return rnd.sample(absent, min(k, len(absent)))


def select_negatives_popular(absent: List[str], k: int, freq: Counter):
    return sorted(absent, key=lambda c: (-freq.get(c, 0), c))[:k]


def select_negatives_adversarial(absent: List[str], present: List[str], k: int, cooccur: dict, freq: Counter, rnd: random.Random):
    # 按与 present 中所有对象的共现次数求和排序，优先选择分数高的 absent 类别
    scores = {}
    for a in absent:
        s = 0
        for p in present:
            s += cooccur.get(p, {}).get(a, 0)
        scores[a] = s
    sorted_by_score = sorted(absent, key=lambda c: (-scores.get(c, 0), -freq.get(c, 0), c))
    chosen = []
    for c in sorted_by_score:
        if len(chosen) >= k:
            break
        if scores.get(c, 0) > 0:
            chosen.append(c)
    # 回退到 popular / random 保证数量
    if len(chosen) < k:
        needed = k - len(chosen)
        pop_fill = [c for c in select_negatives_popular(absent, k, freq) if c not in chosen]
        chosen.extend(pop_fill[:needed])
    if len(chosen) < k:
        remaining = [c for c in absent if c not in chosen]
        if remaining:
            chosen.extend(select_negatives_random(remaining, k - len(chosen), rnd))
    return chosen[:k]


def write_json(path: str, items: List[dict]):
    """将 items 作为一个 JSON 列表写入文件（非 JSONL）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as outf:
        json.dump(items, outf, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to segmentation/annotation json-lines file")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write output POPE files")
    parser.add_argument("--dataset", default="coco", help="Dataset name used in output filenames")
    parser.add_argument("--template", default="Is there a {} in the image?", help="Question template")
    parser.add_argument("--sample_num", type=int, default=3, help="Number of negative objects to sample per image (POPE's sample_num)")
    parser.add_argument("--img_num", type=int, default=500, help="Number of images for building POPE (POPE's img_num)")
    parser.add_argument("--seg_num", type=int, default=1000, help="(ignored) kept for compatibility")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rnd = random.Random(args.seed)

    # 读取所有行
    with open(args.input, "r", encoding="utf-8") as f:
        all_lines = [l for l in f if l.strip()]
    total_lines = len(all_lines)
    if total_lines == 0:
        raise RuntimeError("输入文件为空或无有效行")

    # 统计频率与共现，并写 ground_truth 和 co_occur 文件
    freq, cooccur = build_stats(all_lines)

    os.makedirs(args.output_dir, exist_ok=True)
    gt_path = os.path.join(args.output_dir, f"{args.dataset}_ground_truth_objects.json")
    co_path = os.path.join(args.output_dir, f"{args.dataset}_co_occur.json")
    with open(gt_path, "w", encoding="utf-8") as fg:
        json.dump({k: int(v) for k, v in freq.items()}, fg, ensure_ascii=False, indent=2)
    # co_occur 保存为 dict of dict
    with open(co_path, "w", encoding="utf-8") as fc:
        json.dump({a: dict(b) for a, b in cooccur.items()}, fc, ensure_ascii=False, indent=2)

    # 随机抽取 img_num 张图片（或全部）
    img_num = min(args.img_num, total_lines)
    sampled_indices = rnd.sample(range(total_lines), img_num)

    # 为三种策略分别构建问题列表
    modes = ["random", "popular", "adversarial"]
    outputs = {m: [] for m in modes}
    qid = 1

    for idx in sampled_indices:
        item = json.loads(all_lines[idx])
        img_name = item.get("image")
        present = item.get("objects", [])
        # 正例
        for obj in present:
            for m in modes:
                outputs[m].append({
                    "id": qid,
                    "img": img_name,
                    "text": args.template.format(obj),
                    "labels": "yes"
                })
            qid += 1

        # 负例：用从数据统计得到的所有类别作为候选
        absent = [c for c in freq.keys() if c not in present]
        if not absent:
            continue
        neg_random = select_negatives_random(absent, args.sample_num, rnd)
        neg_popular = select_negatives_popular(absent, args.sample_num, freq)
        neg_adv = select_negatives_adversarial(absent, present, args.sample_num, cooccur, freq, rnd)

        for neg in neg_random:
            outputs["random"].append({
                "id": qid,
                "img": img_name,
                "text": args.template.format(neg),
                "labels": "no"
            })
            qid += 1
        for neg in neg_popular:
            outputs["popular"].append({
                "id": qid,
                "img": img_name,
                "text": args.template.format(neg),
                "labels": "no"
            })
            qid += 1
        for neg in neg_adv:
            outputs["adversarial"].append({
                "id": qid,
                "img": img_name,
                "text": args.template.format(neg),
                "labels": "no"
            })
            qid += 1

    # 写出三种 POPE 文件（JSON 列表，每项包含 id,img,text,labels）
    for m in modes:
        out_path = os.path.join(args.output_dir, f"{args.dataset}_pope_{m}.json")
        write_json(out_path, outputs[m])

    print(f"Wrote POPE files to {args.output_dir}: {', '.join([f'{args.dataset}_pope_{m}.json' for m in modes])}")


if __name__ == "__main__":
    main()
