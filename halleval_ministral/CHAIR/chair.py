import os
import json
import argparse
import re


# Avoid automatic download of NLTK 'punkt' tokenizer which can block or require network.
# We'll use a lightweight regex-based tokenizer instead of `nltk.word_tokenize`.

# ======================
# 工具函数
# ======================
def singularize(w):
    # Lightweight heuristic singularization to avoid external dependencies.
    # Handles common English plurals: 'ies'->'y', trailing 's' -> remove (except short words).
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if w.endswith('s') and len(w) > 3 and not w.endswith('ss'):
        return w[:-1]
    return w

def load_captions(cap_file):
    """
    cap_file: list of {image_id, caption}
    """
    data = json.load(open(cap_file))
    assert isinstance(data, list)
    return data, set(d['image_id'] for d in data)


def load_coco_instances(annotation_path):
    """
    Load MSCOCO instance annotations
    """
    inst_file = os.path.join(annotation_path, 'instances_val2014.json')
    if not os.path.exists(inst_file):
        raise RuntimeError("Missing instances_val2014.json")

    coco = json.load(open(inst_file))

    id2name = {cat['id']: cat['name'] for cat in coco['categories']}

    imid_to_objects = {}
    for ann in coco['annotations']:
        imid = ann['image_id']
        name = id2name[ann['category_id']]
        imid_to_objects.setdefault(imid, set()).add(name)

    return imid_to_objects


# ======================
# CHAIR evaluator with synonyms
# ======================

class SimpleCHAIR:

    def __init__(self, imid_to_objects, synonyms_file=None):
        self.imid_to_objects = imid_to_objects

        # 加载 synonyms.txt
        self.mscoco_objects = set()
        self.inverse_synonym_dict = {}
        if synonyms_file and os.path.exists(synonyms_file):
            with open(synonyms_file) as f:
                synonyms = [line.strip().split(', ') for line in f if line.strip()]
            for syn_group in synonyms:
                self.mscoco_objects.update(syn_group)
                for s in syn_group:
                    self.inverse_synonym_dict[s] = syn_group[0]  # canonical name
        else:
            # 如果没有 synonyms 文件，直接把 GT 对象作为 canonical
            all_objs = set()
            for objs in imid_to_objects.values():
                all_objs.update(objs)
            self.mscoco_objects = all_objs
            self.inverse_synonym_dict = {o: o for o in all_objs}

        # 将 GT 对象映射为 canonical
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(
                self.inverse_synonym_dict.get(obj, obj)
                for obj in self.imid_to_objects[imid]
            )

    def caption_to_objects(self, caption):
        # Simple regex tokenizer: extract alphabetic words and common contractions.
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", caption.lower())
        words = [singularize(w) for w in words if w.isalpha()]

        # 只保留在 mscoco_objects 里的词
        filtered_words = [w for w in words if w in self.mscoco_objects]
        # 转换为 canonical
        node_words = [self.inverse_synonym_dict[w] for w in filtered_words]

        return node_words

    def compute(self, captions):
        num_caps = 0
        num_hallucinated_caps = 0
        hallucinated_words = 0
        total_words = 0
        # 额外统计用于平均与 recall
        per_image_chairi_sum = 0.0
        per_image_recall_sum = 0.0
        total_true_positives = 0
        total_ground_truth_objects = 0

        results = []

        for item in captions:
            imid = item['image_id']
            cap = item['caption']

            gt_objs = self.imid_to_objects.get(imid, set())
            node_words = self.caption_to_objects(cap)

            total_words += len(node_words)
            hallucinated = False
            bad_words = []

            for w in node_words:
                if w not in gt_objs:
                    hallucinated_words += 1
                    bad_words.append(w)
                    hallucinated = True

            num_caps += 1
            if hallucinated:
                num_hallucinated_caps += 1
            # 为便于审查，记录模型生成的物品列表（去重，保留出现顺序）和 GT 列表
            generated_objects = list(dict.fromkeys(node_words))
            ground_truth_objects = sorted(list(gt_objs))

            # 计算 per-image 指标：CHAIRi（已用于 overall CHAIRi），以及 per-image recall
            per_image_chairi = len(bad_words) / max(1, len(node_words))
            per_image_chairi_sum += per_image_chairi

            true_positives = len(set(generated_objects) & set(gt_objs))
            total_true_positives += true_positives
            total_ground_truth_objects += len(gt_objs)
            per_image_recall = true_positives / max(1, len(gt_objs))
            per_image_recall_sum += per_image_recall

            results.append({
                "image_id": imid,
                "caption": cap,
                "generated_objects": generated_objects,
                "ground_truth_objects": ground_truth_objects,
                "hallucinated_words": bad_words,
                "CHAIRs": int(hallucinated),
                "CHAIRi": per_image_chairi,
                "recall": per_image_recall
            })

        return {
            # 原始定义：CHAIRs = fraction of captions that contain >=1 幻觉物品
            "CHAIRs": num_hallucinated_caps / num_caps,
            # 原始定义：CHAIRi = hallucinated words / total predicted object words
            "CHAIRi": hallucinated_words / max(1, total_words),
            # recall：micro (总体 TP / 总 GT) 与 macro (per-image 平均)
            "recall_micro": total_true_positives / max(1, total_ground_truth_objects),
            "recall_macro": per_image_recall_sum / max(1, num_caps),
            "details": results
        }


# ======================
# main
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", required=True)
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--synonyms_file", default="data/synonyms.txt")
    args = parser.parse_args()

    captions, imids = load_captions(args.cap_file)
    imid_to_objects = load_coco_instances(args.annotation_path)

    evaluator = SimpleCHAIR(imid_to_objects, synonyms_file=args.synonyms_file)
    output = evaluator.compute(captions)

    print("CHAIRs: %.3f" % output["CHAIRs"])
    print("CHAIRi: %.3f" % output["CHAIRi"])

    with open("chair_results.json", "w") as f:
        json.dump(output, f, indent=2)

