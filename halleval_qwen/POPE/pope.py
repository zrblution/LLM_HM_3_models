import json
import argparse
import os

def normalize_label(x):
    """Map various label strings to canonical 'yes' or 'no'."""
    if x is None:
        raise ValueError("label is None")
    s = str(x).strip().lower()
    if not s:
        raise ValueError("empty label")
    no_tokens = ["no", "不是", "不", "否", "no.", "n", "none", "没有", "false"]
    yes_tokens = ["yes", "是", "对", "正确", "y", "ya", "true", "有", "存在"]
    for t in no_tokens:
        if t in s:
            return "no"
    for t in yes_tokens:
        if t in s:
            return "yes"
    # fallback heuristics
    if s.startswith(("no", "not", "nah", "n")):
        return "no"
    if s.startswith(("yes", "y", "ya", "true")):
        return "yes"
    raise ValueError(f"无法识别 label: {x}")

def load_json(path):
    return json.load(open(path, encoding="utf-8"))

def evaluate(gt, pred):
    def build_map(items):
        m = {}
        for item in items:
            # Accept multiple possible id fields
            qid = item.get("question_id") or item.get("id") or item.get("questionId")
            if qid is None:
                raise KeyError(f"找不到问题 id 字段 in item: {item}")
            # Accept 'label' or 'labels' as the label field
            raw_lbl = item.get("label")
            if raw_lbl is None:
                raw_lbl = item.get("labels")
            if raw_lbl is None:
                # try common alternatives
                raw_lbl = item.get("gt_answer") or item.get("answer") or item.get("model_prediction")
            if raw_lbl is None:
                raise KeyError(f"找不到 label 字段 in item with id {qid}")
            m[qid] = normalize_label(raw_lbl)
        return m

    gt_map = build_map(gt)
    pred_map = build_map(pred)

    if set(gt_map.keys()) != set(pred_map.keys()):
        raise AssertionError(f"question id 不匹配: gt has {len(gt_map)} ids, pred has {len(pred_map)} ids")

    TP = FP = TN = FN = 0

    for qid in gt_map:
        g = gt_map[qid]
        p = pred_map[qid]

        if g == "yes" and p == "yes":
            TP += 1
        elif g == "no" and p == "yes":
            FP += 1
        elif g == "no" and p == "no":
            TN += 1
        elif g == "yes" and p == "no":
            FN += 1
        else:
            raise ValueError(f"非法 label: {g}, {p}")

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--output_json", required=True, help="Path to write the evaluation results as JSON")
    args = parser.parse_args()

    gt = load_json(args.gt_file)
    pred = load_json(args.pred_file)

    results = evaluate(gt, pred)

    # Print human-readable summary
    print("TP:", results["TP"])
    print("FP:", results["FP"])
    print("TN:", results["TN"])
    print("FN:", results["FN"])
    print()
    print("Accuracy : %.4f" % results["Accuracy"])
    print("Precision: %.4f" % results["Precision"])
    print("Recall   : %.4f" % results["Recall"])
    print("F1 Score : %.4f" % results["F1"])

    # Write full results to JSON file
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as outf:
        json.dump(results, outf, ensure_ascii=False, indent=2)
    print(f"Wrote evaluation results to {args.output_json}")
