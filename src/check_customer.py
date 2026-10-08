import json
from pathlib import Path

with open("data/gold.json", encoding="utf-8") as f:
    gold = {item["id"]: item for item in json.load(f)}

for doc_id in sorted(gold):
    pred_path = Path("data/predictions") / f"{doc_id}.json"
    with open(pred_path, encoding="utf-8") as f:
        pred = json.load(f)

    g = (gold[doc_id].get("customer_name") or "").strip()
    p = (pred.get("customer_name") or "").strip()

    if g.lower() != p.lower():
        print(f"\n=== {doc_id} ===")
        print(f"GOLD: {g}")
        print(f"PRED: {p}")