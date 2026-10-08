"""CLI для запуска пайплайна.

Команды:
    python -m src.cli extract         # извлечь поля из всех документов в gold
    python -m src.cli evaluate        # посчитать метрики
    python -m src.cli check FIELD     # показать расхождения по полю
    python -m src.cli check-metrics   # проверить метрики против порогов
"""

import argparse
import json
import os
import sys

from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.evaluator import evaluate, print_report
from src.extractor import extract_file

load_dotenv()

GOLD_PATH = Path("data/gold.json")
RAW_DIR = Path("data/raw")
PRED_DIR = Path("data/predictions")


def cmd_extract(args):
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    with open(GOLD_PATH, encoding="utf-8") as f:
        gold = json.load(f)

    doc_ids = [item["id"] for item in gold]
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    for i, doc_id in enumerate(doc_ids, 1):
        src = RAW_DIR / f"{doc_id}.json"
        out = PRED_DIR / f"{doc_id}.json"

        if out.exists() and not args.force:
            print(f"[{i}/{len(doc_ids)}] {doc_id} - пропуск")
            continue

        print(f"[{i}/{len(doc_ids)}] {doc_id} - обработка")
        try:
            result = extract_file(src, client)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  Error: {e}")

    print("Готово")


def cmd_evaluate(args):
    rows = evaluate()
    print_report(rows)


def cmd_check(args):
    
    field = args.field

    valid_fields = [
        "customer_name", "tender_number", "category", "deadline",
        "requirements", "participation_conditions",
    ]
    if field not in valid_fields:
        print(f"Неизвестное поле: {field}")
        print(f"Доступные: {', '.join(valid_fields)}")
        return

    with open(GOLD_PATH, encoding="utf-8") as f:
        gold = {item["id"]: item for item in json.load(f)}

    print(f"\n=== Расхождения по полю: {field} ===\n")
    found = 0

    for doc_id in sorted(gold):
        pred_path = PRED_DIR / f"{doc_id}.json"
        if not pred_path.exists():
            continue

        with open(pred_path, encoding="utf-8") as f:
            pred = json.load(f)

        g = gold[doc_id].get(field)
        p = pred.get(field)

        if (g or "") != (p or ""):
            found += 1
            print(f"--- {doc_id} ---")
            print(f"  GOLD: {g!r}")
            print(f"  PRED: {p!r}")
            print()

    print(f"Всего расхождений: {found}")

MIN_METRICS = {
    "customer_name_fuzzy": 0.75,
    "tender_number_exact": 0.90,
    "category_fuzzy": 0.80,
    "deadline_exact": 0.90,
    "requirements_fuzzy_f1": 0.60,
    "participation_conditions_fuzzy_f1": 0.50,
}

def cmd_check_metrics(args):
    rows = evaluate()
    n=len(rows)
    if n == 0:
        print("Нет данных")
        sys.exit(1)

    failed = []
    for field, threshold in MIN_METRICS.items():
        if field.endswith("_fuzzy_f1"):
            base = field.replace("_fuzzy_f1", "")
            value = sum(r[f"{base}_fuzzy_f1"] for r in rows) / n
        elif field.endswith("_fuzzy"):
            value=sum(r[field] for r in rows) / n
        elif field.endswith("_exact"):
            value=sum(r[field] for r in rows) / n
        else:
            continue

        status="OK" if value >= threshold else "FAIL"
        print(f"{field:40s} {value:.2f} (min {threshold:.2f}) {status}")
        if value < threshold:
            failed.append(field)

    if failed:
        print(f"\nПровалено метрик: {len(failed)}")
        sys.exit(1)

    print("\nВсе метрики выше порога")

def main():
    parser = argparse.ArgumentParser(prog="src.cli", description="BidCorpus Tender Extractor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="извлечь поля из документов в gold")
    p_extract.add_argument("--force", action="store_true", help="перегенерировать даже существующие")
    p_extract.set_defaults(func=cmd_extract)

    p_eval = sub.add_parser("evaluate", help="посчитать метрики")
    p_eval.set_defaults(func=cmd_evaluate)

    p_check = sub.add_parser("check", help="показать расхождения по полю")
    p_check.add_argument("field", help="customer_name / tender_number / category / deadline / requirements / participation_conditions")
    p_check.set_defaults(func=cmd_check)

    p_metrics = sub.add_parser("check-metrics", help="проверить метрики против порогов")
    p_metrics.set_defaults(func=cmd_check_metrics)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()