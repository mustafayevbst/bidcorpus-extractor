import json
import re
from pathlib import Path
from unicodedata import normalize as uni_normalize

from rapidfuzz import fuzz

GOLD_PATH = Path("data/gold.json")
PRED_DIR = Path("data/predictions")

SCALAR_FIELDS = ["customer_name", "tender_number", "category", "deadline"]
LIST_FIELDS = ["requirements", "participation_conditions"]

FUZZY_THRESHOLD = 0.85

def strip_accents(s: str) -> str :
    return uni_normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")

def normalize(s) -> str:
    if s is None:
        return ""

    s=str(s).lower().strip()
    s=strip_accents(s)
    s=re.sub(r"[^\w\s]", " ", s)
    s=re.sub(r"\s+", " ", s).strip()
    return s

def scalar_scores(gold_val, pred_val) -> tuple[int, int]:
    g=normalize(gold_val)
    p=normalize(pred_val)

    if not g and not p:
        return 1, 1
    if not g or not p:
        return 0, 0

    exact = 1 if g == p else 0

    if exact:
        return 1, 1

    ratio = fuzz.token_set_ratio(g, p) / 100.0
    fuzzy = 1 if ratio >= FUZZY_THRESHOLD else 0
    return exact, fuzzy

def greedy_match(g_list, p_list, threshold) -> int:
    pairs=[]
    for gi, g_item in enumerate(g_list):
        for pi, p_item in enumerate(p_list):
            score=fuzz.token_set_ratio(g_item, p_item) / 100.0
            if score >= threshold:
                pairs.append((score, gi, pi))

    pairs.sort(reverse=True)

    used_g, used_p = set(), set()
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)

    return len(used_g)

def list_prf(gold_list, pred_list, use_fuzzy: bool) -> tuple[float, float, float]:
    g = [normalize(x) for x in (gold_list or []) if x]
    p = [normalize(x) for x in (pred_list or []) if x]
    g = [x for x in g if x]
    p = [x for x in p if x]

    if not g and not p:
        return 1.0, 1.0, 1.0
    if not g or not p:
        return 0.0, 0.0, 0.0

    if use_fuzzy:
        tp=greedy_match(g, p, FUZZY_THRESHOLD)
    else:
        tp=len(set(g) & set(p))

    precision = tp/len(p)
    recall = tp/len(g)
    f1 = 2*precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def evaluate():
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold=json.load(f)

    rows=[]
    for item in gold:
        doc_id = item["id"]
        pred_path = PRED_DIR / f"{doc_id}.json"
        if not pred_path.exists():
            print(f"Нет предсказания для {doc_id}, пропуск")
            continue

        with open(pred_path, encoding="utf-8") as f:
            pred=json.load(f)

        row={"id": doc_id}

        for field in SCALAR_FIELDS:
            exact, fuzzy = scalar_scores(item.get(field), pred.get(field))
            row[f"{field}_exact"] = exact
            row[f"{field}_fuzzy"]=fuzzy

        for field in LIST_FIELDS:
            pe, re_, fe = list_prf(item.get(field, []), pred.get(field, []), use_fuzzy=False)
            pf, rf, ff = list_prf(item.get(field, []), pred.get(field, []), use_fuzzy=True)

            row[f"{field}_exact_p"] = pe
            row[f"{field}_exact_r"] = re_
            row[f"{field}_exact_f1"] = fe
            row[f"{field}_fuzzy_p"] = pf
            row[f"{field}_fuzzy_r"] = rf
            row[f"{field}_fuzzy_f1"] = ff

        rows.append(row)

    return rows

def print_report(rows):
    n = len(rows)
    if n == 0:
        print("Нет данных")
        return

    print(f"\nОценка на {n} документах, порог fuzzy = {FUZZY_THRESHOLD}\n")
    print("=" * 70)

    print("\nСкалярные поля (accuracy):\n")
    print(f"  {'field':<25} {'exact':>8} {'fuzzy':>8}")
    for field in SCALAR_FIELDS:
        e = sum(r[f"{field}_exact"] for r in rows) / n
        f = sum(r[f"{field}_fuzzy"] for r in rows) / n
        print(f"  {field:<25} {e:>8.2f} {f:>8.2f}")

    print("\nСписки (macro-усреднение):\n")
    print(f"  {'field':<25} {'P':>6} {'R':>6} {'F1':>6}   {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'':<25} {'exact':>18}   {'fuzzy':>18}")
    for field in LIST_FIELDS:
        pe = sum(r[f"{field}_exact_p"] for r in rows) / n
        re_ = sum(r[f"{field}_exact_r"] for r in rows) / n
        fe = sum(r[f"{field}_exact_f1"] for r in rows) / n
        pf = sum(r[f"{field}_fuzzy_p"] for r in rows) / n
        rf = sum(r[f"{field}_fuzzy_r"] for r in rows) / n
        ff = sum(r[f"{field}_fuzzy_f1"] for r in rows) / n
        print(f"  {field:<25} {pe:>6.2f} {re_:>6.2f} {fe:>6.2f}   {pf:>6.2f} {rf:>6.2f} {ff:>6.2f}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    rows=evaluate()
    print_report(rows)