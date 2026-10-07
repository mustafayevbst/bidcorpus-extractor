import json
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_bidcorpus(n_docs: int = 50):
    print("Загружаю bidCorpus_raw с Hugging Face...")
    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/tcepi/bidCorpus/bidCorpus_raw/train-00000-of-00009.parquet",
        split="train",
        streaming=True,
    )
 
    saved=0
    for i, row in enumerate(ds):
        if saved >= n_docs:
            break

        doc ={
            "id": f"doc_{i:04d}",
            "ID_LICITACAO": row.get("ID-LICITACAO"),
            "ID_ARQUIVO": row.get("ID-ARQUIVO"),
            "OBJETO": row.get("OBJETO", "") or "",
            "JULGAMENTO": row.get("JULGAMENTO", "") or "",
            "CONDICAO_PARTICIPACAO": row.get("CONDICAO_PARTICIPACAO", "") or "",
            "HABILITACAO": row.get("HABILITACAO", "") or "",
            "CREDENCIAMENTO": row.get("CREDENCIAMENTO", "") or "",
        }

        if not any([doc["OBJETO"], doc["HABILITACAO"], doc["JULGAMENTO"]]):
            continue

        path = DATA_DIR / f"{doc['id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        saved += 1


    print(f"Сохранено {saved} документов в {DATA_DIR}")
    return saved

if __name__=="__main__":
    download_bidcorpus(n_docs=50)
    