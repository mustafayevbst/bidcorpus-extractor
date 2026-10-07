import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"

RAW_DIR = Path("data/raw")
RU_DIR = Path("data/raw_ru")
RU_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """Ты переводчик тендерных документов с португальского на русский.
Переводи ТОЧНО, сохраняя номера пунктов, статьи законов, названия организаций и юридические термины.
Названия организаций и имена собственные оставляй как есть, но в скобках давай перевод, если он очевиден.
Отвечай ТОЛЬКО переводом, без пояснений и комментариев."""

def translate_text(text:str, client:Groq) -> str:
    if not text or not text.strip():
        return ""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def translate_file(path: Path, client:Groq) ->dict:
    with open(path, encoding="utf-8") as f:
        doc=json.load(f)

    sections = ["OBJETO", "JULGAMENTO", "CONDICAO_PARTICIPACAO", "HABILITACAO", "CREDENCIAMENTO"]

    result = {
        "id": doc["id"],
        "ID_LICITACAO":doc.get("ID_LICITACAO"),
        "ID_ARQUIVO": doc.get("ID_ARQUIVO"),
    }

    for section in sections:
        original = doc.get(section, "") or ""
        if original.strip():
            result[section] = translate_text(original, client)
        else:
            result[section] = ""
    return result

if __name__=="__main__":
    client= Groq(api_key=os.getenv("GROQ_API_KEY"))

    files = sorted(RAW_DIR.glob("*.json"))
    print(f"Найдено {len(files)} файлов для перевода")

    for i, path in enumerate(files, 1):
        out_path = RU_DIR / path.name

        if out_path.exists():
            print(f"[{i}/{len(files)}] {path.name} - уже переведен, пропуск")
            continue

        print(f"[{i}/{len(files)}] {path.name} - перевожу")
        try:
            result= translate_file(path, client)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error: {e}")
    print("Готово")