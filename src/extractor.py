import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.schemas import Tender

load_dotenv()

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """
Ты извлекаешь структурированные данные из тендерных документов на португальском языке.
Отвечай ТОЛЬКО валидным JSON. Никаких пояснений, markdown-оберток и текста вокруг.
Если поле не найдено в тексте, верни null. Если список пустой, верни [].

Схема ответа:
{
    "customer_name": string | null,
    "tender_number": string | null,
    "category": string | null,
    "deadline": string | null,
    "requirements": [string],
    "participation_conditions": [string]
}"""

def extract (text: str, client:Groq) ->dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)

def extract_file(path: Path, client: Groq) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    text = "\n\n".join(
        [
            f"OBJETO: {doc.get('OBJETO', '')}",
            f"HABILITACAO: {doc.get('HABILITACAO', '')}",
            f"JULGAMENTO: {doc.get('JULGAMENTO', '')}",
            f"CONDICAO_PARTICIPACAO: {doc.get('CONDICAO_PARTICIPACAO', '')}",
            f"CREDENCIAMENTO: {doc.get('CREDENCIAMENTO', '')}",
        ]
    )

    result = extract(text, client)
    result["id"] = doc["id"]
    return result


if __name__=="__main__":
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    raw_dir = Path("data/raw")
    files = sorted(raw_dir.glob("*.json"))[:3]

    for path in files:
        print(f"--- {path.name} ---")
        try:
            result=extract_file(path, client)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")
        print()