import json
import os
import time
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
}

Пояснения к полям:

"customer_name" — название организации-заказчика (Prefeitura, Tribunal, Batalhão, Ministério и т.п.).
Указывай полное каноническое название, включая штат/город, если он упомянут в тексте.
Заказчик может упоминаться неявно: в адресе для направления документов, в названии связанного
органа, в реквизитах контракта. Восстанавливай название из контекста, не возвращай null,
если заказчика можно определить.
Примеры:
  "Prefeitura Municipal de Curralinhos-PI"
  "Tribunal Regional do Trabalho da 18ª Região (TRT-18)"
  "31º Batalhão de Infantaria Motorizado"
  "Agência Espacial Brasileira (AEB)"
"tender_number" — номер процедуры, обычно в формате NNN/YYYY или "Pregão Nº NNN/YYYY".
Если в тексте нет номера, верни null.

"category" — это ЧТО ЗАКУПАЮТ, предмет закупки, а не тип процедуры и не критерий оценки.
ХОРОШО: "Combustíveis e lubrificantes", "Material de construção", "Serviços de manutenção predial",
"Equipamentos de informática", "Gêneros alimentícios", "Serviço de perícia médica".
ПЛОХО (это НЕ категория): "Registro de Preços", "Pregão Eletrônico", "MENOR PREÇO",
"Tomada de Preços", "Convite", "Maior Desconto".
Если предмет закупки описан в OBJETO как "aquisição de X" или "contratação de serviço de X",
то категория — это X в краткой форме.

"deadline" — дата и время вскрытия конвертов или окончания приёма заявок,
в формате DD/MM/YYYY HH:MM. Если указана только дата, верни DD/MM/YYYY.

"requirements" — список документов и требований к участникам для допуска к тендеру
(из секции HABILITACAO). Сохраняй формулировки из текста, но можно сокращать длинные
юридические обороты до сути. Не выдумывай требования, которых нет в тексте.

"participation_conditions" — список условий участия в тендере
(из секции CONDICAO_PARTICIPACAO): кто может участвовать, кто не может,
какие ограничения. Сохраняй суть, не перефразируй сильно.

Пример правильного ответа:
{
  "customer_name": "Prefeitura Municipal de Curralinhos-PI",
  "tender_number": "002/2013",
  "category": "Combustíveis e lubrificantes",
  "deadline": "13/05/2013 11:30",
  "requirements": [
    "Prova de situação regular perante o FGTS",
    "Certidão Negativa de Débito (CND) do INSS"
  ],
  "participation_conditions": [
    "Empresas cadastradas previamente com documentação válida na data da abertura",
    "Documentos para habilitação e proposta em envelopes distintos"
  ]
}
"""

def extract (text: str, client:Groq, max_retries: int=5) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=8192,
            )
            raw = response.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            if "429" in str(e):
                wait = 30 * (attempt+1)
                print(f"    429, жду {wait} сек (попытка {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Не удалось после {max_retries}попыток")

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

    with open("data/gold.json", encoding="utf-8") as f:
        gold=json.load(f)

    doc_ids= [item["id"] for item in gold]
    print(f"Документ в gold: {len(doc_ids)}")

    PRED_DIR = Path("data/predictions")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for i, doc_id in enumerate(doc_ids, 1):
        src=Path("data/raw") / f"{doc_id}.json"
        out= PRED_DIR / f"{doc_id}.json"

        if out.exists():
            print(f"[{i}/{len(doc_ids)}] {doc_id} - пропуск")
            continue

        print(f"[{i}/{len(doc_ids)}] {doc_id} - обработка")

        try:
            result=extract_file(src, client)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"ERROR: {e}")
    print("Готово")