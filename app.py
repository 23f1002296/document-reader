import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIPIPE_API_KEY"),
    base_url="https://aipipe.org/openai/v1",
)

app = FastAPI(title="ERP Invoice Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ERPRequest(BaseModel):
    document_id: str
    text: str
    schema_: dict[str, Any] = Field(alias="schema")

    model_config = {
        "populate_by_name": True
    }


EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/erp-extract")
async def erp_extract(req: ERPRequest):

    prompt = f"""
You are an expert invoice extraction system.

Extract structured information from the invoice text.

IMPORTANT:
- Return ONLY valid JSON.
- Follow the supplied JSON Schema EXACTLY.
- Do NOT add extra fields.
- Do NOT omit fields.
- If a value cannot be determined, return null.

Extraction Rules:

- vendor: copy exactly as written.
- currency: return ISO 4217 code (USD, EUR, GBP, INR, JPY).
- total_amount: integer in the main currency unit.
- invoice_date: YYYY-MM-DD.
- due_in_days: integer.
- is_paid: boolean.
- priority: one of low, normal, high, urgent.
- contact_email: copy EXACTLY from the invoice text and lowercase it.
- line_items: preserve their order.
- item_count: number of line_items.

JSON Schema:

{json.dumps(req.schema_, indent=2)}

Invoice Text:

{req.text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "erp_invoice",
                    "schema": req.schema_,
                    "strict": True,
                },
            },
            messages=[
                {
                    "role": "system",
                    "content": "You extract invoices into structured JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    except Exception:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON that exactly matches the supplied schema."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    data = json.loads(response.choices[0].message.content)

    # Override email using regex from original text
    match = EMAIL_RE.search(req.text)
    if match:
        data["contact_email"] = match.group(0).lower()

    # Return exactly the schema keys
    result = {}
    properties = req.schema_.get("properties", {})

    for key in properties:
        result[key] = data.get(key, None)

    return result
