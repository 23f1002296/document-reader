import json
import os
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


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/erp-extract")
async def erp_extract(req: ERPRequest):

    prompt = f"""
You are an expert information extraction system.

Extract information EXACTLY as written.

Rules:
- Copy vendor names exactly.
- Copy email addresses exactly, character for character.
- Copy SKU values exactly.
- Never correct spelling.
- Never normalize vendor names.
- Never infer missing characters.
- Return ONLY valid JSON.
- Follow the provided JSON schema exactly.

For identifiers (emails, SKUs, invoice numbers):

Copy them character-for-character from the source text.

Do NOT autocorrect.

Do NOT infer.

Do NOT rewrite.
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
                    "content": "You extract invoices into structured JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    except Exception:
        # Fallback for providers that don't support json_schema
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    data = json.loads(response.choices[0].message.content)

    # Return exactly the keys in the supplied schema
    properties = req.schema_.get("properties", {})

    result = {}

    for key in properties:
        result[key] = data.get(key, None)

    return result
