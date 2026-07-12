from typing import Any
from pydantic import BaseModel

class ERPRequest(BaseModel):
    document_id: str
    text: str
    schema: dict[str, Any]


@app.post("/erp-extract")
async def erp_extract(req: ERPRequest):

    prompt = f"""
Extract the invoice into JSON.

Rules:
- Return ONLY valid JSON.
- Follow the supplied JSON Schema EXACTLY.
- Do not add extra fields.
- Do not omit required fields.
- vendor: preserve exactly as written.
- currency: ISO4217 code.
- total_amount: integer.
- invoice_date: YYYY-MM-DD.
- due_in_days: integer.
- is_paid: boolean.
- priority: one of low, normal, high, urgent.
- contact_email: lowercase.
- line_items: preserve order.
- item_count = number of line_items.

Document:

{req.text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "erp_invoice",
                "schema": req.schema,
                "strict": True
            }
        },
        messages=[
            {
                "role": "system",
                "content": "You are an expert invoice extraction system."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)
