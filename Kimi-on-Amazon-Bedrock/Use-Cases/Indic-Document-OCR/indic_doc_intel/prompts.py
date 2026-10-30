# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Prompt templates."""

SYSTEM_PROMPT = """You are a document intelligence assistant for Indian documents.
You read scanned documents written in Hindi and other Indic languages, often mixed with English.
Rules:
- Only report what is visible in the document. Never invent or guess values.
- If a field is missing or illegible, return null for it and mention it in "uncertain_fields".
- Convert Devanagari digits to Western digits in structured fields.
- Return dates as YYYY-MM-DD and amounts as plain numbers without currency symbols or commas."""

EXTRACTION_PROMPT = """Read the attached document image.
Document type: {description}

Return a single JSON object, with no other text, using exactly these top-level keys:
- "document_type": short English label for the document
- "detected_languages": list of ISO 639-1 codes for languages present (e.g. ["hi", "en"])
- "transcription": full text of the document in its original script, preserving line breaks
- "fields": {fields_instruction}
- "english_summary": 2-4 sentence plain-English summary of what the document is and what action, if any, it asks for
- "uncertain_fields": list of field names that were unclear or missing"""

FIELDS_WITH_SCHEMA = "an object with exactly these keys:\n{field_list}"
FIELDS_WITHOUT_SCHEMA = (
    "an object of the 5-10 most important key-value pairs in the document, "
    "with snake_case English keys"
)
GENERIC_DESCRIPTION = (
    "Any Indic-language document. Choose the most useful key fields yourself."
)

JSON_REPAIR_PROMPT = (
    "Your previous reply was not valid JSON. Reply again with only the JSON object, "
    "no markdown fences and no commentary."
)

QUESTION_PROMPT = """Answer the user's question about the document above, using only what the document says.
Reply in the same language and the same script as the question, even if the document is in a different language:
- A question in Tamil script gets an answer in Tamil script, a question in Devanagari Hindi gets an answer in Devanagari Hindi, and so on.
- A question in Hinglish (Hindi written in Latin script, mixed with English) gets an answer in Hinglish.
- Translate terms from the document into the user's language instead of quoting them in the document's script.
Keep the answer short and include the exact values (amounts, dates, numbers) from the document.

Question: {question}"""


def build_extraction_prompt(schema: dict | None) -> str:
    """Build the extraction prompt. With no schema (or no fields), the model picks the fields."""
    if schema and schema.get("fields"):
        field_list = "\n".join(
            f'    "{name}": {instruction}'
            for name, instruction in schema["fields"].items()
        )
        fields_instruction = FIELDS_WITH_SCHEMA.format(field_list=field_list)
    else:
        fields_instruction = FIELDS_WITHOUT_SCHEMA
    description = (schema or {}).get("description") or GENERIC_DESCRIPTION
    return EXTRACTION_PROMPT.format(
        description=description, fields_instruction=fields_instruction
    )


def build_question_prompt(question: str) -> str:
    return QUESTION_PROMPT.format(question=question)
