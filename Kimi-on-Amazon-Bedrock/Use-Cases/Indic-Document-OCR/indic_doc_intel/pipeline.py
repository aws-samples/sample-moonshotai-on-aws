# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Document extraction and follow-up Q&A using Kimi K3 on Amazon Bedrock.

A schema is a plain dict that you define for your own document type::

    {
        "description": "What kind of document this is.",
        "fields": {"field_name": "Instruction passed to the model verbatim.", ...},
    }

Pass ``schema=None`` to let the model choose the key fields itself.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from openai import OpenAI

from .config import Settings
from .images import load_image
from .kimi import KimiK3Client, add_usage
from .prompts import SYSTEM_PROMPT, build_extraction_prompt, build_question_prompt

REQUIRED_KEYS = ("document_type", "detected_languages", "transcription", "fields", "english_summary", "uncertain_fields")


@dataclass
class DocumentResult:
    """Structured extraction plus the conversation needed for follow-up questions."""

    data: dict
    conversation: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    @property
    def fields(self) -> dict:
        return self.data.get("fields", {})

    @property
    def english_summary(self) -> str:
        return self.data.get("english_summary", "")

    def to_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False, indent=2)


def normalise_result(data: dict, schema: Optional[dict]) -> dict:
    """Make sure every top-level key and every schema field is present."""
    for key in REQUIRED_KEYS:
        data.setdefault(key, [] if key in ("detected_languages", "uncertain_fields") else None)
    if not isinstance(data["fields"], dict):
        data["fields"] = {}
    for name in (schema or {}).get("fields", {}):
        data["fields"].setdefault(name, None)
    return data


class IndicDocumentIntelligence:
    def __init__(self, settings: Optional[Settings] = None, client: Optional[OpenAI] = None):
        self.kimi = KimiK3Client(settings, client)
        self.settings = self.kimi.settings

    def extract(self, image: Union[str, Path, bytes], schema: Optional[dict] = None) -> DocumentResult:
        """OCR the document image and return structured fields plus an English summary."""
        messages = [
            {
                "role": "user",
                # The model card recommends placing images before text.
                "content": [load_image(image), {"type": "text", "text": build_extraction_prompt(schema)}],
            }
        ]
        usage: dict = {}
        data = self.kimi.chat_json(messages, SYSTEM_PROMPT, usage)
        return DocumentResult(data=normalise_result(data, schema), conversation=messages, usage=usage)

    def ask(self, result: DocumentResult, question: str) -> str:
        """Ask a follow-up question (e.g. in Hinglish) about an already extracted document.

        The question is appended to the extraction conversation, so the model still sees
        the original image. The answer is added to ``result.conversation`` so further
        questions keep the full context.
        """
        user_message = {"role": "user", "content": build_question_prompt(question)}
        reply = self.kimi.chat(result.conversation + [user_message], SYSTEM_PROMPT)
        add_usage(result.usage, reply.usage)
        result.conversation.extend([user_message, reply.message])
        return reply.text
