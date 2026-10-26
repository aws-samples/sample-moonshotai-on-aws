# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Kimi K3 client: settings, API key refresh and JSON replies."""

import json
import logging
import re
from typing import Optional

from .bedrock import ChatResult, chat_completion, get_bedrock_runtime_client, refresh_api_key
from .config import Settings
from .prompts import JSON_REPAIR_PROMPT

logger = logging.getLogger(__name__)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_response(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating markdown fences or stray prose."""
    cleaned = _CODE_FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(cleaned[start : end + 1])


def add_usage(total: dict, usage: dict) -> None:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + value


class KimiK3Client:
    def __init__(self, settings: Optional[Settings] = None, client=None):
        self.settings = settings or Settings.from_env()
        # Only refresh the API key on clients this class created itself.
        self._owns_client = client is None
        self.client = client or get_bedrock_runtime_client(self.settings.region)

    def chat(self, messages: list, system_prompt: str) -> ChatResult:
        if self._owns_client:
            refresh_api_key(self.client, self.settings.region)
        reply = chat_completion(
            self.client,
            model_id=self.settings.model_id,
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
        )
        if reply.finish_reason == "length":
            logger.warning("Reply hit max_tokens (%d) and is truncated; raise KIMI_MAX_TOKENS", self.settings.max_tokens)
        return reply

    def chat_json(self, messages: list, system_prompt: str, usage: dict) -> dict:
        """Send ``messages`` and parse the reply as JSON, asking the model to repair it once.

        ``messages`` is extended in place with the assistant reply (and any repair turn),
        and token usage is added to ``usage``.
        """
        reply = self.chat(messages, system_prompt)
        add_usage(usage, reply.usage)
        messages.append(reply.message)
        try:
            return parse_json_response(reply.text)
        except ValueError:
            # json.JSONDecodeError is a subclass of ValueError.
            logger.warning("Model reply was not valid JSON, asking it to repair once")
            messages.append({"role": "user", "content": JSON_REPAIR_PROMPT})
            reply = self.chat(messages, system_prompt)
            add_usage(usage, reply.usage)
            messages.append(reply.message)
            return parse_json_response(reply.text)
