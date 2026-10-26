# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Kimi K3 client for the OpenAI-compatible Chat Completions API on the Amazon
Bedrock ``bedrock-runtime`` endpoint, which the Kimi K3 model card recommends
over the Converse API.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from aws_bedrock_token_generator import provide_token
from openai import OpenAI

from .config import Settings
from .prompts import JSON_REPAIR_PROMPT

logger = logging.getLogger(__name__)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass
class ChatResult:
    text: str
    message: dict
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""


def get_bedrock_runtime_client(region: str) -> OpenAI:
    """OpenAI SDK client pointed at bedrock-runtime, authenticated with a short-term
    Amazon Bedrock API key generated from your AWS credentials."""
    return OpenAI(
        base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
        api_key=provide_token(region=region),
        max_retries=5,
        timeout=300,
    )


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


def _usage_to_dict(usage) -> dict:
    if usage is None:
        return {}
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
        "cached_prompt_tokens": getattr(prompt_details, "cached_tokens", 0) or 0,
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", 0) or 0,
    }


class KimiK3Client:
    def __init__(self, settings: Optional[Settings] = None, client: Optional[OpenAI] = None):
        self.settings = settings or Settings.from_env()
        # Only refresh the API key on clients this class created itself.
        self._owns_client = client is None
        self.client = client or get_bedrock_runtime_client(self.settings.region)

    def chat(self, messages: list, system_prompt: str) -> ChatResult:
        if self._owns_client:
            # Short-term keys expire after at most 12 hours (or when your session does), so
            # regenerate before each request. Generation is local and makes no network call.
            self.client.api_key = provide_token(region=self.settings.region)

        request = {
            "model": self.settings.model_id,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            # Includes reasoning tokens, so leave generous headroom.
            "max_completion_tokens": self.settings.max_tokens,
        }
        if self.settings.temperature is not None:
            request["temperature"] = self.settings.temperature
        if self.settings.reasoning_effort:
            request["reasoning_effort"] = self.settings.reasoning_effort

        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("Reply hit max_tokens (%d) and is truncated; raise KIMI_MAX_TOKENS", self.settings.max_tokens)

        # Kimi K3 returns its reasoning in a separate reasoning_content field. Only the
        # final answer is kept in the conversation history.
        text = (choice.message.content or "").strip()
        return ChatResult(
            text=text,
            message={"role": "assistant", "content": text},
            usage=_usage_to_dict(response.usage),
            finish_reason=choice.finish_reason or "",
        )

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
