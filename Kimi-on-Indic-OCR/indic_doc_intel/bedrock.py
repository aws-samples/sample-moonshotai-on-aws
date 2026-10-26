# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Thin wrappers around the OpenAI-compatible Chat Completions API on the
Amazon Bedrock ``bedrock-runtime`` endpoint, which the Kimi K3 model card
recommends over the Converse API.
"""

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from aws_bedrock_token_generator import provide_token
from openai import OpenAI
from PIL import Image

# Downscale large scans and fall back to JPEG for very large images. This keeps
# latency and input tokens down without hurting legibility of typical A4 documents.
MAX_IMAGE_BYTES = 3_750_000
MAX_IMAGE_SIDE = 2048


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


def refresh_api_key(client: OpenAI, region: str) -> None:
    """Short-term keys expire after at most 12 hours (or when your session does), so
    regenerate before each request. Generation is local and makes no network call."""
    client.api_key = provide_token(region=region)


def load_image(image: Union[str, Path, bytes]) -> dict:
    """Return a Chat Completions image content part for a file path or raw bytes."""
    raw = Path(image).read_bytes() if isinstance(image, (str, Path)) else image

    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        image_format = "png"
        if buffer.tell() > MAX_IMAGE_BYTES:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            image_format = "jpeg"

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        # "detail" is honoured on Chat Completions; "high" keeps small Devanagari text legible.
        "image_url": {"url": f"data:image/{image_format};base64,{encoded}", "detail": "high"},
    }


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


def chat_completion(
    client: OpenAI,
    model_id: str,
    messages: list,
    system_prompt: str,
    max_tokens: int,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> ChatResult:
    request = {
        "model": model_id,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        # Includes reasoning tokens, so leave generous headroom.
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        request["temperature"] = temperature
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**request)
    choice = response.choices[0]

    # Kimi K3 returns its reasoning in a separate reasoning_content field. Only the
    # final answer is kept in the conversation history.
    text = (choice.message.content or "").strip()
    return ChatResult(
        text=text,
        message={"role": "assistant", "content": text},
        usage=_usage_to_dict(response.usage),
        finish_reason=choice.finish_reason or "",
    )
