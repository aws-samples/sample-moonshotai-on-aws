# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Runtime settings, overridable through environment variables."""

import os
from dataclasses import dataclass
from typing import Optional

# India cross-Region inference profile for Kimi K3. Routes requests between
# Asia Pacific (Mumbai) ap-south-1 and Asia Pacific (Hyderabad) ap-south-2,
# keeping inference inside India.
DEFAULT_MODEL_ID = "in.moonshotai.kimi-k3"
DEFAULT_REGION = "ap-south-1"


@dataclass(frozen=True)
class Settings:
    model_id: str = DEFAULT_MODEL_ID
    region: str = DEFAULT_REGION
    # Counts reasoning tokens as well as the answer.
    max_tokens: int = 16384
    # One of none, low, medium, high, xhigh, max. Unset uses the model default (max).
    reasoning_effort: Optional[str] = None
    # Only sent when set explicitly; Kimi K3 is evaluated at its default sampling settings.
    temperature: Optional[float] = None

    @classmethod
    def from_env(cls) -> "Settings":
        temperature = os.environ.get("KIMI_TEMPERATURE")
        return cls(
            model_id=os.environ.get("KIMI_MODEL_ID", DEFAULT_MODEL_ID),
            # Deliberately not AWS_REGION: the "in." profile only resolves from an
            # India Region, and many shells already export AWS_REGION=us-east-1.
            region=os.environ.get("BEDROCK_REGION", DEFAULT_REGION),
            max_tokens=int(os.environ.get("KIMI_MAX_TOKENS", "16384")),
            reasoning_effort=os.environ.get("KIMI_REASONING_EFFORT") or None,
            temperature=float(temperature) if temperature else None,
        )
