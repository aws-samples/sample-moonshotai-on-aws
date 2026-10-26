# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Indic Document Intelligence with Kimi K3 on Amazon Bedrock."""

from .config import Settings
from .images import load_image
from .pipeline import DocumentResult, IndicDocumentIntelligence

__all__ = [
    "DocumentResult",
    "IndicDocumentIntelligence",
    "Settings",
    "load_image",
]
