# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Indic Document Intelligence with Kimi K3 on Amazon Bedrock."""

from .client import KimiK3Client
from .config import Settings
from .pipeline import DocumentResult, IndicDocumentIntelligence
from .schemas import DOCUMENT_SCHEMAS

__all__ = [
    "DOCUMENT_SCHEMAS",
    "DocumentResult",
    "IndicDocumentIntelligence",
    "KimiK3Client",
    "Settings",
]
