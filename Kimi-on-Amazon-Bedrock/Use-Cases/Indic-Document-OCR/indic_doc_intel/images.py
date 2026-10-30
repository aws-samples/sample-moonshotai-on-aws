# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Turn document scans into Chat Completions image content parts."""

import base64
import io
from pathlib import Path

from PIL import Image

# Downscale large scans and fall back to JPEG for very large images. This keeps
# latency and input tokens down without hurting legibility of typical A4 documents.
MAX_IMAGE_BYTES = 3_750_000
MAX_IMAGE_SIDE = 2048


def load_image(image: str | Path | bytes) -> dict:
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
        "image_url": {
            "url": f"data:image/{image_format};base64,{encoded}",
            "detail": "high",
        },
    }
