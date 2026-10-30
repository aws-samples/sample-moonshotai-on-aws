# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Compare extracted fields against hand-labelled ground truth."""

import json
from pathlib import Path


def _normalise(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value).strip().lower().replace(" ", "").replace(",", "")
    try:
        return round(float(text), 2)
    except ValueError:
        return text


def compare_fields(extracted: dict, expected: dict) -> list:
    """Return one row per expected field: field, expected, extracted, match."""
    return [
        {
            "field": name,
            "expected": expected_value,
            "extracted": extracted.get(name),
            "match": _normalise(extracted.get(name)) == _normalise(expected_value),
        }
        for name, expected_value in expected.items()
    ]


def load_expected(image_path) -> dict:
    """Load sample_data/expected/<image stem>.json for a sample image."""
    image_path = Path(image_path)
    return json.loads(
        (image_path.parent / "expected" / f"{image_path.stem}.json").read_text(
            encoding="utf-8"
        )
    )
