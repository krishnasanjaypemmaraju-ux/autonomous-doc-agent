"""Turn raw LLM text into JSON, tolerating the usual wrapping noise."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object from model output.

    Handles ```json fences and leading/trailing prose such as
    "Here is the plan: {...}". Raises ValueError if no object can be parsed.
    """
    text = _FENCE.sub("", (raw or "").strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg} at char {exc.pos}") from None
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object at the top level")
    return data


def describe_error(exc: Exception) -> str:
    """Short, model-readable description of what was wrong with the output."""
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:6]:
            location = ".".join(str(p) for p in err["loc"]) or "<root>"
            parts.append(f"{location}: {err['msg']}")
        return "; ".join(parts)
    return str(exc)
