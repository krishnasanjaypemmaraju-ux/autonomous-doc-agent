"""Spell out the numbers in a request so the LLM never has to count zeros.

LLMs misread long digit strings (in a real run, "10000000" became "10 crore" instead of
"1 crore"). Code converts each number exactly, in both Indian and international styles,
and the planner receives those readings alongside the request.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d{4,})(?:\.\d+)?(?![\w.])")


def indian_grouping(value: int) -> str:
    digits = str(value)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def _scaled(value: int, units: list[tuple[int, str]]) -> str:
    for size, name in units:
        if value >= size:
            amount = value / size
            text = f"{amount:.2f}".rstrip("0").rstrip(".")
            return f"{text} {name}"
    return f"{value:,}"


def describe_amounts(text: str) -> list[dict[str, str]]:
    """Exact readings of every number with 4+ digits, e.g. 10000000 -> 1 crore / 10 million."""
    readings = []
    seen = set()
    for match in _NUMBER.finditer(text):
        raw = match.group(1)
        value = int(raw.replace(",", ""))
        if value in seen or 1900 <= value <= 2100:  # skip duplicates and likely years
            continue
        seen.add(value)
        readings.append({
            "as_written": raw,
            "indian": f"{indian_grouping(value)} = "
                      + _scaled(value, [(10**7, "crore"), (10**5, "lakh"), (10**3, "thousand")]),
            "international": f"{value:,} = "
                             + _scaled(value, [(10**9, "billion"), (10**6, "million"), (10**3, "thousand")]),
        })
    return readings
