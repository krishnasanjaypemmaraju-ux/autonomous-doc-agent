"""Cheap, deterministic checks that run before any LLM call."""

from __future__ import annotations

import re

MIN_WORDS = 3
MIN_CHARS = 10

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+.{0,30}(instructions|rules|prompt)",
    r"(reveal|print|show|leak)\s+.{0,30}(system\s+prompt|instructions|api[\s_-]?key|secret)",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|dan)\b",
]

_BLOCKED_PATTERNS = [
    r"\b(make|build|synthesi[sz]e|manufacture)\b.{0,40}\b(bomb|explosive|nerve\s+agent|bioweapon|chemical\s+weapon)s?\b",
    r"\b(write|create|develop)\b.{0,30}\b(malware|ransomware|keylogger)\b",
    r"\b(phishing\s+(email|page|kit))\b",
]


class GuardrailViolation(ValueError):
    """Raised when a request is rejected before it reaches the agents."""


def validate_request(raw: str) -> str:
    """Return a cleaned request or raise GuardrailViolation with a helpful message."""
    text = re.sub(r"\s+", " ", raw or "").strip()

    if len(text) < MIN_CHARS or len(text.split()) < MIN_WORDS:
        raise GuardrailViolation(
            "Request is too short to act on. Describe the document you need, "
            "e.g. 'Write a project plan for migrating our CRM to Salesforce'."
        )

    letters = sum(ch.isalpha() for ch in text)
    if letters / len(text) < 0.5:
        raise GuardrailViolation("Request does not look like natural language.")

    lowered = text.lower()
    if any(re.search(p, lowered) for p in _INJECTION_PATTERNS):
        raise GuardrailViolation(
            "Request contains instructions that try to override the agent's rules. "
            "Please describe only the document you need."
        )
    if any(re.search(p, lowered) for p in _BLOCKED_PATTERNS):
        raise GuardrailViolation("Request asks for harmful content and was declined.")

    return text
