from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.mock import MockProvider
from app.llm.providers import ChatProvider, FatalLLMError, RetryableLLMError
from app.main import create_app

STANDARD_REQUEST = (
    "Write a project plan to migrate our on-premise CRM to Salesforce within 4 months. "
    "Team of 5 engineers, budget $120,000. Include timeline, risks and responsibilities."
)
COMPLEX_REQUEST = (
    "We need something for the board next week about moving to AI customer support - "
    "short but cover everything: costs, risks, vendors and rollout. Budget is tight, under "
    "$20k, but leadership wants 24/7 coverage in 12 languages from day one. Not sure if "
    "this should be a proposal or a report."
)


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(_env_file=None, groq_api_key=None, gemini_api_key=None,
                output_dir=tmp_path, llm_max_retries=2)
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def client(tmp_path):
    app = create_app(make_settings(tmp_path), providers=[MockProvider()])
    with TestClient(app) as c:
        yield c


class ScriptedProvider(ChatProvider):
    """Plays back a fixed list of replies; an Exception instance is raised instead."""

    def __init__(self, name: str, replies: list) -> None:
        self.name = name
        self.replies = list(replies)
        self.calls = 0

    async def chat(self, messages, **kwargs) -> str:
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


__all__ = ["ScriptedProvider", "FatalLLMError", "RetryableLLMError", "make_settings",
           "STANDARD_REQUEST", "COMPLEX_REQUEST"]


def checks(failed: str | None = None) -> list[dict]:
    """A complete reviewer checklist (the Review schema requires all five checks)."""
    return [{"name": n, "passed": n != failed, "evidence": "test"}
            for n in ("coverage", "money", "dates", "consistency", "fit", "facts")]
