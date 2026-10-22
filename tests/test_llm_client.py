"""Retry, self-repair and provider-fallback behaviour of the LLM client."""

import asyncio
import json

import pytest

from app.llm.client import LLMClient, LLMUnavailableError
from app.schemas import Review
from app.trace import RunTrace
from tests.conftest import FatalLLMError, RetryableLLMError, ScriptedProvider

VALID = json.dumps({"score": 8, "summary": "Good draft.", "issues": []})


def run(client: LLMClient, trace: RunTrace) -> Review:
    return asyncio.run(client.generate(
        agent="reviewer", system="AGENT: REVIEWER", user="x", schema=Review, trace=trace))


def test_accepts_fenced_json_with_prose():
    provider = ScriptedProvider("p1", [f"Here you go:\n```json\n{VALID}\n```"])
    review = run(LLMClient([provider]), RunTrace())
    assert review.score == 8


def test_repairs_schema_violation_by_feeding_error_back():
    bad = json.dumps({"score": 42, "summary": "x"})  # score out of range
    provider = ScriptedProvider("p1", [bad, VALID])
    trace = RunTrace()
    review = run(LLMClient([provider], max_retries=2), trace)
    assert review.score == 8
    assert trace.json_repairs == 1 and provider.calls == 2


def test_retries_transient_errors_then_succeeds():
    provider = ScriptedProvider("p1", [RetryableLLMError("429", retry_after=0), VALID])
    trace = RunTrace()
    run(LLMClient([provider], max_retries=2, base_delay=0), trace)
    assert trace.retries == 1


def test_falls_back_to_next_provider_on_fatal_error():
    broken = ScriptedProvider("primary", [FatalLLMError("401 invalid key")])
    backup = ScriptedProvider("backup", [VALID])
    trace = RunTrace()
    run(LLMClient([broken, backup], base_delay=0), trace)
    assert trace.provider_fallbacks == 1
    assert trace.providers_used == ["backup"]
    assert trace.warnings == []  # switching between real models is normal, not a warning


def test_rate_limited_model_is_skipped_while_another_is_ready():
    busy = ScriptedProvider("busy", [RetryableLLMError("429", retry_after=60)])
    ready = ScriptedProvider("ready", [VALID, VALID])
    client = LLMClient([busy, ready], max_retries=3, base_delay=0)

    trace = RunTrace()
    run(client, trace)
    assert busy.calls == 1  # no waiting 60s on the busy model
    assert trace.providers_used == ["ready"]

    # Next call: the cooling model is moved behind the ready one, so it isn't even tried.
    assert [p.name for p in client.chain_for("reviewer")] == ["ready", "busy"]
    run(client, RunTrace())
    assert busy.calls == 1


def test_offline_fallback_is_flagged_as_warning():
    from app.llm.mock import MockProvider

    dead = ScriptedProvider("dead", [FatalLLMError("401")])
    trace = RunTrace()
    asyncio.run(LLMClient([dead, MockProvider()]).generate(
        agent="reviewer", system="AGENT: REVIEWER", user="<context>{}</context>",
        schema=Review, trace=trace))
    assert any("offline" in w for w in trace.warnings)


def test_token_usage_is_recorded_per_model():
    from app.llm.providers import ChatText

    provider = ScriptedProvider("p1", [ChatText.with_tokens(VALID, 1234)])
    trace = RunTrace()
    run(LLMClient([provider]), trace)
    assert trace.usage().tokens_by_model == {"p1": 1234}
    assert trace.usage().total_tokens == 1234


def test_falls_back_after_retries_are_exhausted():
    flaky = ScriptedProvider("primary", [RetryableLLMError("503")] * 3)
    backup = ScriptedProvider("backup", [VALID])
    trace = RunTrace()
    run(LLMClient([flaky, backup], max_retries=2, base_delay=0), trace)
    assert flaky.calls == 3 and backup.calls == 1


def test_raises_when_every_provider_fails():
    a = ScriptedProvider("a", [FatalLLMError("down")])
    b = ScriptedProvider("b", ["not json at all"] * 3)
    with pytest.raises(LLMUnavailableError):
        run(LLMClient([a, b], max_retries=2, base_delay=0), RunTrace())
