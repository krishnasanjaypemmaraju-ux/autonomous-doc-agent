"""Per-agent model routing and load distribution."""

import asyncio
import json

from app.llm.client import LLMClient
from app.llm.factory import build_providers, build_routes
from app.llm.mock import MockProvider
from app.orchestrator import DocumentAgent
from app.schemas import Review
from app.trace import RunTrace
from tests.conftest import STANDARD_REQUEST, FatalLLMError, ScriptedProvider, make_settings

VALID = json.dumps({"score": 9, "summary": "Good.", "issues": []})


class NamedMock(MockProvider):
    def __init__(self, name: str) -> None:
        self.name = name


def test_factory_assigns_models_per_agent(tmp_path):
    settings = make_settings(tmp_path, groq_api_key="g", gemini_api_key="m")
    providers = build_providers(settings)
    assert list(providers) == ["groq-large", "groq-small", "groq-extra", "gemini", "gemini-lite",
                              "gemini-backup", "mock"]

    routes = build_routes(settings, providers)
    assert [p.name for p in routes["planner"]] == ["groq/openai/gpt-oss-120b", "gemini/gemini-3.5-flash"]
    assert [p.name for p in routes["writer"]] == [
        "gemini/gemini-3.5-flash-lite", "groq/openai/gpt-oss-20b"]
    assert [p.name for p in routes["reviewer"]] == ["groq/openai/gpt-oss-120b", "gemini/gemini-3.5-flash"]


def test_routes_skip_models_that_are_not_configured(tmp_path):
    settings = make_settings(tmp_path, groq_api_key="g")  # no Gemini key
    routes = build_routes(settings, build_providers(settings))
    assert [p.name for p in routes["writer"]] == ["groq/openai/gpt-oss-20b"]


def test_reviewer_never_uses_offline_model_when_real_ones_exist():
    from app.llm import LLMUnavailableError

    dead = ScriptedProvider("dead", [FatalLLMError("down")])
    client = LLMClient([dead, MockProvider()], base_delay=0)
    import pytest
    with pytest.raises(LLMUnavailableError):
        asyncio.run(client.generate(agent="reviewer", system="AGENT: REVIEWER",
                                    user="<context>{}</context>", schema=Review,
                                    trace=RunTrace(), allow_offline=False))


def test_retry_wait_is_read_from_provider_messages():
    import httpx
    from app.llm.providers import OpenAICompatibleProvider, RetryableLLMError, _try_again_in

    assert _try_again_in("Please try again in 16.045s.") == 16.045
    assert _try_again_in("Please try again in 5m30.5s.") == 330.5
    assert _try_again_in("Please try again in 1h2m3s.") == 3723
    assert _try_again_in("no hint here") is None

    daily = ('{"error":{"message":"Rate limit reached for model `m` on tokens per day (TPD): '
             'Limit 200000, Used 199000, Requested 2500. Please try again in 7m12s."}}')
    provider = OpenAICompatibleProvider(
        "g", "https://x/v1", "m", "k",
        transport=httpx.MockTransport(lambda r: httpx.Response(429, text=daily)))
    import pytest
    with pytest.raises(RetryableLLMError) as info:
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
    assert info.value.retry_after == 432  # the provider's own estimate, not a blind hour


def test_daily_quota_429_rests_model_for_an_hour():
    import httpx
    from app.llm.providers import OpenAICompatibleProvider, RetryableLLMError

    body = '{"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}, {"retryDelay": "20s"}]}}'
    provider = OpenAICompatibleProvider(
        "g", "https://x/v1", "m", "k",
        transport=httpx.MockTransport(lambda r: httpx.Response(429, text=body)))
    import pytest
    with pytest.raises(RetryableLLMError) as info:
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
    assert info.value.retry_after == 3600.0


def test_writer_pool_rotates_round_robin_with_fallbacks_after():
    a, b, c = NamedMock("a"), NamedMock("b"), NamedMock("c")
    client = LLMClient([a, b, c], routes={"writer": [a, b]})
    first = [p.name for p in client.chain_for("writer")]
    second = [p.name for p in client.chain_for("writer")]
    assert first == ["a", "b", "c"]
    assert second == ["b", "a", "c"]
    assert [p.name for p in client.chain_for("planner")] == ["a", "b", "c"]  # unrouted agent


def test_planner_pool_is_a_preference_not_rotated():
    best, backup = NamedMock("best"), NamedMock("backup")
    client = LLMClient([best, backup], routes={"planner": [best, backup]})
    for _ in range(3):
        assert [p.name for p in client.chain_for("planner")] == ["best", "backup"]


def test_routed_agent_still_falls_back_when_its_model_fails():
    dead = ScriptedProvider("dead", [FatalLLMError("down")])
    backup = ScriptedProvider("backup", [VALID])
    client = LLMClient([dead, backup], routes={"reviewer": [dead]}, base_delay=0)
    trace = RunTrace()
    asyncio.run(client.generate(agent="reviewer", system="s", user="u", schema=Review, trace=trace))
    assert trace.calls_by_agent == {"reviewer": {"backup": 1}}


def test_full_run_spreads_writers_across_two_models(tmp_path):
    big, fast_a, fast_b = NamedMock("big"), NamedMock("fast-a"), NamedMock("fast-b")
    client = LLMClient(
        [big, fast_a, fast_b],
        routes={"planner": [big], "reviewer": [big], "writer": [fast_a, fast_b]},
    )
    result = asyncio.run(DocumentAgent(make_settings(tmp_path), client).run(STANDARD_REQUEST))

    by_agent = result.llm.calls_by_agent
    assert by_agent["planner"] == {"big": 1}
    assert set(by_agent["reviewer"]) == {"big"}
    writer_calls = by_agent["writer"]
    assert set(writer_calls) == {"fast-a", "fast-b"}
    assert abs(writer_calls["fast-a"] - writer_calls["fast-b"]) <= 1  # evenly balanced
