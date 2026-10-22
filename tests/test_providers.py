"""HTTP provider against a simulated OpenAI-compatible server (no network)."""

import asyncio
import json

import httpx
import pytest

from app.llm.providers import FatalLLMError, OpenAICompatibleProvider, RetryableLLMError

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "return json"}]


def provider_returning(response: httpx.Response, seen: list | None = None,
                       model: str = "openai/gpt-oss-120b",
                       reasoning_effort: str | None = "low") -> OpenAICompatibleProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return response

    return OpenAICompatibleProvider("groq-test", "https://api.groq.com/openai/v1", model,
                                    api_key="test-key", transport=httpx.MockTransport(handler),
                                    reasoning_effort=reasoning_effort)


def chat(provider) -> str:
    return asyncio.run(provider.chat(MESSAGES, temperature=0.2, json_mode=True, max_tokens=500))


def test_success_sends_openai_payload_and_parses_content():
    seen: list[httpx.Request] = []
    ok = httpx.Response(200, json={"choices": [{"message": {"content": '{"a": 1}'}}]})
    assert chat(provider_returning(ok, seen)) == '{"a": 1}'

    request = seen[0]
    assert request.url.path == "/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] > 500  # headroom for reasoning tokens


def test_non_reasoning_model_gets_plain_payload():
    seen: list[httpx.Request] = []
    ok = httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    chat(provider_returning(ok, seen, model="llama3.1", reasoning_effort=None))
    body = json.loads(seen[0].content)
    assert "reasoning_effort" not in body and body["max_tokens"] == 500


def test_rate_limit_is_retryable_and_reads_retry_after():
    limited = httpx.Response(429, headers={"retry-after": "7"}, text="rate_limit_exceeded")
    with pytest.raises(RetryableLLMError) as info:
        chat(provider_returning(limited))
    assert info.value.retry_after == 7.0


def test_invalid_json_from_model_is_retryable():
    bad = httpx.Response(400, text='{"error": {"code": "json_validate_failed"}}')
    with pytest.raises(RetryableLLMError):
        chat(provider_returning(bad))


@pytest.mark.parametrize("status", [401, 404])
def test_auth_and_unknown_model_are_fatal(status):
    with pytest.raises(FatalLLMError):
        chat(provider_returning(httpx.Response(status, text="nope")))


def test_server_error_is_retryable():
    with pytest.raises(RetryableLLMError):
        chat(provider_returning(httpx.Response(503, text="overloaded")))
