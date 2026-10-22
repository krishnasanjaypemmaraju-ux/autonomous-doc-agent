"""LLM providers.

Groq, Gemini and Ollama all expose an OpenAI-compatible `/chat/completions`
endpoint, so one small HTTP class covers all three. Errors are split into
"retry this provider" and "give up on this provider" so the client can decide
between retrying and falling back.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import httpx

Message = dict[str, str]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class LLMError(Exception):
    pass


class RetryableLLMError(LLMError):
    """Transient failure (rate limit, timeout, 5xx). Same provider may succeed later."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class FatalLLMError(LLMError):
    """Permanent failure for this provider (bad key, unknown model). Move to the next one."""


class ChatText(str):
    """Model reply text that also carries the token count reported by the API."""

    tokens: int = 0

    @classmethod
    def with_tokens(cls, text: str, tokens: int) -> "ChatText":
        reply = cls(text)
        reply.tokens = tokens
        return reply


class ChatProvider(ABC):
    name: str
    # Offline providers (the mock) are a last resort: the client never jumps to
    # them just because a real model is briefly rate-limited.
    offline: bool = False

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.4,
        json_mode: bool = True,
        max_tokens: int = 2048,
    ) -> str:
        """Return the assistant message text."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


class OpenAICompatibleProvider(ChatProvider):
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        # Reasoning models (gpt-oss, Gemini 2.5) think before answering and the
        # thinking counts toward max_tokens. Keep it short for speed and leave
        # headroom so the JSON answer is never cut off.
        self._reasoning_effort = reasoning_effort
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout, transport=transport)

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.4,
        json_mode: bool = True,
        max_tokens: int = 2048,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
            if self._reasoning_effort != "none":
                payload["max_tokens"] = max_tokens + 2048
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.post(self._url, json=payload)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise RetryableLLMError(f"{self.name}: network error ({type(exc).__name__})") from exc

        status = response.status_code
        if status == 200:
            try:
                data = response.json()
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise RetryableLLMError(f"{self.name}: unexpected response shape") from exc
            tokens = (data.get("usage") or {}).get("total_tokens") or 0
            return ChatText.with_tokens(text, int(tokens))

        body = response.text[:300]
        if status == 429 or status >= 500:
            retry_after = _retry_after(response)
            if status == 429 and _DAILY_QUOTA.search(response.text):
                # Daily free quota used up: rest this model until the provider says it
                # recovers (rolling windows often free up in minutes), else for an hour.
                retry_after = _try_again_in(response.text) or 3600.0
            raise RetryableLLMError(f"{self.name}: HTTP {status} {body}", retry_after=retry_after)
        # Groq rejects a completion with 400 json_validate_failed when the model
        # emits invalid JSON in JSON mode. That is the model's fault, not ours: retry.
        if status == 400 and "json_validate_failed" in body:
            raise RetryableLLMError(f"{self.name}: model produced invalid JSON")
        raise FatalLLMError(f"{self.name}: HTTP {status} {body}")

    async def aclose(self) -> None:
        await self._client.aclose()


_DAILY_QUOTA = re.compile(r"PerDay|per day|requests per day|RPD", re.IGNORECASE)
_RETRY_DELAY_IN_BODY =re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
_TRY_AGAIN_IN_BODY = re.compile(
    r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", re.IGNORECASE
)


def _try_again_in(body: str) -> float | None:
    """Parse 'Please try again in 1h2m3.5s' / '5m30s' / '16.04s' into seconds."""
    match = _TRY_AGAIN_IN_BODY.search(body)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)


def _retry_after(response: httpx.Response) -> float | None:
    """How long the provider asked us to wait: header (Groq) or body (Gemini)."""
    value = response.headers.get("retry-after")
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass
    body = response.text
    if match := _RETRY_DELAY_IN_BODY.search(body):
        return float(match.group(1))
    return _try_again_in(body)
