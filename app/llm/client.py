"""Reliable structured generation on top of unreliable LLMs.

For every call the client:
1. sends the prompt to the first provider in the chain,
2. retries transient failures with exponential backoff (honouring Retry-After),
3. validates the answer against a Pydantic schema and, if it is malformed,
   shows the model its own output plus the exact validation error and asks
   for a corrected version ("self-repair"),
4. falls back to the next provider when one is exhausted or misconfigured.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.json_utils import describe_error, extract_json
from app.llm.providers import ChatProvider, FatalLLMError, Message, RetryableLLMError
from app.trace import RunTrace

logger = logging.getLogger("agent.llm")

T = TypeVar("T", bound=BaseModel)

REPAIR_PROMPT = (
    "Your previous reply could not be used: {error}. "
    "Return ONLY the corrected JSON object that follows the required schema. "
    "No markdown, no commentary."
)


class LLMUnavailableError(RuntimeError):
    """Every provider in the chain failed for this call."""


class LLMClient:
    def __init__(
        self,
        providers: list[ChatProvider],
        *,
        routes: dict[str, list[ChatProvider]] | None = None,
        balanced_agents: frozenset[str] = frozenset({"writer"}),
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_retry_wait: float = 30.0,
    ) -> None:
        if not providers:
            raise ValueError("LLMClient needs at least one provider")
        self.providers = providers
        self.routes = routes or {}
        # Agents whose pool is load-balanced (round-robin). For the others the pool
        # order is a preference: best model first, the rest only as fallback.
        self.balanced_agents = balanced_agents
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_retry_wait = max_retry_wait
        self._turns: dict[str, int] = defaultdict(int)
        # Shared across runs: a model that just returned 429 is skipped until it recovers.
        self._cooldown_until: dict[str, float] = {}

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    def chain_for(self, agent: str) -> list[ChatProvider]:
        """Provider order for one call.

        The agent's own pool comes first. For load-balanced agents (the writers) it
        is rotated round-robin, so repeated calls are spread evenly across models.
        Every other provider follows as a fallback, so routing never reduces reliability.
        """
        pool = self.routes.get(agent) or []
        if pool and agent in self.balanced_agents:
            start = self._turns[agent] % len(pool)
            self._turns[agent] += 1
            pool = pool[start:] + pool[:start]
        rest = [p for p in self.providers if p not in pool]
        # Within each group, models that are cooling down after a rate limit go last
        # (stable sort keeps the round-robin order otherwise). Offline stays last.
        ready_first = lambda p: (p.offline, self.cooldown_remaining(p) > 0)  # noqa: E731
        return sorted(pool, key=ready_first) + sorted(rest, key=ready_first)

    def describe_routes(self) -> dict[str, list[str]]:
        return {agent: [p.name for p in pool] for agent, pool in self.routes.items()}

    async def generate(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        schema: type[T],
        trace: RunTrace,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        allow_offline: bool = True,
        validation_context: dict | None = None,
    ) -> T:
        errors: list[str] = []
        chain = self.chain_for(agent)
        if not allow_offline and any(not p.offline for p in chain):
            # Only meaningful when real models exist; a fully offline demo still runs.
            chain = [p for p in chain if not p.offline]
        for index, provider in enumerate(chain):
            if index > 0:
                trace.provider_fallbacks += 1
                logger.info("[%s] %s: routing to %s", trace.run_id, agent, provider.name)
            real_alternatives = [p for p in chain[index + 1:] if not p.offline]

            messages: list[Message] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            attempt = 0
            while attempt <= self.max_retries:
                wait = self.cooldown_remaining(provider)
                if wait > 0:
                    # Rate-limited: use another real model now instead of idling, and
                    # never sit out a long (e.g. daily-quota) cooldown.
                    if wait > self.max_retry_wait or any(
                        self.cooldown_remaining(p) == 0 for p in real_alternatives
                    ):
                        break
                    await asyncio.sleep(wait)

                trace.llm_calls += 1
                try:
                    raw = await provider.chat(
                        messages, temperature=temperature, json_mode=True, max_tokens=max_tokens
                    )
                except RetryableLLMError as exc:
                    errors.append(str(exc))
                    delay = (exc.retry_after if exc.retry_after is not None
                             else min(self.base_delay * 2**attempt, self.max_retry_wait))
                    self._cooldown_until[provider.name] = time.monotonic() + delay
                    trace.retries += 1
                    logger.info("[%s] %s: %s busy (%s), cooling down %.1fs", trace.run_id, agent,
                                provider.name, str(exc)[:60], delay)
                    attempt += 1
                    continue
                except FatalLLMError as exc:
                    errors.append(str(exc))
                    # Bad key / retired model: stop sending it traffic for a while.
                    self._cooldown_until[provider.name] = time.monotonic() + 600
                    logger.warning("[%s] %s: %s", trace.run_id, agent, exc)
                    break

                trace.add_tokens(provider.name, getattr(raw, "tokens", 0))
                try:
                    result = schema.model_validate(extract_json(raw), context=validation_context)
                except (ValueError, ValidationError) as exc:
                    problem = describe_error(exc)
                    errors.append(f"{provider.name}: invalid output ({problem})")
                    attempt += 1
                    if attempt > self.max_retries:
                        break
                    trace.json_repairs += 1
                    messages = [
                        messages[0],
                        messages[1],
                        {"role": "assistant", "content": raw[:6000]},
                        {"role": "user", "content": REPAIR_PROMPT.format(error=problem)},
                    ]
                    continue

                trace.note_provider(provider.name, agent)
                if provider.offline and index > 0:
                    trace.warn("Hosted LLMs were unavailable for some steps; the offline model "
                               "was used, so parts of the document are generic.")
                return result

        last = errors[-1] if errors else "no provider responded"
        raise LLMUnavailableError(f"{agent}: all LLM providers failed (last error: {last})")

    def cooldown_remaining(self, provider: ChatProvider) -> float:
        return max(0.0, self._cooldown_until.get(provider.name, 0.0) - time.monotonic())

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()
