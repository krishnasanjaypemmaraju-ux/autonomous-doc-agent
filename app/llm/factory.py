"""Build the LLM providers and decide which model serves which agent.

Each provider gets a short alias. Settings map every agent to a pool of aliases:

    planner  -> groq-large, gemini                 strongest models for the key decisions
    writer   -> gemini-lite, groq-small, groq-extra   most calls: rotated across three
                                                       independent rate-limit buckets
    reviewer -> groq-large, gemini                 a different model from the writers,
                                                    so the critic is independent

Free tiers limit each *model* separately (e.g. 8K tokens/minute per Groq model),
so spreading the work multiplies the usable capacity, and an outage at one
company only slows things down instead of stopping them.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.mock import MockProvider
from app.llm.providers import (
    GEMINI_BASE_URL,
    GROQ_BASE_URL,
    ChatProvider,
    OpenAICompatibleProvider,
)

logger = logging.getLogger("agent.llm")


def build_providers(settings: Settings) -> dict[str, ChatProvider]:
    """All configured providers, keyed by alias, in default fallback order."""
    timeout = settings.llm_timeout_seconds
    providers: dict[str, ChatProvider] = {}

    def add(alias: str, base_url: str, model: str | None, key: str | None, prefix: str) -> None:
        if not model or any(getattr(p, "model", None) == model for p in providers.values()):
            return
        providers[alias] = OpenAICompatibleProvider(
            f"{prefix}/{model}", base_url, model, key, timeout, reasoning_effort=_effort(model)
        )

    if settings.groq_api_key:
        add("groq-large", GROQ_BASE_URL, settings.groq_model, settings.groq_api_key, "groq")
        add("groq-small", GROQ_BASE_URL, settings.groq_fallback_model, settings.groq_api_key, "groq")
        add("groq-extra", GROQ_BASE_URL, settings.groq_extra_model, settings.groq_api_key, "groq")

    if settings.gemini_api_key:
        add("gemini", GEMINI_BASE_URL, settings.gemini_model, settings.gemini_api_key, "gemini")
        add("gemini-lite", GEMINI_BASE_URL, settings.gemini_lite_model, settings.gemini_api_key,
            "gemini")
        add("gemini-backup", GEMINI_BASE_URL, settings.gemini_backup_model,
            settings.gemini_api_key, "gemini")

    if settings.ollama_enabled:
        add("ollama", settings.ollama_base_url, settings.ollama_model, None, "ollama")

    if settings.mock_fallback or not providers:
        if not providers:
            logger.warning("No LLM API key configured - running on the offline mock LLM.")
        providers["mock"] = MockProvider()

    return providers


def build_routes(settings: Settings, providers: dict[str, ChatProvider]) -> dict[str, list[ChatProvider]]:
    """Preferred provider pool for each agent. Aliases that aren't configured are skipped."""
    wanted = {
        "planner": settings.planner_models,
        "writer": settings.writer_models,
        "reviewer": settings.reviewer_models,
    }
    routes: dict[str, list[ChatProvider]] = {}
    for agent, spec in wanted.items():
        aliases = [a.strip() for a in spec.split(",") if a.strip()]
        unknown = [a for a in aliases if a not in providers]
        if unknown:
            logger.info("%s: skipping unconfigured model(s) %s", agent, ", ".join(unknown))
        pool = [providers[a] for a in aliases if a in providers]
        if pool:
            routes[agent] = pool
    return routes


def _effort(model: str) -> str | None:
    """Reasoning setting per model family (None = parameter not sent)."""
    name = model.lower()
    if "gpt-oss" in name:
        return "low"
    if "qwen3" in name:
        return "none"  # skip <think> output; plain JSON answer
    if name.startswith("gemini") and "lite" not in name:
        return "low"
    return None
