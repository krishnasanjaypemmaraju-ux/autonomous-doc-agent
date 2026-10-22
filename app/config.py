"""Application settings, loaded from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM providers (tried in this order: Groq -> Gemini -> Ollama -> offline mock) ---
    groq_api_key: str | None = None
    # Llama 3.x models were retired on Groq's free tier (Aug 2026); gpt-oss is the
    # recommended replacement. Check https://console.groq.com/docs/models if this changes.
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str | None = "openai/gpt-oss-20b"

    # Third Groq model = third independent 8K tokens/minute bucket.
    groq_extra_model: str | None = "qwen/qwen3.8-27b"

    gemini_api_key: str | None = None
    # gemini-2.5-flash allows only 20 free requests/day; the 3.5 models have their own quotas.
    gemini_model: str = "gemini-3.5-flash"
    gemini_lite_model: str | None = "gemini-3.5-flash-lite"

    ollama_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1"

    # Offline mock LLM used as the last resort so the API never hard-fails in a demo.
    mock_fallback: bool = True

    # --- Which model serves which agent ---
    # Aliases: groq-large, groq-small, groq-extra, gemini, gemini-lite, ollama.
    # Writers make most of the calls, so they rotate (round-robin) across two
    # providers' independent rate-limit buckets. Planner/reviewer lists are a
    # preference order. Any configured model not listed (e.g. groq-extra) is still
    # used as a fallback.
    planner_models: str = "groq-large,gemini"
    writer_models: str = "gemini-lite,groq-small"
    reviewer_models: str = "groq-large,gemini"

    # --- Reliability ---
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=5, ge=0, le=8)
    # Free-tier per-minute limits recover within 60s; wait that long before the offline model.
    max_retry_wait_seconds: float = Field(default=60.0, ge=0)

    # --- Agent behaviour ---
    max_parallel_writers: int = Field(default=2, ge=1, le=8)
    quality_threshold: int = Field(default=8, ge=1, le=10)
    max_revision_rounds: int = Field(default=2, ge=0, le=3)

    output_dir: Path = BASE_DIR / "outputs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
