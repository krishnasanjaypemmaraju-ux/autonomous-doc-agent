"""Per-run state: the agent's live TODO list, warnings and LLM usage stats."""

from __future__ import annotations

import logging
import time
import uuid

from app.schemas import LLMUsage, TaskStatus, TodoItem

logger = logging.getLogger("agent")


class RunTrace:
    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.todo: list[TodoItem] = []
        self.warnings: list[str] = []
        self.llm_calls = 0
        self.retries = 0
        self.json_repairs = 0
        self.provider_fallbacks = 0
        self.providers_used: list[str] = []
        # {"writer": {"gemini/...": 4, "groq/...": 3}, ...} - successful calls per model
        self.calls_by_agent: dict[str, dict[str, int]] = {}
        self.tokens_by_model: dict[str, int] = {}
        self._started_at = time.perf_counter()
        self._task_starts: dict[int, float] = {}

    # ---- TODO list -------------------------------------------------------- #
    def add_task(self, title: str, agent: str) -> TodoItem:
        item = TodoItem(id=len(self.todo) + 1, title=title, agent=agent)
        self.todo.append(item)
        logger.info("[%s] + TODO #%d %s (%s)", self.run_id, item.id, title, agent)
        return item

    def start(self, item: TodoItem) -> None:
        item.status = "in_progress"
        self._task_starts[item.id] = time.perf_counter()
        logger.info("[%s] > #%d %s", self.run_id, item.id, item.title)

    def done(self, item: TodoItem, detail: str | None = None) -> None:
        self._close(item, "done", detail)

    def fail(self, item: TodoItem, detail: str) -> None:
        self._close(item, "failed", detail)

    def skip(self, item: TodoItem, detail: str) -> None:
        self._close(item, "skipped", detail)

    def _close(self, item: TodoItem, status: TaskStatus, detail: str | None) -> None:
        item.status = status
        item.detail = detail
        started = self._task_starts.pop(item.id, None)
        if started is not None:
            item.duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info("[%s] %s #%d %s %s", self.run_id, status.upper(), item.id, item.title,
                    f"- {detail}" if detail else "")

    # ---- Diagnostics ------------------------------------------------------ #
    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            logger.warning("[%s] %s", self.run_id, message)

    def note_provider(self, name: str, agent: str = "unknown") -> None:
        if name not in self.providers_used:
            self.providers_used.append(name)
        calls = self.calls_by_agent.setdefault(agent, {})
        calls[name] = calls.get(name, 0) + 1

    def add_tokens(self, provider: str, tokens: int) -> None:
        if tokens:
            self.tokens_by_model[provider] = self.tokens_by_model.get(provider, 0) + tokens

    def usage(self) -> LLMUsage:
        return LLMUsage(
            calls=self.llm_calls,
            retries=self.retries,
            json_repairs=self.json_repairs,
            provider_fallbacks=self.provider_fallbacks,
            providers_used=self.providers_used,
            calls_by_agent=self.calls_by_agent,
            total_tokens=sum(self.tokens_by_model.values()),
            tokens_by_model=self.tokens_by_model,
        )

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._started_at) * 1000)
