"""Planner agent: request -> document plan (type, assumptions, conflicts, sections)."""

from __future__ import annotations

from datetime import date

from app.agents.amounts import describe_amounts
from app.agents.prompts import PLANNER_SYSTEM, with_context
from app.llm import LLMClient
from app.schemas import Plan
from app.trace import RunTrace


class PlannerAgent:
    name = "planner"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def plan(self, request: str, trace: RunTrace, locale: str | None = None) -> Plan:
        context = {"user_request": request, "today": date.today().isoformat()}
        if locale:
            context["requester_locale"] = locale
        if amounts := describe_amounts(request):
            # Exact, code-computed readings: the model must not re-count the digits.
            context["numbers_in_request"] = amounts
        user = with_context("Create the execution plan for this document request.", context)
        return await self.llm.generate(
            agent=self.name,
            system=PLANNER_SYSTEM,
            user=user,
            schema=Plan,
            trace=trace,
            temperature=0.3,
            max_tokens=1500,
        )
