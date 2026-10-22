"""Planner agent: request -> document plan (type, assumptions, conflicts, sections)."""

from __future__ import annotations

from datetime import date

from app.agents.prompts import PLANNER_SYSTEM, with_context
from app.llm import LLMClient
from app.schemas import Plan
from app.trace import RunTrace


class PlannerAgent:
    name = "planner"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def plan(self, request: str, trace: RunTrace) -> Plan:
        user = with_context(
            "Create the execution plan for this document request.",
            {"user_request": request, "today": date.today().isoformat()},
        )
        return await self.llm.generate(
            agent=self.name,
            system=PLANNER_SYSTEM,
            user=user,
            schema=Plan,
            trace=trace,
            temperature=0.3,
            max_tokens=1500,
        )
