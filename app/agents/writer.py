"""Writer agent: writes (or revises) one section as typed content blocks."""

from __future__ import annotations

from datetime import date

from app.agents.prompts import WRITER_SYSTEM, with_context
from app.llm import LLMClient
from app.schemas import Plan, ReviewIssue, SectionContent, SectionSpec, total_budget
from app.trace import RunTrace


class WriterAgent:
    name = "writer"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def write(
        self, request: str, plan: Plan, section: SectionSpec, trace: RunTrace
    ) -> SectionContent:
        context = self._base_context(request, plan, section)
        context["mode"] = "write"
        return await self._generate(f"Write section {section.id}: {section.title}.", context, trace)

    async def revise(
        self,
        request: str,
        plan: Plan,
        section: SectionSpec,
        current: SectionContent,
        issues: list[ReviewIssue],
        trace: RunTrace,
        other_sections: dict[str, str] | None = None,
    ) -> SectionContent:
        context = self._base_context(request, plan, section)
        context["mode"] = "revise"
        context["previous_blocks"] = current.model_dump()["blocks"]
        if other_sections:
            # Consistency issues span sections; the reviser must see what it must agree with.
            context["other_sections_for_consistency"] = other_sections
        context["reviewer_feedback"] = [
            {"problem": i.problem, "fix_instruction": i.fix_instruction} for i in issues
        ]
        return await self._generate(
            f"Revise section {section.id}: {section.title} to fix every reviewer issue.",
            context,
            trace,
        )

    async def _generate(self, instruction: str, context: dict, trace: RunTrace) -> SectionContent:
        # The plan's total budget lets code reject cost tables that exceed it.
        budget = total_budget(context["key_facts"])
        return await self.llm.generate(
            agent=self.name,
            system=WRITER_SYSTEM,
            user=with_context(instruction, context),
            schema=SectionContent,
            trace=trace,
            temperature=0.6,
            max_tokens=1800,
            validation_context={"budget_total": budget} if budget else None,
        )

    @staticmethod
    def _base_context(request: str, plan: Plan, section: SectionSpec) -> dict:
        return {
            "user_request": request,
            "today": date.today().isoformat(),
            "document": {
                "type": plan.document_type,
                "title": plan.title,
                "audience": plan.audience,
                "tone": plan.tone,
                "intent": plan.intent_summary,
                "assumptions": plan.assumptions,
                "conflict_resolutions": plan.conflicts,
            },
            "key_facts": plan.key_facts,
            # The full outline lets parallel writers avoid overlapping each other.
            "outline": [f"{s.id}. {s.title}" for s in plan.sections],
            "section": section.model_dump(),
        }
