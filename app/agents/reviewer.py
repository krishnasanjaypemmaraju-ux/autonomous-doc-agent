"""Reviewer agent: scores the full draft and lists actionable issues (reflection)."""

from __future__ import annotations

from app.agents.prompts import REVIEWER_SYSTEM, with_context
from app.llm import LLMClient
from app.schemas import Plan, Review, SectionContent
from app.trace import RunTrace


class ReviewerAgent:
    name = "reviewer"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def review(
        self,
        request: str,
        plan: Plan,
        contents: dict[int, SectionContent],
        review_round: int,
        trace: RunTrace,
        previous: Review | None = None,
    ) -> Review:
        draft = [
            {"section_id": s.id, "title": s.title, "content": contents[s.id].to_text()}
            for s in plan.sections
        ]
        context = {
            "user_request": request,
            "review_round": review_round,
            "plan": {
                "document_type": plan.document_type,
                "title": plan.title,
                "audience": plan.audience,
                "assumptions": plan.assumptions,
                "conflict_resolutions": plan.conflicts,
                "key_facts": plan.key_facts,
            },
            "draft": draft,
        }
        if previous is not None:
            context["previous_score"] = previous.score
            context["previous_issues"] = [i.model_dump() for i in previous.issues]
        review = await self.llm.generate(
            agent=self.name,
            system=REVIEWER_SYSTEM,
            user=with_context("Review this draft against the request.", context),
            schema=Review,
            trace=trace,
            temperature=0.1,
            max_tokens=1200,
            # A canned review would fake the quality score; better to report none.
            allow_offline=False,
        )
        # Drop issues pointing at sections that do not exist.
        valid_ids = {s.id for s in plan.sections}
        review.issues = [i for i in review.issues if i.section_id in valid_ids]
        for issue in review.issues:
            issue.related_section_ids = [
                sid for sid in dict.fromkeys(issue.related_section_ids)
                if sid in valid_ids and sid != issue.section_id
            ]
        return review
