"""Orchestrator: coordinates Planner -> Writers -> Reviewer -> Renderer.

The TODO list is built by the agents' own decisions: the planner's outline
creates one writing task per section, and the reviewer's verdict decides
whether revision tasks get added at run time.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from datetime import datetime

from app.agents import PlannerAgent, ReviewerAgent, WriterAgent
from app.config import Settings
from app.guardrails import validate_request
from app.llm import LLMClient, LLMUnavailableError
from app.rendering.docx_renderer import RenderMeta, render_docx
from app.schemas import (
    AgentResponse,
    DocumentInfo,
    Plan,
    PlanSummary,
    Review,
    ReviewIssue,
    ReviewReport,
    SectionContent,
    SectionSpec,
)
from app.trace import RunTrace

logger = logging.getLogger("agent")


class DocumentAgent:
    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self.settings = settings
        self.llm = llm
        self.planner = PlannerAgent(llm)
        self.writer = WriterAgent(llm)
        self.reviewer = ReviewerAgent(llm)

    async def run(self, raw_request: str, trace: RunTrace | None = None) -> AgentResponse:
        """Run the full pipeline. Pass a trace to watch its TODO list update live."""
        trace = trace or RunTrace()

        # 1. Guardrails (raises GuardrailViolation -> HTTP 422)
        guard = trace.add_task("Validate request and apply guardrails", "guardrails")
        trace.start(guard)
        request = validate_request(raw_request)
        trace.done(guard, "request accepted")

        # 2. Plan (no plan = nothing to do, so failure here propagates -> HTTP 503)
        plan_task = trace.add_task("Analyse request and build execution plan", "planner")
        trace.start(plan_task)
        try:
            plan = await self.planner.plan(request, trace)
        except LLMUnavailableError as exc:
            trace.fail(plan_task, str(exc))
            raise
        trace.done(
            plan_task,
            f"{plan.document_type}: {len(plan.sections)} sections, "
            f"{len(plan.assumptions)} assumptions, {len(plan.conflicts)} conflicts resolved",
        )

        # 3. Write every section (parallel, bounded)
        contents = await self._write_sections(request, plan, trace)

        # 4. Reflect: review -> revise flagged sections -> re-review
        contents, report = await self._reflect(request, plan, contents, trace)

        # 5. Render the Word document
        render_task = trace.add_task("Render Microsoft Word document", "renderer")
        trace.start(render_task)
        filename = self._filename(plan.title, trace.run_id)
        path = self.settings.output_dir / filename
        await asyncio.to_thread(
            render_docx, plan, contents, report, self._meta(trace), path
        )
        word_count = sum(c.word_count() for c in contents.values())
        trace.done(render_task, f"{filename} ({word_count} words)")

        return self._response(plan, contents, report, trace, filename, word_count)

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    async def _write_sections(
        self, request: str, plan: Plan, trace: RunTrace
    ) -> dict[int, SectionContent]:
        semaphore = asyncio.Semaphore(self.settings.max_parallel_writers)
        tasks = {
            s.id: trace.add_task(f"Write section {s.id}: {s.title}", "writer")
            for s in plan.sections
        }

        async def write_one(section: SectionSpec) -> tuple[int, SectionContent]:
            item = tasks[section.id]
            async with semaphore:
                trace.start(item)
                try:
                    content = await self.writer.write(request, plan, section, trace)
                except LLMUnavailableError as exc:
                    # Recover: keep going with a clearly marked placeholder.
                    trace.fail(item, str(exc))
                    trace.warn(f"Section '{section.title}' could not be generated; "
                               "a placeholder was inserted.")
                    return section.id, SectionContent.placeholder(section, "LLM unavailable")
                trace.done(item, f"{len(content.blocks)} blocks, {content.word_count()} words")
                return section.id, content

        results = await asyncio.gather(*(write_one(s) for s in plan.sections))
        return dict(results)

    # ------------------------------------------------------------------ #
    # Reflection loop
    # ------------------------------------------------------------------ #
    async def _reflect(
        self,
        request: str,
        plan: Plan,
        contents: dict[int, SectionContent],
        trace: RunTrace,
    ) -> tuple[dict[int, SectionContent], ReviewReport]:
        threshold = self.settings.quality_threshold
        titles = {s.id: s.title for s in plan.sections}

        item = trace.add_task("Self-review the full draft against the request", "reviewer")
        initial = await self._safe_review(request, plan, contents, 1, item, trace)
        if initial is None:
            return contents, self._report(None, None, 0, [], [], threshold)

        best, best_review = contents, initial
        issues_found = [f"[{i.severity}] {titles[i.section_id]}: {i.problem}" for i in initial.issues]
        revised: list[str] = []
        notes: list[str] = []
        rounds = 0

        while (
            best_review.score < threshold
            and best_review.issues
            and rounds < self.settings.max_revision_rounds
        ):
            rounds += 1
            # Fix what matters first; minor polish only when nothing bigger is left.
            targets = [i for i in best_review.issues if i.severity != "minor"] or best_review.issues
            by_section: dict[int, list[ReviewIssue]] = defaultdict(list)
            for issue in targets:
                for sid in [issue.section_id, *issue.related_section_ids]:
                    by_section[sid].append(issue)

            candidate = await self._revise_sections(request, plan, best, by_section, trace)

            item = trace.add_task(f"Verify fixes: re-review revised draft (round {rounds + 1})",
                                  "reviewer")
            new_review = await self._safe_review(request, plan, candidate, rounds + 1, item, trace,
                                                 previous=best_review)
            if new_review is None:
                best = candidate  # revisions address concrete feedback; keep them
                revised += [titles[i] for i in by_section]
                break
            if new_review.score < best_review.score:
                notes.append(f"Round {rounds} revision scored lower ({best_review.score} -> "
                             f"{new_review.score}), so the previous version was kept.")
                logger.info("[%s] %s", trace.run_id, notes[-1])
                break
            best, best_review = candidate, new_review
            revised += [titles[i] for i in by_section]

        report = self._report(initial, best_review, rounds, revised, issues_found, threshold)
        report.notes = notes
        return best, report

    async def _revise_sections(
        self,
        request: str,
        plan: Plan,
        contents: dict[int, SectionContent],
        by_section: dict[int, list[ReviewIssue]],
        trace: RunTrace,
    ) -> dict[int, SectionContent]:
        sections = {s.id: s for s in plan.sections}
        semaphore = asyncio.Semaphore(self.settings.max_parallel_writers)
        updated = dict(contents)

        async def revise_one(section_id: int, issues: list[ReviewIssue]) -> None:
            section = sections[section_id]
            item = trace.add_task(f"Revise section {section_id}: {section.title}", "writer")
            others = {
                f"{sid}. {sections[sid].title}": text.to_text()[:1500]
                for sid, text in contents.items() if sid != section_id
            }
            async with semaphore:
                trace.start(item)
                try:
                    updated[section_id] = await self.writer.revise(
                        request, plan, section, contents[section_id], issues, trace,
                        other_sections=others,
                    )
                except LLMUnavailableError as exc:
                    trace.fail(item, f"kept original: {exc}")
                    return
                trace.done(item, f"fixed {len(issues)} issue(s)")

        await asyncio.gather(*(revise_one(sid, iss) for sid, iss in by_section.items()))
        return updated

    async def _safe_review(self, request, plan, contents, review_round, item, trace,
                           previous: Review | None = None) -> Review | None:
        trace.start(item)
        try:
            review = await self.reviewer.review(request, plan, contents, review_round, trace,
                                                previous=previous)
        except LLMUnavailableError as exc:
            trace.fail(item, str(exc))
            trace.warn("Quality review was skipped because the LLM was unavailable.")
            return None
        threshold = self.settings.quality_threshold
        verdict = "passes" if review.score >= threshold else "below"
        trace.done(item, f"score {review.score}/10 ({verdict} threshold {threshold}), "
                         f"{len(review.issues)} issue(s)")
        for issue in review.issues:
            logger.info("[%s]    issue [%s] section %d: %s", trace.run_id, issue.severity,
                        issue.section_id, issue.problem)
        return review

    @staticmethod
    def _report(initial, final, rounds, revised, issues_found, threshold) -> ReviewReport:
        return ReviewReport(
            initial_score=initial.score if initial else None,
            final_score=final.score if final else None,
            threshold=threshold,
            revision_rounds=rounds,
            revised_sections=list(dict.fromkeys(revised)),
            issues_found=issues_found,
            strengths=final.strengths if final else [],
            summary=final.summary if final else "Review unavailable.",
        )

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _filename(title: str, run_id: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60] or "document"
        return f"{slug}_{run_id[:8]}.docx"

    @staticmethod
    def _meta(trace: RunTrace) -> RenderMeta:
        return RenderMeta(
            run_id=trace.run_id,
            generated_at=datetime.now(),
            # The render task is still running while the file is written; it is
            # finished by the time anyone reads the appendix.
            todo=[
                t.model_copy(update={"status": "done"}) if t.status == "in_progress" else t.model_copy()
                for t in trace.todo
            ],
            providers=[
                f"{agent.capitalize()}: "
                + ", ".join(f"{name} ({count} call{'s' if count > 1 else ''})"
                            for name, count in models.items())
                for agent, models in trace.calls_by_agent.items()
            ],
        )

    def _response(
        self,
        plan: Plan,
        contents: dict[int, SectionContent],
        report: ReviewReport,
        trace: RunTrace,
        filename: str,
        word_count: int,
    ) -> AgentResponse:
        return AgentResponse(
            run_id=trace.run_id,
            status="completed_with_warnings" if trace.warnings else "completed",
            summary=_summary(plan, report, word_count),
            document=DocumentInfo(
                filename=filename,
                download_url=f"/documents/{filename}",
                title=plan.title,
                document_type=plan.document_type,
                section_count=len(plan.sections),
                word_count=word_count,
            ),
            plan=PlanSummary(
                document_type=plan.document_type,
                title=plan.title,
                audience=plan.audience,
                tone=plan.tone,
                intent_summary=plan.intent_summary,
                assumptions=plan.assumptions,
                conflicts_resolved=plan.conflicts,
                key_facts=plan.key_facts,
                outline=[f"{s.id}. {s.title}" for s in plan.sections],
            ),
            todo=trace.todo,
            review=report,
            llm=trace.usage(),
            warnings=trace.warnings,
            duration_ms=trace.elapsed_ms,
        )


def _summary(plan: Plan, report: ReviewReport, word_count: int) -> str:
    parts = [
        f"Planned and wrote a {len(plan.sections)}-section {plan.document_type} "
        f"\"{plan.title}\" ({word_count:,} words) for {plan.audience}."
    ]
    if plan.assumptions or plan.conflicts:
        parts.append(
            f"Made {len(plan.assumptions)} explicit assumption(s) and resolved "
            f"{len(plan.conflicts)} conflicting/ambiguous requirement(s)."
        )
    if report.initial_score is None:
        parts.append("Quality review was unavailable.")
    elif report.revised_sections:
        parts.append(
            f"Self-review scored the draft {report.initial_score}/10; after revising "
            f"{len(report.revised_sections)} section(s) it scored {report.final_score}/10."
        )
    else:
        parts.append(f"Self-review score: {report.final_score}/10.")
    return " ".join(parts)
