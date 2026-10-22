"""Deterministic offline LLM.

Used (1) by the test-suite, (2) as the last link of the fallback chain so the
API still produces a usable document when every hosted model is down, and
(3) to run the project with no API key at all. It reads the same <context>
JSON the real models receive and answers in the same schemas, so the rest of
the pipeline cannot tell the difference.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.llm.providers import ChatProvider, Message

_CONTEXT = re.compile(r"<context>\s*(\{.*\})\s*</context>", re.DOTALL)

PEOPLE = ["Priya Sharma", "Arjun Mehta", "Meena Iyer", "Ravi Kumar", "Sara Thomas", "Daniel Reed"]

# (keywords, document type, [(section title, format, objective)])
TEMPLATES: list[tuple[tuple[str, ...], str, list[tuple[str, str, str]]]] = [
    (("minutes", "meeting notes"), "Meeting Minutes", [
        ("Meeting Overview", "table", "Date, time, location, chair and attendees."),
        ("Agenda", "bullets", "Items that were scheduled for discussion."),
        ("Discussion Summary", "mixed", "Key points raised for each agenda item."),
        ("Decisions Made", "bullets", "Decisions agreed by the group."),
        ("Action Items", "table", "Actions with owner, deadline and status."),
        ("Next Meeting", "prose", "Date and focus of the next meeting."),
    ]),
    (("sop", "standard operating", "procedure"), "Standard Operating Procedure", [
        ("Purpose and Scope", "prose", "Why the procedure exists and what it covers."),
        ("Roles and Responsibilities", "table", "Who does what in the procedure."),
        ("Prerequisites", "bullets", "Tools, access and inputs required before starting."),
        ("Procedure Steps", "table", "Numbered steps with owner and expected output."),
        ("Quality Checks", "bullets", "Controls that confirm the procedure was followed."),
        ("Revision History", "table", "Version, date, author and change summary."),
    ]),
    (("technical design", "architecture", "system design", "design doc"), "Technical Design Document", [
        ("Overview", "prose", "Problem being solved and the proposed system."),
        ("Requirements", "bullets", "Functional and non-functional requirements."),
        ("Architecture", "mixed", "Main components and how they interact."),
        ("Technology Choices", "table", "Chosen technologies with rationale."),
        ("Risks and Mitigations", "table", "Technical risks with likelihood, impact and mitigation."),
        ("Delivery Plan", "table", "Milestones with owners and target dates."),
    ]),
    (("product spec", "specification", "prd", "product requirement"), "Product Specification", [
        ("Product Overview", "prose", "What the product is and the problem it solves."),
        ("Target Users", "bullets", "Primary personas and their needs."),
        ("Functional Requirements", "table", "Requirements with priority and acceptance criteria."),
        ("Non-Functional Requirements", "bullets", "Performance, security and availability targets."),
        ("Success Metrics", "table", "KPIs with baseline and target."),
        ("Release Plan", "table", "Phases with scope and target dates."),
    ]),
    (("proposal", "propose", "pitch"), "Business Proposal", [
        ("Executive Summary", "prose", "The recommendation and why it matters, in brief."),
        ("Problem Statement", "mixed", "Current situation and the cost of doing nothing."),
        ("Proposed Solution", "mixed", "What will be delivered and how."),
        ("Cost Estimate", "table", "Cost line items with estimates and totals."),
        ("Implementation Timeline", "table", "Phases with duration, owner and outcome."),
        ("Risks and Mitigations", "table", "Key risks with impact and mitigation."),
        ("Recommendation and Next Steps", "bullets", "Decision requested and immediate actions."),
    ]),
    (("project plan", "plan", "roadmap", "migrat", "launch", "rollout"), "Project Plan", [
        ("Executive Summary", "prose", "Goal, scope and expected outcome of the project."),
        ("Objectives and Scope", "bullets", "Measurable objectives and what is in or out of scope."),
        ("Timeline and Milestones", "table", "Phases and milestones with dates and owners."),
        ("Team and Responsibilities", "table", "Roles with named owners and responsibilities."),
        ("Budget", "table", "Budget line items with estimates."),
        ("Risks and Mitigations", "table", "Risks with likelihood, impact and mitigation."),
        ("Success Metrics", "bullets", "KPIs used to judge success."),
    ]),
]
DEFAULT_TEMPLATE = ("Business Report", [
    ("Executive Summary", "prose", "Key findings and the recommendation in brief."),
    ("Background", "prose", "Context and why this report was produced."),
    ("Key Findings", "bullets", "The most important facts and observations."),
    ("Options Analysis", "table", "Options compared on cost, benefit and risk."),
    ("Risks", "table", "Main risks with impact and mitigation."),
    ("Recommendations", "bullets", "Recommended actions with owners."),
])

_BUDGET_TIGHT = ("tight", "cheap", "low cost", "low-cost", "under $", "limited budget", "minimal budget", "small budget")
_SCOPE_LARGE = ("enterprise", "premium", "24/7", "all features", "everything", "every language", "all languages", "languages")
_BRIEF = ("short", "brief", "concise", "one page", "one-page")
_EXHAUSTIVE = ("cover everything", "comprehensive", "in detail", "all aspects", "everything")


class MockProvider(ChatProvider):
    name = "mock-offline"
    offline = True

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.4,
        json_mode: bool = True,
        max_tokens: int = 2048,
    ) -> str:
        system = messages[0]["content"]
        context = _read_context(messages[1]["content"])
        if "AGENT: PLANNER" in system:
            result = _plan(context)
        elif "AGENT: WRITER" in system:
            result = _write(context)
        elif "AGENT: REVIEWER" in system:
            result = _review(context)
        else:
            result = {"error": "unknown agent"}
        return json.dumps(result)


def _read_context(text: str) -> dict[str, Any]:
    match = _CONTEXT.search(text)
    return json.loads(match.group(1)) if match else {}


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _topic(request: str) -> str:
    end = r"(?:[.,;:!?(]| - | — | with | within | by | in \d| so | but |$)"
    match = None
    # "about X" names the subject more reliably than "for X" ("for the board ...").
    for lead in (r"about|regarding", r"for|on|to"):
        match = re.search(rf"\b(?:{lead})\s+(.+?){end}", request, re.IGNORECASE)
        if match:
            break
    topic = match.group(1) if match else request
    topic = re.sub(r"^(our|the|a|an|my|new)\s+", "", topic.strip(), flags=re.IGNORECASE)
    words = topic.split()[:7]
    return " ".join(w if w.isupper() else w.capitalize() for w in words) or "Business Initiative"


def _plan(ctx: dict[str, Any]) -> dict[str, Any]:
    request = ctx.get("user_request", "")
    low = request.lower()
    doc_type, sections = DEFAULT_TEMPLATE
    for keywords, t_type, t_sections in TEMPLATES:
        if _has(low, keywords):
            doc_type, sections = t_type, t_sections
            break
    topic = _topic(request)

    assumptions = []
    if not _has(low, ("$", "budget", "cost", "usd", "inr", "₹", "k ")):
        assumptions.append("No budget was given; all figures are indicative estimates.")
    if not re.search(r"\b(day|week|month|quarter|q[1-4]|deadline|20\d\d|year)s?\b", low):
        assumptions.append("No timeline was given; a 12-week delivery window is assumed.")
    if not _has(low, ("team", "people", "engineer", "staff", "attendee", "members")):
        assumptions.append("A cross-functional team of 5-7 people is assumed.")
    assumptions.append("Names, dates and figures are illustrative mock data where none were supplied.")

    conflicts = []
    if _has(low, _BUDGET_TIGHT) and _has(low, _SCOPE_LARGE):
        conflicts.append("Tight budget vs. broad scope -> phased delivery: core scope within "
                         "budget first, extended scope as a separately costed Phase 2.")
    if _has(low, _BRIEF) and _has(low, _EXHAUSTIVE):
        conflicts.append("Brevity vs. full coverage -> short sections led by an executive "
                         "summary, with detail moved into tables.")
    if _has(low, ("not sure", "unsure", "something", " or a ")):
        conflicts.append(f"Document type unclear -> produced a {doc_type} because the request "
                         "asks for a decision backed by costs, risks and a plan.")

    return {
        "document_type": doc_type,
        "title": f"{topic} - {doc_type}",
        "audience": "Leadership and project stakeholders",
        "tone": "professional and concise",
        "intent_summary": f"Give stakeholders a clear, actionable {doc_type.lower()} for {topic}.",
        "assumptions": assumptions,
        "conflicts": conflicts,
        "key_facts": [
            f"Project sponsor: {PEOPLE[0]}, Head of Operations",
            f"Delivery lead: {PEOPLE[1]}, Programme Manager",
            "Indicative budget: $70,400 including 10% contingency",
            "Delivery window: 12 weeks in four phases",
        ],
        "sections": [
            {"id": i, "title": t, "objective": o, "preferred_format": f}
            for i, (t, f, o) in enumerate(sections, start=1)
        ],
    }


def _table_for(title: str) -> dict[str, Any]:
    t = title.lower()
    if _has(t, ("timeline", "milestone", "delivery", "release", "implementation")):
        return {"type": "table", "headers": ["Phase", "Duration", "Owner", "Outcome"], "rows": [
            ["1. Discovery", "Weeks 1-2", PEOPLE[0], "Requirements signed off"],
            ["2. Build", "Weeks 3-8", PEOPLE[1], "Core scope delivered"],
            ["3. Test", "Weeks 9-10", PEOPLE[2], "UAT passed"],
            ["4. Launch", "Weeks 11-12", PEOPLE[3], "Live with hypercare"],
        ]}
    if _has(t, ("budget", "cost")):
        return {"type": "table", "headers": ["Line item", "Estimate (USD)", "Notes"], "rows": [
            ["People", "48,000", "Blended team rate"],
            ["Licences and tools", "12,000", "Annual subscriptions"],
            ["Training", "4,000", "End-user enablement"],
            ["Contingency (10%)", "6,400", "Held centrally"],
            ["Total", "70,400", "Indicative estimate"],
        ]}
    if "risk" in t:
        return {"type": "table", "headers": ["Risk", "Likelihood", "Impact", "Mitigation"], "rows": [
            ["Scope creep", "Medium", "High", "Change control board, weekly scope review"],
            ["Key-person dependency", "Medium", "Medium", "Pairing and documented runbooks"],
            ["Data quality issues", "High", "Medium", "Early data profiling and cleansing sprint"],
        ]}
    if _has(t, ("team", "role", "responsib", "action")):
        return {"type": "table", "headers": ["Owner", "Responsibility", "Due"], "rows": [
            [PEOPLE[0], "Overall delivery and stakeholder updates", "Weekly"],
            [PEOPLE[1], "Technical design and build", "End of week 8"],
            [PEOPLE[2], "Test strategy and sign-off", "End of week 10"],
        ]}
    if _has(t, ("option", "technology", "comparison", "requirement", "metric")):
        return {"type": "table", "headers": ["Item", "Description", "Priority", "Measure"], "rows": [
            ["A", "Core capability for the first release", "High", "Adoption > 80%"],
            ["B", "Integration with existing systems", "High", "Zero manual re-keying"],
            ["C", "Reporting and analytics", "Medium", "Weekly dashboard live"],
        ]}
    return {"type": "table", "headers": ["Item", "Detail"], "rows": [
        ["Date", "To be confirmed"], ["Owner", PEOPLE[0]], ["Status", "Draft"],
    ]}


def _write(ctx: dict[str, Any]) -> dict[str, Any]:
    section = ctx.get("section", {})
    title = section.get("title", "Section")
    objective = section.get("objective", "").rstrip(".")
    fmt = section.get("preferred_format", "mixed")
    doc_title = ctx.get("document", {}).get("title", "this document")

    paragraph = {"type": "paragraph", "text": (
        f"This section of **{doc_title}** covers: {objective[:1].lower() + objective[1:]}. "
        "It reflects the stated assumptions and is written so a stakeholder can act on it "
        "without further context."
    )}
    bullets = {"type": "bullets", "items": [
        f"Clear ownership: {PEOPLE[0]} is accountable for {title.lower()}.",
        "Progress is reviewed weekly against the agreed milestones.",
        "Open questions are logged and resolved within five working days.",
        "Any change to scope, cost or dates is approved by the steering group.",
    ]}

    if ctx.get("mode") == "revise":
        blocks = list(ctx.get("previous_blocks") or [paragraph])
        blocks.append({"type": "bullets", "items": [
            f"Accountable owner: {PEOPLE[1]}",
            "Target date: end of week 6",
            "Success measure: agreed KPI met and signed off by the sponsor",
        ]})
        return {"blocks": blocks}

    if fmt == "table":
        blocks = [paragraph, _table_for(title)]
    elif fmt == "bullets":
        blocks = [paragraph, bullets]
    elif fmt == "prose":
        blocks = [paragraph, {"type": "paragraph", "text": (
            "The approach prioritises early, visible results while keeping risk low, so "
            "decision-makers can confirm direction before larger commitments are made."
        )}]
    else:
        blocks = [paragraph, bullets]
    return {"blocks": blocks}


def _checks(failed: str | None = None) -> list[dict[str, Any]]:
    names = ("coverage", "money", "dates", "consistency", "fit", "facts")
    return [{"name": n, "passed": n != failed, "evidence": "verified offline"} for n in names]


def _review(ctx: dict[str, Any]) -> dict[str, Any]:
    # The score is computed by code from the issues: one critical -> 7, one minor -> 9.
    draft = ctx.get("draft", [])
    if ctx.get("review_round", 1) == 1 and draft:
        target = next(
            (s for s in draft if _has(s["title"].lower(), ("risk", "action", "recommend", "next"))),
            draft[-1],
        )
        return {
            "checks": _checks(failed="consistency"),
            "summary": "Solid structure, but some sections lack named owners and measurable deadlines.",
            "strengths": ["Logical structure", "Assumptions stated up front"],
            "issues": [{
                "section_id": target["section_id"],
                "severity": "critical",
                "problem": "No accountable owner or measurable deadline.",
                "fix_instruction": "Add a named owner, a target date and a success measure.",
            }],
        }
    return {
        "checks": _checks(),
        "summary": "Complete, consistent and actionable; ready to share with stakeholders.",
        "strengths": ["Covers every part of the request", "Concrete owners and dates"],
        "issues": [{
            "section_id": draft[0]["section_id"] if draft else 1,
            "severity": "minor",
            "problem": "The opening paragraph could be one sentence shorter.",
            "fix_instruction": "Tighten the first paragraph.",
        }],
    }
