"""Data contracts shared by the API and the agents.

Every LLM output is parsed into one of these models, so a malformed answer is
caught immediately (and repaired) instead of breaking a later step.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

SectionFormat = Literal["prose", "bullets", "table", "mixed"]
TaskStatus = Literal["pending", "in_progress", "done", "failed", "skipped"]


# --------------------------------------------------------------------------- #
# API input
# --------------------------------------------------------------------------- #
class AgentRequest(BaseModel):
    request: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language description of the document you need.",
        examples=[
            "Write a project plan to migrate our on-premise CRM to Salesforce within "
            "4 months. Team of 5 engineers, budget $120,000."
        ],
    )
    locale: str | None = Field(
        default=None,
        max_length=100,
        description="Optional requester locale, e.g. 'en-IN; timezone Asia/Kolkata'. Used to "
                    "infer currency and conventions when the request doesn't state them.",
        examples=["en-IN; timezone Asia/Kolkata"],
    )


# --------------------------------------------------------------------------- #
# Planner output
# --------------------------------------------------------------------------- #
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_MONTH_FIRST = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_DAY_FIRST = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")


def _month(name: str) -> int | None:
    name = name.lower()
    return next((n for full, n in _MONTHS.items() if full.startswith(name) and len(name) >= 3), None)


def invalid_dates(text: str) -> list[str]:
    """Dates written in the text that don't exist, e.g. '2027-02-29' or 'April 31, 2027'."""
    found: list[tuple[str, int, int, int]] = []
    for m in _ISO_DATE.finditer(text):
        found.append((m.group(0), int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _MONTH_FIRST.finditer(text):
        if (month := _month(m.group(1))):
            found.append((m.group(0), int(m.group(3)), month, int(m.group(2))))
    for m in _DAY_FIRST.finditer(text):
        if (month := _month(m.group(2))):
            found.append((m.group(0), int(m.group(3)), month, int(m.group(1))))
    bad = []
    for raw, year, month, day in found:
        try:
            date(year, month, day)
        except ValueError:
            bad.append(raw)
    return bad


_ONE_NUMBER = re.compile(
    r"^[^\d]*?(\d[\d,]*(?:\.\d+)?)\s*(k|lakh|lakhs|l)?\b[^\d]*$", re.IGNORECASE
)
_MULTIPLIER = {"k": 1_000, "l": 100_000, "lakh": 100_000, "lakhs": 100_000}


def _amount(cell: str) -> float | None:
    """The single number in a table cell ('INR 1,20,000', '$15k', '35,000 (est.)'), else None."""
    text = cell.replace(" ", " ").replace(" ", " ").strip()
    if not text or "%" in text:
        return None
    match = _ONE_NUMBER.match(text)
    if not match:
        return None  # no number, or several numbers / a range: not checkable
    value = float(match.group(1).replace(",", ""))
    return value * _MULTIPLIER.get((match.group(2) or "").lower(), 1)


def table_total_errors(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Check that a final 'Total' row equals the sum of the rows above it, per numeric column.

    LLMs are unreliable at arithmetic; code is not. Only clear-cut columns are
    checked (every item row holds exactly one number), so prose tables are ignored.
    """
    if len(rows) < 3 or not rows[-1] or not rows[-1][0].strip().lower().startswith("total"):
        return []
    items = rows[:-1]
    if any("subtotal" in (row[0] or "").lower() for row in items):
        return []
    errors = []
    for col in range(1, len(headers)):
        total = _amount(rows[-1][col]) if col < len(rows[-1]) else None
        values = [_amount(row[col]) for row in items]
        if total is None or any(v is None for v in values):
            continue
        summed = sum(values)
        if abs(summed - total) > max(1.0, total * 0.005):
            errors.append(
                f"table '{headers[0]}' column '{headers[col]}': the items add up to "
                f"{summed:,.0f} but the Total row says {total:,.0f}"
            )
    return errors


_MONEY = re.compile(
    r"(?:₹|INR|Rs\.?|\$|USD|US\$|€|EUR|£|GBP)\s*(\d[\d,]*(?:\.\d+)?)\s*"
    r"(crores?|cr|lakhs?|million|mn|m|k)?\b",
    re.IGNORECASE,
)
_SCALE = {"crore": 10**7, "crores": 10**7, "cr": 10**7, "lakh": 10**5, "lakhs": 10**5,
          "million": 10**6, "mn": 10**6, "m": 10**6, "k": 10**3}
_PART_WORDS = ("cost", "budget", "contingency", "reserve", "allocation", "fee", "spend",
               "expense")


def _money(text: str) -> float | None:
    match = _MONEY.search(text.replace(" ", " ").replace(" ", " "))
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    return value * _SCALE.get((match.group(2) or "").lower(), 1)


_PLACEHOLDER = re.compile(
    r"\[(?:insert|your|requester|client|company|name|tbd|tbc|xx)[^\]]{0,30}\]"
    r"|\b(?:City|Vendor|Company|Contractor|Supplier|Person|Client) [A-E]\b"
    r"|\bTBD\b|\bXXX+\b|\bLorem ipsum\b",
    re.IGNORECASE,
)


def placeholders(text: str) -> list[str]:
    """Template leftovers such as '[Requester Name]', 'City A' or 'TBD'."""
    return sorted({m.group(0) for m in _PLACEHOLDER.finditer(text)})


def total_budget(key_facts: list[str]) -> float | None:
    for fact in key_facts:
        label, sep, value = fact.partition(":")
        lowered = label.lower()
        if sep and "total" in lowered and ("budget" in lowered or "cost" in lowered):
            return _money(value)
    return None


def table_over_budget(headers: list[str], rows: list[list[str]], budget: float) -> list[str]:
    """Money columns whose items add up to more than the plan's total budget."""
    errors = []
    item_rows = [r for r in rows if not r[0].strip().lower().startswith(("total", "grand", "sub"))]
    if len(item_rows) < 2:
        return errors
    for col in range(1, len(headers)):
        header = headers[col].lower()
        if not any(w in header for w in ("cost", "amount", "budget", "inr", "₹", "$", "usd",
                                         "estimate", "price", "allocation", "spend")):
            continue
        if "per" in header or "%" in header or "rate" in header:
            continue  # per-person / per-unit columns are not totals
        values = [_money(r[col]) or _amount(r[col]) for r in item_rows]
        if any(v is None for v in values):
            continue
        if sum(values) > budget * 1.005:
            errors.append(f"column '{headers[col]}' adds up to {sum(values):,.0f}, which is more "
                          f"than the total budget {budget:,.0f}; the items must fit within it")
    return errors


def budget_overrun(key_facts: list[str]) -> str | None:
    """'Total budget: X' must not be smaller than the money parts listed beside it."""
    total = None
    parts: list[tuple[str, float]] = []
    for fact in key_facts:
        label, sep, value = fact.partition(":")
        if not sep or "%" in value.split("(")[0]:
            continue  # percentages describe shares, not amounts
        amount = _money(value)
        if amount is None:
            continue
        lowered = label.lower()
        if "total" in lowered and ("budget" in lowered or "cost" in lowered):
            total = total or amount
        elif any(word in lowered for word in _PART_WORDS):
            parts.append((label.strip(), amount))
    if total is None or len(parts) < 2:
        return None
    summed = sum(a for _, a in parts)
    if summed > total * 1.005:
        listed = " + ".join(f"{label} {a:,.0f}" for label, a in parts)
        return (f"key_facts budget parts add up to {summed:,.0f} ({listed}), which is more than "
                f"the total budget {total:,.0f}. Adjust the parts so they fit within the total")
    return None


def _clean_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in value if str(v).strip()]


class SectionSpec(BaseModel):
    id: int = Field(ge=1)
    title: str = Field(min_length=2, max_length=120)
    objective: str = Field(min_length=5, max_length=800)
    preferred_format: SectionFormat = "mixed"

    @field_validator("preferred_format", mode="before")
    @classmethod
    def _normalise_format(cls, v: Any) -> str:
        v = str(v or "mixed").strip().lower()
        return v if v in {"prose", "bullets", "table", "mixed"} else "mixed"


class Plan(BaseModel):
    document_type: str = Field(min_length=2, max_length=80)
    title: str = Field(min_length=3, max_length=160)
    audience: str = Field(min_length=2, max_length=200)
    tone: str = Field(default="professional", max_length=80)
    intent_summary: str = Field(min_length=5, max_length=600)
    assumptions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    # Shared "fact sheet": every writer uses these exact numbers, names and dates,
    # so sections written in parallel cannot contradict each other.
    key_facts: list[str] = Field(default_factory=list)
    sections: list[SectionSpec] = Field(min_length=3, max_length=10)

    @field_validator("assumptions", "conflicts", mode="before")
    @classmethod
    def _clean_lists(cls, v: Any) -> list[str]:
        return _clean_text_list(v)[:10]

    @field_validator("key_facts", mode="before")
    @classmethod
    def _clean_facts(cls, v: Any) -> list[str]:
        if isinstance(v, dict):  # {"Budget": "$20k"} -> ["Budget: $20k"]
            v = [f"{k}: {val}" for k, val in v.items()]
        return _clean_text_list(v)[:15]

    @model_validator(mode="after")
    def _dates_must_exist(self) -> "Plan":
        # LLMs are unreliable with calendars (e.g. 2027-02-29). Catching it here sends the
        # error back to the planner through the self-repair loop, before any writing starts,
        # because an invalid date in key_facts would otherwise spread to every section.
        texts = self.key_facts + self.assumptions + [s.objective for s in self.sections]
        bad = sorted({d for text in texts for d in invalid_dates(text)})
        if bad:
            raise ValueError(
                f"these dates do not exist on the calendar: {', '.join(bad)}. "
                "Replace them with real dates (e.g. use the last valid day of the month)."
            )
        if overrun := budget_overrun(self.key_facts):
            raise ValueError(overrun)
        found = placeholders(" ".join(texts + [self.title]))
        if found:
            raise ValueError(f"replace these placeholders with concrete values: {', '.join(found)}")
        return self

    @model_validator(mode="after")
    def _renumber_sections(self) -> "Plan":
        # LLMs sometimes skip or repeat ids; the rest of the pipeline relies on 1..n.
        for index, section in enumerate(self.sections, start=1):
            section.id = index
        return self


# --------------------------------------------------------------------------- #
# Writer output: a section is a list of typed content blocks
# --------------------------------------------------------------------------- #
class ParagraphBlock(BaseModel):
    type: Literal["paragraph"]
    text: str = Field(min_length=1)


class BulletsBlock(BaseModel):
    type: Literal["bullets"]
    items: list[str] = Field(min_length=1)

    @field_validator("items", mode="before")
    @classmethod
    def _clean_items(cls, v: Any) -> list[str]:
        return _clean_text_list(v)


class TableBlock(BaseModel):
    type: Literal["table"]
    headers: list[str] = Field(min_length=1, max_length=8)
    rows: list[list[str]] = Field(min_length=1, max_length=30)

    @field_validator("headers", mode="before")
    @classmethod
    def _headers_to_str(cls, v: Any) -> list[str]:
        return [str(h).strip() for h in (v or [])]

    @field_validator("rows", mode="before")
    @classmethod
    def _cells_to_str(cls, v: Any) -> list[list[str]]:
        rows = []
        for row in v or []:
            if isinstance(row, dict):  # {"Phase": "x", "Weeks": "y"} -> ["x", "y"]
                row = list(row.values())
            rows.append(["" if c is None else str(c).strip() for c in row])
        return rows

    @model_validator(mode="after")
    def _align_columns(self) -> "TableBlock":
        width = len(self.headers)
        self.rows = [(row + [""] * width)[:width] for row in self.rows]
        return self


ContentBlock = Annotated[
    Union[ParagraphBlock, BulletsBlock, TableBlock], Field(discriminator="type")
]


class SectionContent(BaseModel):
    blocks: list[ContentBlock] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def _facts_must_check_out(self, info: ValidationInfo) -> "SectionContent":
        # Deterministic checks for what LLMs get wrong; failures go back to the writer
        # through the self-repair loop with the exact problem described.
        problems = []
        text = self.to_text()
        bad_dates = sorted(set(invalid_dates(text)))
        if bad_dates:
            problems.append(f"these dates do not exist on the calendar: {', '.join(bad_dates)}")
        found = placeholders(text)
        if found:
            problems.append(f"replace these placeholders with concrete values: {', '.join(found)}")
        budget = (info.context or {}).get("budget_total") if info else None
        for block in self.blocks:
            if isinstance(block, TableBlock):
                problems.extend(table_total_errors(block.headers, block.rows))
                if budget:
                    problems.extend(table_over_budget(block.headers, block.rows, budget))
        if problems:
            raise ValueError("; ".join(problems) + ". Fix the numbers so they are consistent.")
        return self

    def word_count(self) -> int:
        return len(self.to_text().split())

    def to_text(self) -> str:
        """Plain-text rendering, used to show the draft to the reviewer."""
        lines: list[str] = []
        for block in self.blocks:
            if isinstance(block, ParagraphBlock):
                lines.append(block.text)
            elif isinstance(block, BulletsBlock):
                lines.extend(f"- {item}" for item in block.items)
            else:
                lines.append(" | ".join(block.headers))
                lines.extend(" | ".join(row) for row in block.rows)
        return "\n".join(lines)

    @classmethod
    def placeholder(cls, spec: SectionSpec, reason: str) -> "SectionContent":
        return cls(
            blocks=[
                ParagraphBlock(
                    type="paragraph",
                    text=(
                        f"[Draft pending] This section could not be generated automatically "
                        f"({reason}). Intended content: {spec.objective}"
                    ),
                )
            ]
        )


# --------------------------------------------------------------------------- #
# Reviewer output
# --------------------------------------------------------------------------- #
class ReviewIssue(BaseModel):
    section_id: int
    # Other sections that must change too (e.g. a date that disagrees across sections).
    related_section_ids: list[int] = Field(default_factory=list)
    severity: Literal["critical", "major", "minor"] = "major"
    problem: str = Field(min_length=3)
    fix_instruction: str = Field(min_length=3)

    @field_validator("related_section_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[int]:
        out = []
        for item in v or []:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out

    @field_validator("severity", mode="before")
    @classmethod
    def _normalise_severity(cls, v: Any) -> str:
        v = str(v or "major").strip().lower()
        return v if v in {"critical", "major", "minor"} else "major"


REQUIRED_CHECKS = (
    "coverage",          # every explicit requirement of the request is addressed
    "money",             # amounts per budget category, summed across ALL sections
    "dates",             # dates exist, are in order, planned items are in the future
    "consistency",       # names, roles, numbers and choices agree with key_facts
    "fit",               # audience, currency and tone suit who asked
    "facts",             # real-world facts are right (e.g. a city is in the state named)
)
SEVERITY_PENALTY = {"critical": 3.0, "major": 1.5, "minor": 0.5}


class ReviewCheck(BaseModel):
    name: str
    passed: bool
    evidence: str = Field(default="", description="What was verified, e.g. the sums")

    @field_validator("name", mode="before")
    @classmethod
    def _norm_name(cls, v: Any) -> str:
        return str(v or "").strip().lower()


class Review(BaseModel):
    """The reviewer reports checks and issues; the SCORE is computed by code from the issues.

    LLM self-scores are unreliable (a weak model happily gives 10/10 to a draft with
    double-counted money), so the model's number is ignored.
    """

    summary: str = Field(min_length=3)
    checks: list[ReviewCheck] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=8)
    score: int = Field(default=10, ge=1, le=10)

    @field_validator("strengths", mode="before")
    @classmethod
    def _clean_strengths(cls, v: Any) -> list[str]:
        return _clean_text_list(v)

    @field_validator("score", mode="before")
    @classmethod
    def _ignore_model_score(cls, v: Any) -> int:
        return 10  # replaced below; the model's own number is not trusted

    @model_validator(mode="after")
    def _compute_score(self) -> "Review":
        done = {c.name for c in self.checks}
        missing = [name for name in REQUIRED_CHECKS if name not in done]
        if missing:  # the checklist is what makes the review trustworthy
            raise ValueError(f"checks must include: {', '.join(missing)}")
        # A failed check must be backed by an issue, otherwise the review is inconsistent.
        failed = [c.name for c in self.checks if not c.passed]
        if failed and not self.issues:
            raise ValueError(f"checks {failed} failed but no issues were reported; "
                             "add an issue for each problem found")
        self.rescore()
        return self

    def rescore(self) -> None:
        """Score from the issues (call again if issues are filtered afterwards)."""
        penalty = sum(SEVERITY_PENALTY[i.severity] for i in self.issues)
        self.score = max(1, min(10, math.floor(10 - penalty)))


# --------------------------------------------------------------------------- #
# API output
# --------------------------------------------------------------------------- #
class TodoItem(BaseModel):
    id: int
    title: str
    agent: str
    status: TaskStatus = "pending"
    detail: str | None = None
    duration_ms: int | None = None


class PlanSummary(BaseModel):
    document_type: str
    title: str
    audience: str
    tone: str
    intent_summary: str
    assumptions: list[str]
    conflicts_resolved: list[str]
    key_facts: list[str]
    outline: list[str]


class ReviewReport(BaseModel):
    initial_score: int | None
    final_score: int | None
    threshold: int
    revision_rounds: int
    revised_sections: list[str]
    issues_found: list[str]
    strengths: list[str]
    summary: str
    notes: list[str] = Field(default_factory=list)
    remaining_issues: list[str] = Field(
        default_factory=list,
        description="Points the final review still flags (listed even when the score passes)",
    )
    checks: list[str] = Field(
        default_factory=list,
        description="The final review's checklist with the evidence the reviewer verified",
    )
    scoring: str = Field(
        default="Computed by code from the reviewer's issues: 10 minus 3 per critical, "
                "1.5 per major and 0.5 per minor issue, rounded down.",
    )


class DocumentInfo(BaseModel):
    filename: str
    download_url: str
    title: str
    document_type: str
    section_count: int
    word_count: int


class LLMUsage(BaseModel):
    calls: int
    retries: int
    json_repairs: int
    provider_fallbacks: int
    providers_used: list[str]
    calls_by_agent: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Successful calls per agent, per model"
    )
    total_tokens: int = 0
    tokens_by_model: dict[str, int] = Field(
        default_factory=dict, description="Tokens consumed per model (shows load distribution)"
    )


class AgentResponse(BaseModel):
    run_id: str
    status: Literal["completed", "completed_with_warnings"]
    summary: str
    document: DocumentInfo
    plan: PlanSummary
    todo: list[TodoItem]
    review: ReviewReport
    llm: LLMUsage
    warnings: list[str]
    duration_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    llm_chain: list[str] = Field(description="All configured models, in fallback order")
    agent_models: dict[str, list[str]] = Field(
        description="Preferred models per agent (writers are load-balanced across theirs)"
    )


class ErrorResponse(BaseModel):
    error: str
    detail: str


class JobCreated(BaseModel):
    job_id: str
    status_url: str = Field(description="Poll this URL to follow progress and get the result")


class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    todo: list[TodoItem] = Field(description="The agent's live task list")
    result: AgentResponse | None = None
    error: ErrorResponse | None = None


class DocumentListItem(BaseModel):
    filename: str
    download_url: str
    size_kb: float
    created_at: str
