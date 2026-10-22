"""Data contracts shared by the API and the agents.

Every LLM output is parsed into one of these models, so a malformed answer is
caught immediately (and repaired) instead of breaking a later step.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

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
    def _dates_must_exist(self) -> "SectionContent":
        bad = sorted(set(invalid_dates(self.to_text())))
        if bad:
            raise ValueError(f"these dates do not exist on the calendar: {', '.join(bad)}")
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


class Review(BaseModel):
    score: int = Field(ge=1, le=10)
    summary: str = Field(min_length=3)
    strengths: list[str] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=8)

    @field_validator("strengths", mode="before")
    @classmethod
    def _clean_strengths(cls, v: Any) -> list[str]:
        return _clean_text_list(v)


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


class ErrorResponse(BaseModel):
    error: str
    detail: str
