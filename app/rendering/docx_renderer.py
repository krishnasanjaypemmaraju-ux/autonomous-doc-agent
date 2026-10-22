"""Deterministic Word renderer: typed content blocks -> polished .docx.

The LLM never writes formatting. It returns structured blocks and this module
owns the layout, so every document is consistently styled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.document import Document as DocxDocument
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.table import _Cell
from docx.text.paragraph import Paragraph

from app.schemas import (
    BulletsBlock,
    ParagraphBlock,
    Plan,
    ReviewReport,
    SectionContent,
    TableBlock,
    TodoItem,
)

FONT = "Calibri"
PRIMARY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0x2E, 0x75, 0xB6)
MUTED = RGBColor(0x59, 0x59, 0x59)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PRIMARY_HEX, ACCENT_HEX = "1F3A5F", "2E75B6"
ZEBRA_HEX, CALLOUT_HEX, LABEL_HEX, BORDER_HEX = "F2F6FC", "EAF1FB", "E8EEF6", "BFCBDA"

_INLINE = re.compile(r"(\*\*[^*]+\*\*|(?<![*\w])\*[^*\s][^*]*?\*(?![*\w]))")
CONTENT_WIDTH_CM = 16.4  # A4 (21 cm) minus 2 x 2.3 cm margins
_STATUS_LABEL = {"done": "Done", "failed": "Failed", "skipped": "Skipped",
                 "pending": "Pending", "in_progress": "Running"}


@dataclass
class RenderMeta:
    run_id: str
    generated_at: datetime
    todo: list[TodoItem]
    providers: list[str]


def render_docx(
    plan: Plan,
    contents: dict[int, SectionContent],
    review: ReviewReport,
    meta: RenderMeta,
    path: Path,
) -> Path:
    doc = Document()
    _setup_styles(doc)
    _setup_page(doc, plan, meta)

    _cover(doc, plan, review, meta)
    _overview(doc, plan)
    for section in plan.sections:
        doc.add_heading(f"{section.id}. {section.title}", level=1)
        _render_blocks(doc, contents[section.id])
    _appendix(doc, review, meta)

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


# --------------------------------------------------------------------------- #
# Document skeleton
# --------------------------------------------------------------------------- #
def _setup_styles(doc: DocxDocument) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(11)
    normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for level, size in ((1, 16), (2, 13)):
        style = doc.styles[f"Heading {level}"]
        rfonts = style.element.get_or_add_rPr().get_or_add_rFonts()
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            rfonts.attrib.pop(qn(attr), None)  # theme fonts would override ours
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = PRIMARY
        style.paragraph_format.space_before = Pt(18 if level == 1 else 12)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True


def _setup_page(doc: DocxDocument, plan: Plan, meta: RenderMeta) -> None:
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Cm(2.2)
    section.left_margin = section.right_margin = Cm(2.3)
    section.different_first_page_header_footer = True  # clean cover page

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _styled_run(header, plan.title, size=9, color=MUTED)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _styled_run(footer, "Page ", size=9, color=MUTED)
    _add_field(footer, "PAGE")
    _styled_run(footer, " of ", size=9, color=MUTED)
    _add_field(footer, "NUMPAGES")
    _styled_run(footer, f"   |   Ref {meta.run_id}", size=9, color=MUTED)


def _cover(doc: DocxDocument, plan: Plan, review: ReviewReport, meta: RenderMeta) -> None:
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(110)

    label = doc.add_paragraph()
    _styled_run(label, plan.document_type.upper(), size=11, bold=True, color=ACCENT)

    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(10)
    _styled_run(title, plan.title, size=28, bold=True, color=PRIMARY)

    intent = doc.add_paragraph()
    _styled_run(intent, plan.intent_summary, size=13, italic=True, color=MUTED)

    rule = doc.add_paragraph()
    _bottom_border(rule, ACCENT_HEX)
    rule.paragraph_format.space_after = Pt(24)

    quality = (
        f"{review.final_score}/10 (automated review"
        + (f", {len(review.remaining_issues)} open point(s) listed in Appendix A)"
           if review.remaining_issues else ")")
        if review.final_score is not None else "Not reviewed"
    )
    rows = [
        ("Prepared for", plan.audience),
        ("Document type", plan.document_type),
        ("Date", meta.generated_at.strftime("%d %B %Y")),
        ("Quality review", quality),
        ("Reference", meta.run_id),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    _table_borders(table)
    for key, value in rows:
        cells = table.add_row().cells
        _shade(cells[0], LABEL_HEX)
        _styled_run(cells[0].paragraphs[0], key, size=10, bold=True, color=PRIMARY)
        _styled_run(cells[1].paragraphs[0], value, size=10)
    _set_col_widths(table, [Cm(4.5), Cm(11.5)])
    doc.add_page_break()


def _overview(doc: DocxDocument, plan: Plan) -> None:
    doc.add_heading("Contents", level=1)
    for section in plan.sections:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_after = Pt(2)
        _styled_run(p, f"{section.id}.  ", bold=True, color=ACCENT)
        _styled_run(p, section.title)

    if plan.key_facts:
        doc.add_heading("Key Facts at a Glance", level=1)
        rows = []
        for fact in plan.key_facts:
            label, sep, value = fact.partition(":")
            rows.append([label.strip(), value.strip()] if sep else ["", fact])
        _render_table(doc, TableBlock(type="table", headers=["Item", "Detail"], rows=rows),
                      widths=[Cm(5), Cm(11)])

    if plan.assumptions or plan.conflicts:
        doc.add_heading("Assumptions and Clarifications", level=1)
        intro = doc.add_paragraph()
        _styled_run(intro, "The request did not specify everything needed, so the agent made "
                           "the following explicit decisions. Please confirm or correct them.",
                    color=MUTED)
        box = doc.add_table(rows=1, cols=1)
        _table_borders(box, color=ACCENT_HEX)
        cell = box.rows[0].cells[0]
        _shade(cell, CALLOUT_HEX)
        first = cell.paragraphs[0]
        if plan.assumptions:
            _styled_run(first, "Assumptions", bold=True, color=PRIMARY)
            for item in plan.assumptions:
                _rich_text(cell.add_paragraph(style="List Bullet"), _sentence(item), size=10)
            first = None
        if plan.conflicts:
            heading = first or cell.add_paragraph()
            if first is None:
                heading.paragraph_format.space_before = Pt(8)
            _styled_run(heading, "Conflicting or ambiguous requirements - and how they were resolved",
                        bold=True, color=PRIMARY)
            for item in plan.conflicts:
                _rich_text(cell.add_paragraph(style="List Bullet"),
                           _sentence(item).replace("->", "→"), size=10)


def _appendix(doc: DocxDocument, review: ReviewReport, meta: RenderMeta) -> None:
    doc.add_page_break()
    doc.add_heading("Appendix A - How this document was produced", level=1)
    intro = doc.add_paragraph()
    _styled_run(intro, (
        "This document was planned, written and quality-checked autonomously by a multi-agent "
        "pipeline (Planner → parallel Writers → Reviewer). The table below is the "
        "agent's own task list, including tasks it added for itself during the run."
    ), color=MUTED, size=10)

    rows = [
        [str(t.id), t.title, t.agent.capitalize(), _STATUS_LABEL.get(t.status, t.status),
         f"{t.duration_ms / 1000:.1f}s" if t.duration_ms is not None else "-"]
        for t in meta.todo
    ]
    if rows:
        _render_table(doc, TableBlock(type="table", headers=["#", "Task", "Agent", "Status", "Time"],
                                      rows=rows), widths=[Cm(1), Cm(9.2), Cm(2.4), Cm(1.8), Cm(1.6)])

    doc.add_heading("Quality review", level=2)
    if review.initial_score is None:
        _rich_text(doc.add_paragraph(), "The automated review could not run for this document.")
    else:
        line = f"Initial score **{review.initial_score}/10**"
        if review.revised_sections:
            line += (f", final score **{review.final_score}/10** after revising: "
                     f"{', '.join(review.revised_sections)}.")
        elif review.revision_rounds:
            line += " - a revision was attempted (see note below)."
        else:
            line += " - no revision needed."
        _rich_text(doc.add_paragraph(), f"{line} Acceptance threshold: {review.threshold}/10.")
        if review.summary:
            _rich_text(doc.add_paragraph(), f"Reviewer summary: {review.summary}")
        for note in review.notes:
            _rich_text(doc.add_paragraph(), note)
        if review.checks:
            p = doc.add_paragraph()
            _styled_run(p, "What the reviewer verified:", bold=True, color=PRIMARY)
            for check in review.checks:
                _rich_text(doc.add_paragraph(style="List Bullet"), check, size=10)
        if review.remaining_issues:
            p = doc.add_paragraph()
            _styled_run(p, "Open points the reviewer flagged - please check before sending:",
                        bold=True, color=PRIMARY)
            for issue in review.remaining_issues:
                _rich_text(doc.add_paragraph(style="List Bullet"), issue)
        note = doc.add_paragraph()
        _styled_run(note, f"Scoring: {review.scoring} This is an automated review, not a "
                          "guarantee; please verify key figures before relying on them.",
                    size=9, italic=True, color=MUTED)
    if meta.providers:
        p = doc.add_paragraph()
        _styled_run(p, "Language models used by each agent:", size=9, bold=True, color=MUTED)
        for line in meta.providers:
            _styled_run(doc.add_paragraph(style="List Bullet"), line, size=9, color=MUTED)


# --------------------------------------------------------------------------- #
# Content blocks
# --------------------------------------------------------------------------- #
def _render_blocks(doc: DocxDocument, content: SectionContent) -> None:
    for block in content.blocks:
        if isinstance(block, ParagraphBlock):
            _rich_text(doc.add_paragraph(), block.text)
        elif isinstance(block, BulletsBlock):
            for item in block.items:
                _rich_text(doc.add_paragraph(style="List Bullet"), item)
        else:
            _render_table(doc, block)


def _render_table(doc: DocxDocument, block: TableBlock, widths: list | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(block.headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _table_borders(table)

    header = table.rows[0]
    _repeat_header(header)
    for cell, text in zip(header.cells, block.headers):
        _cell_borders(cell, PRIMARY_HEX)  # no light grid lines inside the dark header
        _shade(cell, PRIMARY_HEX)
        _styled_run(cell.paragraphs[0], text, size=10, bold=True, color=WHITE)

    for index, row in enumerate(block.rows):
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            if index % 2 == 1:
                _shade(cell, ZEBRA_HEX)
            _rich_text(cell.paragraphs[0], text, size=10)
    _set_col_widths(table, widths or _auto_widths(block))
    _keep_together(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _keep_together(table, max_rows: int = 15) -> None:
    """Stop a table splitting badly across pages (e.g. a header row alone at the bottom).

    Rows never split internally. Small tables stay on one page; larger ones at
    least keep the header with the first rows.
    """
    rows = table.rows
    for row in rows:
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.insert(0, cant_split)  # cantSplit precedes tblHeader in the schema
    together = len(rows) - 1 if len(rows) <= max_rows else 3
    for row in rows[:together]:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = True


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


def _rich_text(paragraph: Paragraph, text: str, size: int | None = None) -> None:
    """Write text, turning **bold** / *italic* into real formatting and dropping stray markdown."""
    text = re.sub(r"^\s*(#+|[-*•])\s+", "", text.strip())
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            _styled_run(paragraph, part[2:-2], size=size, bold=True)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            _styled_run(paragraph, part[1:-1], size=size, italic=True)
        else:
            _styled_run(paragraph, part, size=size)


def _auto_widths(block: TableBlock) -> list:
    """Give wordy columns more room: width proportional to typical text length."""
    columns = len(block.headers)
    weights = []
    for col in range(columns):
        lengths = [len(block.headers[col])] + [len(row[col]) for row in block.rows]
        typical = sum(sorted(lengths)[len(lengths) // 2:]) / max(1, len(lengths) - len(lengths) // 2)
        weights.append(min(max(typical, 6), 60))
    minimum = 1.6
    spare = CONTENT_WIDTH_CM - minimum * columns
    total = sum(weights)
    return [Cm(minimum + spare * w / total) for w in weights]


_TYPOGRAPHY = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-",   # hyphen variants old Word renders oddly
    " ": " ", " ": " ", " ": " ",   # non-breaking / narrow spaces
    "​": "", "­": "",                    # zero-width space, soft hyphen
})


def _normalize(text: str) -> str:
    """LLMs love typographic hyphens and spaces; Word 2007-era fonts render them badly."""
    return text.translate(_TYPOGRAPHY)


def _styled_run(paragraph: Paragraph, text: str, *, size: int | None = None, bold: bool = False,
                italic: bool = False, color: RGBColor | None = None):
    run = paragraph.add_run(_normalize(text))
    if size:
        run.font.size = Pt(size)
    run.font.bold = bold or None
    run.font.italic = italic or None
    if color is not None:
        run.font.color.rgb = color
    return run


# --------------------------------------------------------------------------- #
# Low-level OOXML helpers (python-docx has no API for these).
# Elements are inserted in schema order so Word never flags the file as corrupt.
# --------------------------------------------------------------------------- #
def _shade(cell: _Cell, fill_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tc_pr.insert_element_before(
        shd, "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign",
        "w:hideMark", "w:headers", "w:cellIns", "w:cellDel", "w:cellMerge", "w:tcPrChange",
    )


def _cell_borders(cell: _Cell, color_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color_hex)
        borders.append(el)
    tc_pr.insert_element_before(
        borders, "w:shd", "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText",
        "w:vAlign", "w:hideMark", "w:headers", "w:cellIns", "w:cellDel", "w:cellMerge",
        "w:tcPrChange",
    )


def _table_borders(table, color: str = BORDER_HEX) -> None:
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    tbl_pr = table._tbl.tblPr
    existing = tbl_pr.find(qn("w:tblBorders"))
    if existing is not None:
        tbl_pr.remove(existing)
    tbl_pr.insert_element_before(
        borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
        "w:tblCaption", "w:tblDescription", "w:tblPrChange",
    )


def _repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def _bottom_border(paragraph: Paragraph, color_hex: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for key, value in (("w:val", "single"), ("w:sz", "12"), ("w:space", "1"), ("w:color", color_hex)):
        bottom.set(qn(key), value)
    border.append(bottom)
    p_pr.insert_element_before(
        border, "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap",
        "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi",
        "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing",
        "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
        "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
        "w:pPrChange",
    )


def _add_field(paragraph: Paragraph, instruction: str) -> None:
    """Insert a live Word field such as PAGE or NUMPAGES."""
    def run_with(element) -> None:
        run = paragraph.add_run()
        run.font.size = Pt(9)
        run.font.color.rgb = MUTED
        run._r.append(element)

    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")

    run_with(begin)
    run_with(instr)
    run_with(separate)
    _styled_run(paragraph, "1", size=9, color=MUTED)
    run_with(end)


def _set_col_widths(table, widths: list) -> None:
    table.autofit = False
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
