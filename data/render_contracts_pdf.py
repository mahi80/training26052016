"""Render the six synthetic contracts (data/contracts/*.md) to realistic PDFs.

WHY THIS EXISTS
---------------
In production, contracts do NOT arrive as tidy markdown. They arrive as PDFs —
signed, paginated, client-facing documents — and the very first step of any
document-intelligence pipeline is *extraction*: pulling structured text
(headings, clauses, lists) out of the PDF with a tool such as PyMuPDF or
pdfminer BEFORE any tree indexing happens. The markdown files in
``data/contracts/`` represent the *output* of that extraction step: the clean,
heading-structured corpus that the PageIndex tree builder
(``src/pageindex/tree_builder.py``) consumes.

This module closes the realism gap by generating the *upstream* artifact: a
properly typeset PDF rendition of each contract (cover page, numbered section
hierarchy, justified clause text, page footers). Trainees can use these PDFs
to (a) demo the full real-world flow — PDF -> extracted text -> PageIndex
tree -> reasoning RAG — and (b) practice extraction with ``fitz``/``pdfminer``
and compare their output against the known-good markdown.

DESIGN NOTES (teaching points)
------------------------------
* **reportlab platypus** ("Page Layout And Typography Using Scripts") builds a
  PDF from a *story* — a list of Flowables (Paragraph, Spacer, Table...) — and
  lets a callback decorate every page (footers, page numbers).
* **Determinism**: ``invariant=1`` makes reportlab use a fixed creation date
  and a content-derived document ID, so re-running the script produces
  byte-identical PDFs — essential for reproducible training data and tests.
* **Markdown subset**: the parser below handles exactly the constructs found
  in the six source files (H1 title, a bold metadata line, ##/### numbered
  headings, ordered/bullet list items, ``**bold**`` inline, horizontal rules).
  Anything unrecognized falls back to a plain body paragraph — the renderer
  must never crash on an unexpected line.

Run from the project root:  ``python data/render_contracts_pdf.py``
Outputs:                    ``data/contracts_pdf/<same-stem>.pdf`` (idempotent)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --- paths (always relative to this file; never hardcode absolutes) ---------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONTRACTS_DIR: Path = PROJECT_ROOT / "data" / "contracts"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "contracts_pdf"

# --- page geometry & fixed strings ------------------------------------------
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 22 * mm
FRAME_WIDTH = PAGE_WIDTH - 2 * MARGIN

FOOTER_TEXT = "Zero-to-Hero Training — synthetic sample contract"
SAMPLE_NOTICE = "SAMPLE — synthetic training document, not legal advice"

INK = colors.HexColor("#1a1a2e")        # body text
ACCENT = colors.HexColor("#16324f")     # headings / rules
MUTED = colors.HexColor("#6b6b7b")      # footer / secondary text
NOTICE_BG = colors.HexColor("#fdf6e3")  # cover notice background

# =============================================================================
# 1. Markdown parsing — exactly the subset used by the contract corpus
# =============================================================================


@dataclass
class Block:
    """One parsed markdown block: heading, paragraph, list, rule or metadata."""

    kind: str                       # h1 h2 h3 h4 p ol ul hr meta
    text: str = ""
    items: list[tuple[str, str]] = field(default_factory=list)  # (marker, text)


@dataclass
class ContractMeta:
    """Cover-page facts parsed from the H1 title and the bold metadata line."""

    doc_type: str        # e.g. "Master Service Agreement"
    party_a: str         # e.g. "GlobalTrade Logistics Inc."
    party_b: str         # e.g. "SwiftShip Express, Inc."
    agreement_no: str    # e.g. "SS-MSA-2024-117"
    effective_date: str  # e.g. "January 15, 2024"


_OL_RE = re.compile(r"^(\d+)[.)]\s+(.+)$")
_UL_RE = re.compile(r"^[-*+]\s+(.+)$")
_HR_RE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})\s*$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_SIG_RE = re.compile(r"signature|execution|in witness", re.IGNORECASE)


def md_inline(text: str) -> str:
    """Escape XML specials, then map ``**bold**`` to reportlab ``<b>`` markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _BOLD_RE.sub(r"<b>\1</b>", text)


def parse_markdown(text: str) -> list[Block]:
    """Split contract markdown into typed blocks (never raises on odd input)."""
    blocks: list[Block] = []
    para_buf: list[str] = []

    def flush() -> None:
        if para_buf:
            blocks.append(Block("p", " ".join(para_buf)))
            para_buf.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            blocks.append(Block(f"h{min(level, 4)}", line[level:].strip()))
            continue
        if _HR_RE.match(line):
            flush()
            blocks.append(Block("hr"))
            continue
        if "Agreement No" in line and "Effective Date:" in line and "|" in line:
            flush()
            blocks.append(Block("meta", line))  # consumed by the cover page
            continue
        if m := _OL_RE.match(line):
            flush()
            item = (f"{m.group(1)}.", m.group(2))
            if blocks and blocks[-1].kind == "ol":
                blocks[-1].items.append(item)
            else:
                blocks.append(Block("ol", items=[item]))
            continue
        if m := _UL_RE.match(line):
            flush()
            item = ("•", m.group(1))
            if blocks and blocks[-1].kind == "ul":
                blocks[-1].items.append(item)
            else:
                blocks.append(Block("ul", items=[item]))
            continue
        para_buf.append(line)  # fallback: anything unknown is body text
    flush()
    return blocks


def extract_meta(blocks: list[Block]) -> ContractMeta:
    """Pull cover-page facts out of the H1 + metadata blocks (regex, tolerant)."""
    title = next((b.text for b in blocks if b.kind == "h1"), "Contract")
    meta_line = next((b.text for b in blocks if b.kind == "meta"), "")

    doc_type, sep, parties = title.partition(" — ")
    if not sep:
        doc_type, parties = title, ""

    m = re.search(r"Between \*\*(.+?)\*\*.*?and \*\*(.+?)\*\*", meta_line)
    if m:
        party_a, party_b = m.group(1), m.group(2)
    else:  # fall back to the "A & B" tail of the H1 title
        bits = [p.strip() for p in parties.split("&")]
        party_a = bits[0] if bits and bits[0] else "Party A"
        party_b = bits[-1] if len(bits) > 1 else "Party B"

    no = re.search(r"Agreement No\.?\s*([A-Z0-9][A-Z0-9-]*)", meta_line)
    date = re.search(r"Effective Date:\s*([^|*]+)", meta_line)
    return ContractMeta(
        doc_type=doc_type.strip(),
        party_a=party_a.strip(),
        party_b=party_b.strip(),
        agreement_no=no.group(1) if no else "N/A",
        effective_date=date.group(1).strip() if date else "N/A",
    )


# =============================================================================
# 2. Styles & page furniture
# =============================================================================


def make_styles() -> dict[str, ParagraphStyle]:
    """Named paragraph styles for the cover and the clause hierarchy."""
    base = dict(fontName="Helvetica", textColor=INK)
    return {
        "cover_type": ParagraphStyle("cover_type", fontName="Helvetica-Bold",
                                     fontSize=22, leading=27, alignment=TA_CENTER,
                                     textColor=ACCENT),
        "cover_party": ParagraphStyle("cover_party", fontName="Helvetica-Bold",
                                      fontSize=14, leading=18, alignment=TA_CENTER,
                                      textColor=INK),
        "cover_small": ParagraphStyle("cover_small", fontSize=10.5, leading=15,
                                      alignment=TA_CENTER, textColor=MUTED, **{
                                          k: v for k, v in base.items()
                                          if k != "textColor"}),
        "cover_meta": ParagraphStyle("cover_meta", fontSize=11, leading=16,
                                     alignment=TA_CENTER, **base),
        "notice": ParagraphStyle("notice", fontSize=9.5, leading=13,
                                 alignment=TA_CENTER, **base),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13,
                             leading=16, spaceBefore=14, spaceAfter=6,
                             textColor=ACCENT, keepWithNext=1),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11,
                             leading=14, spaceBefore=10, spaceAfter=4,
                             textColor=ACCENT, keepWithNext=1),
        "h4": ParagraphStyle("h4", fontName="Helvetica-BoldOblique", fontSize=10,
                             leading=13, spaceBefore=8, spaceAfter=3,
                             textColor=ACCENT, keepWithNext=1),
        "body": ParagraphStyle("body", fontSize=9.5, leading=13.5,
                               alignment=TA_JUSTIFY, spaceAfter=6, **base),
        "list": ParagraphStyle("list", fontSize=9.5, leading=13.5,
                               alignment=TA_JUSTIFY, spaceAfter=3,
                               leftIndent=18, bulletIndent=2, **base),
        "sig": ParagraphStyle("sig", fontSize=9.5, leading=15, **base),
    }


def make_page_decorator(meta: ContractMeta):
    """Closure drawing the running footer (and a header on pages >= 2)."""

    def decorate(canvas, doc) -> None:
        canvas.saveState()
        if doc.page > 1:  # discreet header with the agreement number
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 12 * mm,
                                   f"Agreement No. {meta.agreement_no}")
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, 16 * mm, PAGE_WIDTH - MARGIN, 16 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 11.5 * mm, FOOTER_TEXT)
        canvas.drawRightString(PAGE_WIDTH - MARGIN, 11.5 * mm, f"Page {doc.page}")
        canvas.restoreState()

    return decorate


# =============================================================================
# 3. Story assembly (cover -> clauses -> optional signature table)
# =============================================================================


def build_cover(meta: ContractMeta, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """Title block: doc type, parties, agreement no., effective date, notice."""
    rule = HRFlowable(width="70%", thickness=1, color=ACCENT, hAlign="CENTER")
    notice = Table(
        [[Paragraph(
            f"<b>{md_inline(SAMPLE_NOTICE)}</b><br/>"
            "Generated for the Zero-to-Hero LangGraph training. All parties, "
            "figures, addresses and terms are fictitious.",
            styles["notice"])]],
        colWidths=[FRAME_WIDTH * 0.78], hAlign="CENTER",
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), NOTICE_BG),
            ("BOX", (0, 0), (-1, -1), 0.8, ACCENT),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]),
    )
    return [
        Spacer(1, 38 * mm),
        rule,
        Spacer(1, 8 * mm),
        Paragraph(md_inline(meta.doc_type), styles["cover_type"]),
        Spacer(1, 10 * mm),
        Paragraph("between", styles["cover_small"]),
        Spacer(1, 3 * mm),
        Paragraph(md_inline(meta.party_a), styles["cover_party"]),
        Spacer(1, 3 * mm),
        Paragraph("and", styles["cover_small"]),
        Spacer(1, 3 * mm),
        Paragraph(md_inline(meta.party_b), styles["cover_party"]),
        Spacer(1, 10 * mm),
        rule,
        Spacer(1, 8 * mm),
        Paragraph(f"Agreement No. {md_inline(meta.agreement_no)}", styles["cover_meta"]),
        Paragraph(f"Effective Date: {md_inline(meta.effective_date)}", styles["cover_meta"]),
        Spacer(1, 16 * mm),
        notice,
        PageBreak(),
    ]


def build_signature_table(meta: ContractMeta,
                          styles: dict[str, ParagraphStyle]) -> Table:
    """Two-column execution block, one column per party (only rendered when
    the source markdown actually contains a signature/execution section)."""
    cells = []
    for party in (meta.party_a, meta.party_b):
        cells.append(Paragraph(
            f"<b>For and on behalf of</b><br/>{md_inline(party)}<br/><br/>"
            "Signature: ______________________<br/>"
            "Name: __________________________<br/>"
            "Title: ___________________________<br/>"
            "Date: ___________________________",
            styles["sig"]))
    return Table([cells], colWidths=[FRAME_WIDTH / 2] * 2, hAlign="LEFT",
                 style=TableStyle([
                     ("VALIGN", (0, 0), (-1, -1), "TOP"),
                     ("LINEABOVE", (0, 0), (-1, 0), 0.6, ACCENT),
                     ("TOPPADDING", (0, 0), (-1, -1), 10),
                 ]))


def build_story(blocks: list[Block], meta: ContractMeta,
                styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """Convert parsed blocks into the full platypus story."""
    story: list[Flowable] = build_cover(meta, styles)
    has_signature_section = False
    for block in blocks:
        if block.kind in ("h1", "meta"):
            continue  # consumed by the cover page
        if block.kind in ("h2", "h3", "h4"):
            if _SIG_RE.search(block.text):
                has_signature_section = True
            story.append(Paragraph(md_inline(block.text), styles[block.kind]))
        elif block.kind in ("ol", "ul"):
            for marker, item_text in block.items:
                story.append(Paragraph(md_inline(item_text), styles["list"],
                                       bulletText=marker))
            story.append(Spacer(1, 3))
        elif block.kind == "hr":
            story.append(HRFlowable(width="100%", thickness=0.6, color=ACCENT,
                                    spaceBefore=6, spaceAfter=6))
        else:  # "p" and any unknown construct fall back to body text
            if block.text.upper().startswith("IN WITNESS"):
                has_signature_section = True
            story.append(Paragraph(md_inline(block.text), styles["body"]))
    if has_signature_section:
        story.append(Spacer(1, 8 * mm))
        story.append(build_signature_table(meta, styles))
    return story


# =============================================================================
# 4. Rendering & CLI entry point
# =============================================================================


def render_contract(md_path: Path, pdf_path: Path) -> int:
    """Render one markdown contract to PDF; return the number of pages."""
    blocks = parse_markdown(md_path.read_text(encoding="utf-8"))
    meta = extract_meta(blocks)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=20 * mm, bottomMargin=24 * mm,
        title=f"{meta.doc_type} — {meta.party_a} & {meta.party_b}",
        author=meta.party_a, subject=SAMPLE_NOTICE,
        creator="render_contracts_pdf.py (Zero-to-Hero Training)",
        invariant=1,  # fixed timestamps + content-derived ID => byte-stable
    )
    decorate = make_page_decorator(meta)
    doc.build(build_story(blocks, meta, styles),
              onFirstPage=decorate, onLaterPages=decorate)
    return doc.page


def main() -> None:
    """Convert every contract in data/contracts/ to data/contracts_pdf/."""
    md_files = sorted(CONTRACTS_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(
            f"No markdown contracts found in {CONTRACTS_DIR} - run "
            "data/generate_contracts.py first.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for md_path in md_files:
        pdf_path = OUTPUT_DIR / f"{md_path.stem}.pdf"
        pages = render_contract(md_path, pdf_path)
        size_kb = pdf_path.stat().st_size / 1024
        print(f"[ok] {md_path.name:<36} -> {pdf_path.parent.name}/"
              f"{pdf_path.name:<36} {pages} pages, {size_kb:6.1f} KB")
    print(f"Rendered {len(md_files)} contracts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
