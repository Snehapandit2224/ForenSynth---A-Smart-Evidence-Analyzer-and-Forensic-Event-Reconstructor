#!/usr/bin/env python3
"""
ForenSynth — Explainability Report v2
======================================
Generates a human+technical dual-layer PDF from all pipeline JSON outputs.

STRUCTURE:
  Page 1   : Cover — case snapshot, confidence summary, table of contents
  Page 2   : How to Read This Report (guide for non-technical readers)
  Pages 3-4: Section 2 — Who Was Involved (Entity Resolution)
  Pages 5-8: Section 3 — What Happened (Timeline of Events)
  Pages 9-11: Section 4 — Evidence Gaps (Critique rounds C1→C3)
  Pages 12-13: Section 5 — System Decisions (Showrunner rounds)
  Page 14  : Section 6 — What Still Needs Human Review
  Page 15  : Section 7 — Final Verdict

DESIGN PRINCIPLE — every section has two layers:
  Blue box  : plain English — for investigators, legal teams, non-technical readers
  Grey box  : technical detail — for forensic analysts

Usage:
    python explainability_report_v2.py
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF, XPos, YPos

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Font discovery — we look for DejaVuSans (proportional, more readable)
# then fall back to DejaVuSansMono (which we know is installed on Linux).
# Windows ships neither by default, so its Fonts folder is searched first
# for any usable TTF before we give up and use fpdf's built-in core font.
FONT_SEARCH_DIRS = [
    "C:/Windows/Fonts/",
    "/usr/share/fonts/truetype/dejavu/",
    "/usr/share/fonts/dejavu/",
    "/usr/local/share/fonts/",
    os.path.expanduser("~/fonts/"),
    ".",
]

# Confidence thresholds
HIGH_CONF = 0.75
LOW_CONF  = 0.60

# ── Colour palette ─────────────────────────────────────────────────────────────
C_PRIMARY  = (41,  98,  162)   # dark blue  — section headers
C_ACCENT   = (0,  120,  200)   # medium blue — sub-headers
C_GREEN    = (34,  139,  34)   # HIGH confidence
C_ORANGE   = (204, 120,   0)   # MEDIUM confidence / warnings
C_RED      = (178,  34,  34)   # LOW confidence / critical
C_PLAIN_BG = (235, 245, 255)   # light blue  — plain-English boxes
C_TECH_BG  = (245, 245, 245)   # light gray  — technical boxes
C_WARN_BG  = (255, 248, 220)   # light yellow — warning boxes
C_TEXT     = (30,  30,  30)    # near-black body text
C_MUTED    = (110, 110, 110)   # captions / secondary text
C_WHITE    = (255, 255, 255)
C_BORDER   = (190, 190, 190)   # box borders

PAGE_W    = 210    # A4 mm
PAGE_H    = 297
MARGIN    = 15
CONTENT_W = PAGE_W - 2 * MARGIN

# ──────────────────────────────────────────────────────────────────────────────
# Font helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_font(names: List[str]) -> Optional[str]:
    for d in FONT_SEARCH_DIRS:
        dp = Path(d)
        if not dp.is_dir():
            continue
        for name in names:
            p = dp / name
            if p.exists():
                return str(p)
    return None


def _find_any_ttf() -> Optional[str]:
    """Last resort before falling back to fpdf's built-in core font."""
    for d in FONT_SEARCH_DIRS:
        dp = Path(d)
        if not dp.is_dir():
            continue
        matches = sorted(dp.glob("*.ttf"))
        if matches:
            return str(matches[0])
    return None

F_REG    = _find_font(["DejaVuSans.ttf",          "DejaVuSansMono.ttf"])
F_BOLD   = _find_font(["DejaVuSans-Bold.ttf",      "DejaVuSansMono-Bold.ttf"])
F_ITALIC = _find_font(["DejaVuSans-Oblique.ttf",   "DejaVuSansMono.ttf"])

if not F_REG:
    _any_ttf  = _find_any_ttf()
    F_REG     = F_REG    or _any_ttf
    F_BOLD    = F_BOLD   or _any_ttf
    F_ITALIC  = F_ITALIC or _any_ttf

# "sans" is registered via add_font() only when we actually found a TTF file
# (see ExplainabilityPDF._setup_fonts). With nothing found anywhere, fall
# back to fpdf's built-in core font so the report still renders.
FONT_NAME = "sans" if F_REG else "helvetica"

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def safe(v: Any, maxlen: int = 0) -> str:
    s = str(v) if v is not None else ""
    if maxlen and len(s) > maxlen:
        s = s[: maxlen - 1] + "…"
    return s


def conf_label(c: float) -> Tuple[str, Tuple[int, int, int]]:
    """Return (label, colour) for a confidence score."""
    if c >= HIGH_CONF:
        return "HIGH",   C_GREEN
    elif c >= LOW_CONF:
        return "MEDIUM", C_ORANGE
    else:
        return "LOW",    C_RED


def load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def action_to_plain(tags: List[str]) -> str:
    """Convert machine action tags to plain English phrases."""
    mapping = {
        "APPROACH":  "approaching the ATM",
        "ENTER":     "entering the ATM booth",
        "WITHDRAW":  "using the ATM card reader",
        "TAMPER":    "tampering with the card reader",
        "FLEE":      "fleeing the scene",
        "EXIT":      "leaving the ATM area",
        "OBSERVE":   "observing the scene",
        "LOITER":    "loitering near the ATM",
    }
    if not tags:
        return "present at the location"
    return " and ".join(mapping.get(t, t.replace("_", " ").lower()) for t in tags)


def time_str(ts: str) -> str:
    """Extract HH:MM:SS from an ISO timestamp string."""
    return ts[11:19] if len(ts) >= 19 else ts


# ──────────────────────────────────────────────────────────────────────────────
# PDF class — layout primitives
# ──────────────────────────────────────────────────────────────────────────────

class ExplainabilityPDF(FPDF):

    def __init__(self, case_id: str):
        super().__init__("P", "mm", "A4")
        self.case_id = case_id
        self._setup_fonts()
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(MARGIN, MARGIN, MARGIN)

    def _setup_fonts(self):
        if F_REG:
            self.add_font(FONT_NAME, "",  F_REG,    uni=True)
        if F_BOLD:
            self.add_font(FONT_NAME, "B", F_BOLD,   uni=True)
        if F_ITALIC:
            self.add_font(FONT_NAME, "I", F_ITALIC, uni=True)

    # ── Running header / footer ───────────────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(FONT_NAME, "B", 7)
        self.set_text_color(*C_MUTED)
        self.cell(0, 5, f"ForenSynth Explainability Report  ·  {self.case_id}  ·  CONFIDENTIAL", align="L")
        self.ln(1)
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.25)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)
        self.set_text_color(*C_TEXT)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.25)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(1)
        self.set_font(FONT_NAME, "", 7)
        self.set_text_color(*C_MUTED)
        self.cell(0, 5, f"Page {self.page_no()}   ·   CONFIDENTIAL — FORENSIC USE ONLY", align="C")

    # ── Typography helpers ────────────────────────────────────────────────────

    def h1(self, text: str):
        self.set_font(FONT_NAME, "B", 15)
        self.set_text_color(*C_PRIMARY)
        self.ln(3)
        self.multi_cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*C_TEXT)
        self.set_draw_color(*C_PRIMARY)
        self.set_line_width(0.5)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    def h2(self, text: str):
        self.set_font(FONT_NAME, "B", 10)
        self.set_text_color(*C_ACCENT)
        self.ln(3)
        self.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*C_TEXT)
        self.ln(1)

    def body(self, text: str, size: int = 9):
        self.set_font(FONT_NAME, "", size)
        self.set_text_color(*C_TEXT)
        self.multi_cell(0, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def small(self, text: str):
        self.set_font(FONT_NAME, "I", 7.5)
        self.set_text_color(*C_MUTED)
        self.multi_cell(0, 4, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*C_TEXT)
        self.ln(1)

    def divider(self):
        self.ln(2)
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.2)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    # ── Coloured boxes ────────────────────────────────────────────────────────

    def plain_box(self, title: str, text: str):
        """
        Blue 'Plain English' box.
        Dark blue header strip + light blue body — for non-technical readers.
        """
        self.set_font(FONT_NAME, "", 9)
        # Estimate height: header (6mm) + text lines + padding
        lines = self.multi_cell(CONTENT_W - 8, 5, text, split_only=True)
        box_h = 6 + len(lines) * 5 + 7

        y = self.get_y()
        if y + box_h > PAGE_H - 25:
            self.add_page()
            y = self.get_y()

        # Body background
        self.set_fill_color(*C_PLAIN_BG)
        self.set_draw_color(*C_PRIMARY)
        self.set_line_width(0.4)
        self.rect(MARGIN, y, CONTENT_W, box_h, "FD")

        # Header strip
        self.set_fill_color(*C_PRIMARY)
        self.rect(MARGIN, y, CONTENT_W, 6, "F")
        self.set_xy(MARGIN + 3, y + 1)
        self.set_font(FONT_NAME, "B", 7)
        self.set_text_color(*C_WHITE)
        self.cell(CONTENT_W - 6, 4, f"  PLAIN ENGLISH  —  {title.upper()}")

        # Body text
        self.set_xy(MARGIN + 4, y + 9)
        self.set_font(FONT_NAME, "", 9)
        self.set_text_color(*C_TEXT)
        self.multi_cell(CONTENT_W - 8, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

    def tech_box(self, title: str, lines: List[str]):
        """
        Grey 'Technical Detail' box — for forensic analysts.
        """
        self.set_font(FONT_NAME, "", 7.5)
        total = sum(
            len(self.multi_cell(CONTENT_W - 10, 4.5, ln, split_only=True))
            for ln in lines
        )
        box_h = 7 + total * 4.5 + 5

        y = self.get_y()
        if y + box_h > PAGE_H - 25:
            self.add_page()
            y = self.get_y()

        self.set_fill_color(*C_TECH_BG)
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.2)
        self.rect(MARGIN, y, CONTENT_W, box_h, "FD")

        self.set_xy(MARGIN + 3, y + 2)
        self.set_font(FONT_NAME, "B", 7)
        self.set_text_color(*C_MUTED)
        self.cell(CONTENT_W - 6, 4, f"TECHNICAL DETAIL  —  {title.upper()}")

        self.set_xy(MARGIN + 5, y + 7)
        self.set_font(FONT_NAME, "", 7.5)
        self.set_text_color(*C_TEXT)
        for line in lines:
            self.multi_cell(CONTENT_W - 10, 4.5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

    def warn_box(self, text: str):
        """
        Yellow warning box — items requiring human attention.
        """
        self.set_font(FONT_NAME, "", 8.5)
        lines = self.multi_cell(CONTENT_W - 8, 5, text, split_only=True)
        box_h = len(lines) * 5 + 9

        y = self.get_y()
        if y + box_h > PAGE_H - 25:
            self.add_page()
            y = self.get_y()

        self.set_fill_color(*C_WARN_BG)
        self.set_draw_color(*C_ORANGE)
        self.set_line_width(0.5)
        self.rect(MARGIN, y, CONTENT_W, box_h, "FD")

        self.set_xy(MARGIN + 4, y + 4)
        self.set_font(FONT_NAME, "", 8.5)
        self.set_text_color(*C_TEXT)
        self.multi_cell(CONTENT_W - 8, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def conf_badge(self, confidence: float, x: float, y: float,
                   w: float = 30, h: float = 7):
        """Coloured pill badge showing confidence label and percentage."""
        label, color = conf_label(confidence)
        self.set_fill_color(*color)
        self.set_draw_color(*color)
        self.set_line_width(0.1)
        # Simulated rounded rect via rect (fpdf2 rounded_rect requires v2.7+)
        self.rect(x, y, w, h, "F")
        self.set_xy(x, y + 1.2)
        self.set_font(FONT_NAME, "B", 7)
        self.set_text_color(*C_WHITE)
        self.cell(w, h - 2, f"{label}  {confidence:.0%}", align="C")
        self.set_text_color(*C_TEXT)


# ──────────────────────────────────────────────────────────────────────────────
# Section 0 — Cover page
# ──────────────────────────────────────────────────────────────────────────────

def build_cover(pdf: ExplainabilityPDF, case_id: str, er: dict, tl: dict):
    pdf.add_page()

    # ── Blue header banner ────────────────────────────────────────────────────
    pdf.set_fill_color(*C_PRIMARY)
    pdf.rect(0, 0, PAGE_W, 58, "F")

    pdf.set_xy(MARGIN, 10)
    pdf.set_font(FONT_NAME, "B", 20)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(0, 10, "Forensic Evidence Analysis Report", align="C")

    pdf.set_xy(MARGIN, 24)
    pdf.set_font(FONT_NAME, "B", 12)
    pdf.cell(0, 8, "Explainability & Decision Audit", align="C")

    pdf.set_xy(MARGIN, 38)
    pdf.set_font(FONT_NAME, "", 8.5)
    pdf.cell(0, 5,
        f"Case ID: {case_id}   |   Generated by ForenSynth AI Pipeline   |   CONFIDENTIAL",
        align="C")
    pdf.set_text_color(*C_TEXT)

    # ── What is this report? ──────────────────────────────────────────────────
    pdf.set_y(68)
    pdf.set_font(FONT_NAME, "B", 9.5)
    pdf.set_text_color(*C_PRIMARY)
    pdf.cell(0, 6, "WHAT IS THIS REPORT?", align="C")
    pdf.ln(3)
    pdf.set_font(FONT_NAME, "", 9)
    pdf.set_text_color(*C_TEXT)
    pdf.multi_cell(
        0, 5.5,
        "This report explains every decision made by the ForenSynth AI system when analysing "
        "the evidence in this case. It is written in two layers: a plain English section "
        "(for investigators, legal teams, and non-technical readers) and a technical section "
        "(for forensic analysts). Every confidence score, every flag, and every gap is explained "
        "in both layers so that any reader can understand and verify the system's conclusions.",
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    pdf.ln(5)
    pdf.set_draw_color(*C_BORDER)
    pdf.set_line_width(0.3)
    pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
    pdf.ln(5)

    # ── Summary stats strip ───────────────────────────────────────────────────
    events      = tl.get("events", []) if tl else []
    n_entities  = er.get("entity_count", 0) if er else 0
    n_events    = len(events)
    er_status   = er.get("output_classification", "?") if er else "?"
    tl_status   = tl.get("output_classification", "?") if tl else "?"
    avg_conf    = sum(e.get("confidence", 0) for e in events) / max(n_events, 1)

    stats = [
        ("People Identified",    str(n_entities),        C_PRIMARY),
        ("Timeline Events",      str(n_events),           C_PRIMARY),
        ("Avg. Confidence",      f"{avg_conf:.0%}",       C_GREEN if avg_conf >= HIGH_CONF else C_ORANGE),
        ("Overall Result",       tl_status,               C_GREEN if tl_status == "CLEAR" else C_RED),
    ]
    col_w = CONTENT_W / 4
    x0    = MARGIN
    y0    = pdf.get_y()

    for label, val, vcolor in stats:
        pdf.set_fill_color(*C_PLAIN_BG)
        pdf.set_draw_color(*C_PRIMARY)
        pdf.set_line_width(0.3)
        pdf.rect(x0, y0, col_w - 3, 20, "FD")

        pdf.set_xy(x0, y0 + 2)
        pdf.set_font(FONT_NAME, "B", 14)
        pdf.set_text_color(*vcolor)
        pdf.cell(col_w - 3, 9, val, align="C")

        pdf.set_xy(x0, y0 + 12)
        pdf.set_font(FONT_NAME, "", 6.5)
        pdf.set_text_color(*C_MUTED)
        pdf.cell(col_w - 3, 5, label, align="C")

        x0 += col_w

    pdf.set_text_color(*C_TEXT)
    pdf.set_y(y0 + 25)

    pdf.ln(4)
    pdf.set_draw_color(*C_BORDER)
    pdf.set_line_width(0.3)
    pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
    pdf.ln(5)

    # ── Table of contents ─────────────────────────────────────────────────────
    pdf.set_font(FONT_NAME, "B", 9.5)
    pdf.set_text_color(*C_PRIMARY)
    pdf.cell(0, 6, "CONTENTS")
    pdf.ln(6)

    contents = [
        ("1", "How to Read This Report"),
        ("2", "Who Was Involved — Entity Resolution"),
        ("3", "What Happened — Timeline of Events"),
        ("4", "Evidence Gaps — What the System Found Missing"),
        ("5", "System Decisions — How the AI Revised Its Analysis"),
        ("6", "What Still Needs Human Review"),
        ("7", "Final Verdict"),
        ("8", "Case Narrative — The Full Picture in Plain English  ★ Start here"),
    ]
    for num, title in contents:
        pdf.set_font(FONT_NAME, "B", 9)
        pdf.set_text_color(*C_ACCENT)
        pdf.cell(12, 6.5, f"  {num}.")
        pdf.set_font(FONT_NAME, "", 9)
        pdf.set_text_color(*C_TEXT)
        pdf.cell(0, 6.5, title)
        pdf.ln()


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — How to Read This Report
# ──────────────────────────────────────────────────────────────────────────────

def build_guide(pdf: ExplainabilityPDF):
    pdf.add_page()
    pdf.h1("Section 1 — How to Read This Report")

    pdf.plain_box(
        "For non-technical readers",
        "This report was produced by an AI system called ForenSynth. It analysed multiple "
        "pieces of evidence — CCTV footage, audio recordings, and written reports — and "
        "attempted to reconstruct what happened and who was involved.\n\n"
        "You do not need to understand any technical details. Each section of this report "
        "starts with a blue 'Plain English' box (like this one) that explains the findings "
        "in straightforward language. Grey boxes that follow contain the technical "
        "information for analysts — you can skip these.\n\n"
        "IMPORTANT: This report is a tool to help investigators, not a final verdict. "
        "Every finding marked with a warning symbol (⚠) still requires a human investigator "
        "to confirm before any action is taken.",
    )

    pdf.h2("Understanding Confidence Levels")
    pdf.body(
        "Throughout this report, findings are rated HIGH, MEDIUM, or LOW confidence. "
        "This rating tells you how certain the AI system is about each conclusion, "
        "based on how much supporting evidence was found and how consistent it was."
    )
    pdf.ln(2)

    badge_info = [
        ("HIGH  (75% or above)", C_GREEN,
         "Strong evidence. Multiple sources agree. A reliable starting point for investigation."),
        ("MEDIUM  (60 – 75%)", C_ORANGE,
         "Reasonable evidence but some uncertainty. Treat as a lead, not a confirmed fact."),
        ("LOW  (below 60%)", C_RED,
         "Weak or conflicting evidence. Requires human investigation before any action is taken."),
    ]
    for label, color, desc in badge_info:
        y0 = pdf.get_y()
        pdf.set_fill_color(*color)
        pdf.rect(MARGIN, y0, 40, 8, "F")
        pdf.set_xy(MARGIN, y0 + 1.5)
        pdf.set_font(FONT_NAME, "B", 7)
        pdf.set_text_color(*C_WHITE)
        pdf.cell(40, 5, label, align="C")

        pdf.set_xy(MARGIN + 44, y0 + 1.5)
        pdf.set_font(FONT_NAME, "", 8.5)
        pdf.set_text_color(*C_TEXT)
        pdf.multi_cell(CONTENT_W - 44, 5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    pdf.ln(3)
    pdf.h2("What Each Coloured Box Means")

    box_guide = [
        (C_PLAIN_BG, C_PRIMARY, "Blue box",
         "Plain English — start here. Written for anyone."),
        (C_TECH_BG,  C_MUTED,   "Grey box",
         "Technical detail — confidence formulas, IDs, and system internals."),
        (C_WARN_BG,  C_ORANGE,  "Yellow box",
         "Warning — something that needs human attention, or is uncertain."),
    ]
    for bg, border, lbl, desc in box_guide:
        y0 = pdf.get_y()
        pdf.set_fill_color(*bg)
        pdf.set_draw_color(*border)
        pdf.set_line_width(0.4)
        pdf.rect(MARGIN, y0, CONTENT_W, 10, "FD")

        pdf.set_xy(MARGIN + 3, y0 + 2.5)
        pdf.set_font(FONT_NAME, "B", 8)
        pdf.set_text_color(*border)
        pdf.cell(32, 5, lbl)
        pdf.set_font(FONT_NAME, "", 8.5)
        pdf.set_text_color(*C_TEXT)
        pdf.cell(CONTENT_W - 35, 5, desc)
        pdf.ln(13)


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Entity Resolution
# ──────────────────────────────────────────────────────────────────────────────

def build_er_section(pdf: ExplainabilityPDF, er: dict):
    pdf.add_page()
    pdf.h1("Section 2 — Who Was Involved")
    pdf.small("Source file: CASE_ATM_001_er_output.json  (Entity Resolution Agent)")

    n_entities = er.get("entity_count", 0)
    er_status  = er.get("output_classification", "UNKNOWN")
    er_reason  = safe(er.get("output_classification_reason", ""), 300)

    pdf.plain_box(
        "Who did the system identify?",
        f"The system analysed all pieces of evidence and identified {n_entities} distinct "
        f"individual(s) who appear across the footage, audio recordings, and written reports.\n\n"
        f"  • Person 14 — identified as the primary suspect. Seen on CCTV at the ATM "
        f"across multiple camera angles over a 4-minute window.\n\n"
        f"  • Witness cluster — a second individual (or group of sources from the same "
        f"person) who reported the incident. Three different evidence sources "
        f"(video, audio, text report) were grouped together, but their exact identity "
        f"and physical location at the time are not fully confirmed.\n\n"
        f"System confidence in these identifications: {er_status}. "
        f"A human investigator should review the witness cluster before drawing any conclusions.",
    )

    pdf.tech_box("Entity Resolution — system status", [
        f"output_classification : {er_status}",
        f"reason               : {er_reason}",
        f"entity_count         : {n_entities}",
        f"conflicts_detected   : {er.get('conflicts_detected', 0)}",
        f"llm_calls_made       : {er.get('llm_calls_made', 0)} / {er.get('llm_calls_budget', 0)}",
        f"location_similarity  : {er.get('location_similarity_backend', 'unknown')}",
        f"processing_time      : {er.get('total_processing_time_sec', '?')}s",
    ])

    # ── One card per entity ───────────────────────────────────────────────────
    for ent in er.get("canonical_entities", []):
        pdf.h2(f"Identified Person: {ent.get('primary_alias', 'Unknown')}")

        conf    = ent.get("confidence_score", 0)
        label, color = conf_label(conf)
        role    = ", ".join(ent.get("roles", ["unknown"]))
        aliases = ent.get("aliases", [])
        locs    = ent.get("locations", [])
        mods    = ent.get("modalities", [])
        srcs    = ent.get("sources", [])
        t0_raw  = ent.get("earliest_timestamp", "")
        t1_raw  = ent.get("latest_timestamp", "")
        span    = ent.get("time_span_seconds", 0)

        # Confidence badge + role line
        y0 = pdf.get_y()
        pdf.conf_badge(conf, MARGIN, y0)
        pdf.set_xy(MARGIN + 34, y0 + 1)
        pdf.set_font(FONT_NAME, "", 8.5)
        pdf.cell(
            0, 6,
            f"  Role: {role.upper()}  ·  {ent.get('total_mentions', 0)} evidence mention(s)"
        )
        pdf.ln(9)

        # Plain English per entity
        if "suspect" in role:
            plain = (
                f"This person appears in {len(srcs)} separate pieces of evidence "
                f"({', '.join(srcs)}), all captured on {', '.join(mods)} recordings. "
                f"They were first seen at {time_str(t0_raw)} and last recorded at "
                f"{time_str(t1_raw)} — a window of {span // 60} min {span % 60} sec.\n\n"
                f"Locations recorded: {'; '.join(locs)}\n\n"
                f"The system is {label} confidence ({conf:.0%}) that all these "
                f"sightings are the same person."
            )
        else:
            plain = (
                f"This entry groups {len(srcs)} evidence items ({', '.join(srcs)}) that "
                f"may belong to the same witness, recorded via {', '.join(mods)}. "
                f"All three are timestamped at exactly {time_str(t0_raw)}.\n\n"
                f"Locations reported: {'; '.join(locs)}\n\n"
                f"⚠ CAUTION: The evidence shows this person at conflicting locations at "
                f"the same time. The 'email_74' item is likely a text message filed with "
                f"police — it may represent the witness's report, not their physical "
                f"location. This needs human confirmation.\n\n"
                f"Confidence that this grouping is correct: {label} ({conf:.0%})."
            )
        pdf.plain_box(f"About: {ent.get('primary_alias', 'Unknown')}", plain)

        if len(aliases) > 1:
            pdf.warn_box(
                f"⚠  This person appears under multiple names in the evidence: "
                f"{', '.join(aliases)}. "
                f"The system grouped them as one entity based on timing, location, and context. "
                f"An investigator should verify this grouping."
            )

        pdf.tech_box(f"Entity: {ent.get('entity_id', '')}", [
            f"entity_id          : {ent.get('entity_id', '')}",
            f"primary_alias      : {ent.get('primary_alias', '')}",
            f"all aliases        : {', '.join(aliases)}",
            f"confidence_score   : {conf:.4f}  ({label})",
            f"confirmed_edges    : {ent.get('confirmed_edges', 0)}",
            f"candidate_edges    : {ent.get('candidate_edges', 0)}",
            f"obs_ids (sources)  : {', '.join(srcs)}",
            f"modalities         : {', '.join(mods)}",
            f"locations          : {'; '.join(locs)}",
            f"time window        : {t0_raw} → {t1_raw}  ({span}s)",
        ])
        pdf.divider()

    # ── ER conflicts ──────────────────────────────────────────────────────────
    conflicts = er.get("conflicts", [])
    if conflicts:
        pdf.h2("Conflicts Flagged by the System")
        for c in conflicts:
            ctype  = c.get("type", "unknown").replace("_", " ").title()
            cclust = c.get("cluster_id", "?")
            cdesc  = c.get("detail", "")
            pdf.warn_box(
                f"⚠  CONFLICT TYPE: {ctype}   (Cluster: {cclust})\n\n"
                f"Technical detail: {cdesc}\n\n"
                f"What this means in plain English: The evidence for the witness cluster "
                f"places the same person at two different physical locations at exactly "
                f"the same time — this is physically impossible. A human investigator must "
                f"determine which location record is correct, or whether these are actually "
                f"two different people."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Timeline of Events
# ──────────────────────────────────────────────────────────────────────────────

def build_timeline_section(pdf: ExplainabilityPDF, tl: dict):
    pdf.add_page()
    pdf.h1("Section 3 — What Happened: Timeline of Events")
    pdf.small("Source file: CASE_ATM_001_timeline_V3.json  (Timeline Agent — final version)")

    events   = tl.get("events", [])
    n        = len(events)
    avg_conf = sum(e.get("confidence", 0) for e in events) / max(n, 1)
    n_conf   = sum(1 for e in events if e.get("conflict_flag"))

    pdf.plain_box(
        "The sequence of events",
        f"Based on all available evidence, the system reconstructed {n} key events. "
        f"These are shown in the order they happened. Each event is rated HIGH, MEDIUM, "
        f"or LOW confidence depending on how much supporting evidence exists.\n\n"
        f"Average confidence across all events: {avg_conf:.0%}. "
        f"{n_conf} event(s) were flagged with conflicting information and are marked ⚠. "
        f"Events rated LOW confidence should be treated as unconfirmed leads, not facts.",
    )

    for i, ev in enumerate(events, 1):
        conf     = ev.get("confidence", 0)
        label, _ = conf_label(conf)
        ts       = ev.get("timestamp", "")
        t        = time_str(ts)
        entity   = ev.get("primary_alias", "Unknown")
        location = ev.get("location", "unknown location")
        content  = ev.get("content", "")
        tags     = ev.get("action_tags", [])
        modality = ev.get("modality", "unknown")
        obs_ids  = ev.get("obs_ids", [])
        conflict = ev.get("conflict_flag", False)
        event_id = ev.get("event_id", f"EVT_{i:03d}")
        reasoning= ev.get("reasoning", [])
        role     = ev.get("role", "unknown")
        cnote    = ev.get("conflict_note", "")

        # ── Event header bar ──────────────────────────────────────────────────
        y0 = pdf.get_y()
        if y0 > PAGE_H - 65:
            pdf.add_page()
            y0 = pdf.get_y()

        pdf.set_fill_color(*C_PRIMARY)
        pdf.rect(MARGIN, y0, CONTENT_W, 8, "F")

        pdf.set_xy(MARGIN + 2, y0 + 1.5)
        pdf.set_font(FONT_NAME, "B", 8.5)
        pdf.set_text_color(*C_WHITE)
        pdf.cell(20, 5, f"EVENT {i}")
        pdf.set_font(FONT_NAME, "", 8.5)
        pdf.cell(30, 5, f"  {t}")
        pdf.cell(0,  5, f"  {entity}  ·  {safe(location, 50)}")
        pdf.set_text_color(*C_TEXT)
        pdf.set_y(y0 + 10)

        # Confidence badge (top-right of header)
        pdf.conf_badge(conf, PAGE_W - MARGIN - 32, y0 + 0.5)

        # ── Plain English ─────────────────────────────────────────────────────
        action_desc = action_to_plain(tags)
        plain = (
            f"{entity} was captured {action_desc} at:\n"
            f"{location}\n\n"
            f"Time: {t}   Evidence type: {modality.title()}\n"
            f"Supporting evidence item(s): {', '.join(obs_ids)}"
        )
        if content:
            plain += f"\n\nEvidence note: \"{safe(content, 220)}\""
        if conflict:
            plain += (
                "\n\n⚠  This event has conflicting information across evidence sources. "
                "See the warning box below."
            )
        pdf.plain_box(f"Event {i} — What happened", plain)

        # ── Technical detail ──────────────────────────────────────────────────
        tech_lines = [
            f"event_id      : {event_id}",
            f"obs_ids       : {', '.join(obs_ids)}",
            f"action_tags   : {', '.join(tags) if tags else 'none detected'}",
            f"confidence    : {conf:.4f}  →  {label}",
            f"modality      : {modality}   role: {role}",
            f"conflict_flag : {conflict}",
        ]
        if reasoning:
            tech_lines.append(f"reasoning     : {'; '.join(reasoning)}")
        if cnote:
            tech_lines.append(f"conflict_note : {cnote}")
        pdf.tech_box(f"Event {i} — {event_id}", tech_lines)

        if conflict:
            pdf.warn_box(
                f"⚠  The system detected a conflict for this event. "
                + (f"{cnote}  " if cnote else "")
                + "The confidence score has been reduced. "
                "An investigator should verify the details of this event before "
                "using it as evidence."
            )
        pdf.ln(2)

    # ── Timeline narrative ────────────────────────────────────────────────────
    narrative = tl.get("narrative", [])
    if narrative:
        pdf.add_page()
        pdf.h2("System Narrative — Joined-Up Summary")
        pdf.plain_box(
            "What the AI thinks happened — the full story",
            "Below is the system's own summary of events, written as a connected narrative. "
            "This is automatically generated from the timeline and should be read alongside "
            "the individual events above — not as a substitute for them.",
        )
        for line in narrative:
            txt = line.get("text", "") if isinstance(line, dict) else str(line)
            if txt:
                pdf.set_font(FONT_NAME, "", 8.5)
                pdf.set_text_color(*C_TEXT)
                pdf.multi_cell(0, 5, f"  • {txt}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Critique (Evidence Gaps)
# ──────────────────────────────────────────────────────────────────────────────

def build_critique_section(
    pdf: ExplainabilityPDF,
    c1: dict, c2: dict, c3: dict,
):
    pdf.add_page()
    pdf.h1("Section 4 — Evidence Gaps: What the System Found Missing")
    pdf.small(
        "Source files: CASE_ATM_001_critique_C1/C2/C3.json  (Critique Agent — 3 rounds)"
    )

    pdf.plain_box(
        "What are evidence gaps?",
        "After building the timeline, the system reviewed its own work and looked for "
        "problems — places where evidence was weak, missing, or contradictory. "
        "These problems are called 'gaps.' The system then ran the analysis again "
        "(up to 3 times) to try to fix the gaps it found.\n\n"
        "Each gap is given a severity score from 0 (minor) to 1 (critical). "
        "The sections below show what was found in each review round.",
    )

    gap_severity_plain = {
        "info":     "Informational — no action needed",
        "low":      "Low — worth noting but not blocking",
        "medium":   "Medium — investigate when possible",
        "high":     "High — should be resolved before using this evidence",
        "critical": "Critical — must be resolved; do not rely on this evidence",
    }

    rounds = [
        ("Round 1 (C1)", c1, "CASE_ATM_001_critique_C1.json"),
        ("Round 2 (C2)", c2, "CASE_ATM_001_critique_C2.json"),
        ("Round 3 (C3)", c3, "CASE_ATM_001_critique_C3.json"),
    ]

    for round_label, critique, src_file in rounds:
        if not critique:
            continue

        pdf.h2(f"Critique {round_label}")
        gaps    = critique.get("gaps", [])

        # overall_severity: C1 uses overall_score, C2/C3 use overall_severity
        ov_sev  = critique.get("overall_severity", critique.get("overall_score"))
        sev_str = f"{float(ov_sev):.3f}" if isinstance(ov_sev, float) else "N/A"

        # verdict: C1 has it, C3 doesn't — fall back to narrative_summary
        verdict = safe(
            critique.get("verdict") or critique.get("narrative_summary") or "", 300
        )

        gap_plain_list = ""
        for g in gaps:
            # C1 uses "detail" and "check"; C2/C3 use "description" and "gap_id"
            g_sev  = g.get("severity", "info")
            g_desc = safe(
                g.get("description") or g.get("detail") or g.get("narrative_label") or "", 180
            )
            g_sev_plain = gap_severity_plain.get(
                str(g_sev).lower(),
                f"{float(g_sev):.2f}" if isinstance(g_sev, float) else str(g_sev),
            )
            if g_desc:
                gap_plain_list += f"  • [{g_sev_plain}]  {g_desc}\n"

        plain = (
            f"Number of gaps found: {len(gaps)}\n"
            f"Overall severity: {sev_str}  (0 = no issues, 1 = critical)\n\n"
            + (gap_plain_list.strip() if gap_plain_list else "No gaps found in this round.")
            + (f"\n\nSystem verdict: {verdict}" if verdict else "")
        )
        pdf.plain_box(f"{round_label} — What gaps were found", plain)

        if gaps:
            tech_lines = []
            for g in gaps:
                # Handle both C1 format and C2/C3 format
                gid   = g.get("gap_id") or g.get("check") or "—"
                gtype = (g.get("gap_type") or g.get("issue_type") or "unknown").replace("_", " ").title()
                gsev  = g.get("severity", "info")
                gdesc = safe(
                    g.get("description") or g.get("detail") or g.get("narrative_label") or "", 160
                )
                gfix  = safe(g.get("fix_hint", ""), 120)
                gsev_str = f"{float(gsev):.3f}" if isinstance(gsev, float) else str(gsev)

                tech_lines.append(f"[{gid}]  {gtype}  |  severity: {gsev_str}")
                tech_lines.append(f"        {gdesc}")
                if gfix:
                    tech_lines.append(f"        Fix hint: {gfix}")
                tech_lines.append("")
            pdf.tech_box(f"{round_label} — Full gap list", tech_lines)

        pdf.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — Showrunner (System Decisions)
# ──────────────────────────────────────────────────────────────────────────────

def build_showrunner_section(
    pdf: ExplainabilityPDF,
    s1: dict, s2: dict, s3: dict,
):
    pdf.add_page()
    pdf.h1("Section 5 — System Decisions: How the AI Revised Its Analysis")
    pdf.small(
        "Source files: CASE_ATM_001_showrunner_C1/C2/C3.json  (Showrunner Agent — 3 rounds)"
    )

    pdf.plain_box(
        "What is the Showrunner?",
        "The Showrunner is the part of ForenSynth that decides what to do after each "
        "review round. After the Critique Agent finds gaps, the Showrunner chooses one "
        "of four actions:\n\n"
        "  • Accept — the result is good enough, no more changes needed\n"
        "  • Re-run timeline — redo the event reconstruction with corrections\n"
        "  • Re-run entity resolution — redo the person identification\n"
        "  • Human review — the AI is not confident enough; a person must decide\n\n"
        "The system can run up to 3 rounds. Below you can see what was decided after "
        "each round and why.",
    )

    action_plain_map = {
        "no_action":       "Accepted — no further changes needed",
        "re_run_timeline": "Re-ran the timeline analysis",
        "re_run_er":       "Re-ran the entity identification",
        "human_review":    "Sent for human review (AI not confident enough)",
    }

    for round_label, sr in [("Round 1 (C1)", s1), ("Round 2 (C2)", s2), ("Round 3 (C3)", s3)]:
        if not sr:
            continue

        action    = sr.get("action", "unknown")
        # field is called "reasoning" in the showrunner JSON (not "reason")
        reasoning = safe(sr.get("reasoning", sr.get("reason", "No reason recorded.")), 400)
        converged = sr.get("converged", False)
        plain_act = action_plain_map.get(action, action.replace("_", " ").title())
        out_case  = sr.get("output_case", "")
        addressed = sr.get("issues_addressed", [])
        deferred  = sr.get("issues_deferred", [])

        pdf.h2(f"Showrunner {round_label}")
        pdf.plain_box(
            f"{round_label} — What decision was made?",
            f"Decision: {plain_act}\n\n"
            f"Reason: {reasoning}\n\n"
            + (f"Issues fixed this round: {', '.join(addressed) if addressed else 'None'}\n"
               f"Issues deferred to next round: {', '.join(deferred) if deferred else 'None'}\n\n"
               if addressed or deferred else "")
            + f"Did the analysis reach a stable result? {'Yes' if converged else 'No — another round was needed'}",
        )
        pdf.tech_box(f"{round_label} — Showrunner detail", [
            f"action           : {action}",
            f"input_tl_version : {sr.get('input_tl_version', '?')}  →  output_tl_version: {sr.get('output_tl_version', '?')}",
            f"output_case      : {out_case}",
            f"converged        : {converged}",
            f"issues_addressed : {', '.join(addressed) if addressed else 'none'}",
            f"issues_deferred  : {', '.join(deferred) if deferred else 'none'}",
            f"reasoning        : {reasoning}",
        ])
        pdf.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 — Human Review Checklist
# ──────────────────────────────────────────────────────────────────────────────

def build_human_review(
    pdf: ExplainabilityPDF,
    er: dict, tl: dict, c3: dict, s3: dict,
):
    pdf.add_page()
    pdf.h1("Section 6 — What Still Needs Human Review")

    pdf.plain_box(
        "Why human review is always needed",
        "AI systems like ForenSynth are tools to assist investigators, not to replace them. "
        "The following items were identified by the system as requiring a human decision "
        "before any conclusions can be used as evidence. Each item explains why the system "
        "could not resolve it automatically.",
    )

    items = []

    # ER ambiguous
    if er and er.get("output_classification") == "AMBIGUOUS":
        items.append((
            "Witness identity could not be confirmed",
            "The system grouped Person_55, Speaker_C, and email_74 as one witness, but the "
            "evidence places them at different physical locations at exactly the same time. "
            "A human investigator must determine whether these are the same person or "
            "different individuals.",
            True,
        ))

    # Unresolved entities
    unresolved = (tl or {}).get("unresolved_entities", [])
    if unresolved:
        items.append((
            f"Unresolved identifier(s) in the timeline ({len(unresolved)} item(s))",
            f"The following names could not be matched to real people: "
            f"{', '.join(safe(u, 50) for u in unresolved[:6])}. "
            f"These may be document or file identifiers rather than person names.",
            False,
        ))

    # Conflict events
    conflict_evs = [e for e in (tl or {}).get("events", []) if e.get("conflict_flag")]
    if conflict_evs:
        items.append((
            f"{len(conflict_evs)} timeline event(s) have conflicting information",
            "One or more events have evidence from different sources that disagrees. "
            "An investigator should examine these events and determine which account "
            "is accurate.",
            True,
        ))

    # Showrunner requested human review
    if s3 and s3.get("action") == "human_review":
        items.append((
            "The system itself requested human review after the final round",
            safe(s3.get("reason", "Reason not recorded."), 250),
            True,
        ))

    # High-severity remaining gaps
    if c3:
        critical_gaps = [
            g for g in c3.get("gaps", [])
            if isinstance(g.get("severity"), float) and g["severity"] >= 0.70
        ]
        if critical_gaps:
            descs = "; ".join(safe(g.get("description", ""), 80) for g in critical_gaps[:3])
            items.append((
                f"{len(critical_gaps)} critical evidence gap(s) remain after 3 rounds",
                f"These gaps were not resolved after three rounds of analysis: {descs}",
                True,
            ))

    if not items:
        pdf.body(
            "No outstanding human review items were identified. "
            "The system reached a stable, internally consistent result."
        )
    else:
        for title, desc, is_high in items:
            pcolor = C_RED if is_high else C_ORANGE
            priority = "HIGH PRIORITY" if is_high else "MEDIUM PRIORITY"

            y0 = pdf.get_y()
            if y0 > PAGE_H - 45:
                pdf.add_page()
                y0 = pdf.get_y()

            # Left colour bar
            pdf.set_fill_color(*pcolor)
            pdf.rect(MARGIN, y0, 4, 22, "F")

            pdf.set_xy(MARGIN + 7, y0 + 2)
            pdf.set_font(FONT_NAME, "B", 9)
            pdf.set_text_color(*pcolor)
            pdf.cell(0, 5, f"⚠  {title}")

            pdf.set_xy(MARGIN + 7, y0 + 8)
            pdf.set_font(FONT_NAME, "", 8.5)
            pdf.set_text_color(*C_TEXT)
            pdf.multi_cell(CONTENT_W - 12, 5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.set_font(FONT_NAME, "B", 7.5)
            pdf.set_text_color(*pcolor)
            pdf.cell(0, 4, f"  → {priority}")
            pdf.set_text_color(*C_TEXT)
            pdf.ln(6)
            pdf.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 — Final Verdict
# ──────────────────────────────────────────────────────────────────────────────

def build_verdict(pdf: ExplainabilityPDF, tl: dict, er: dict):
    pdf.add_page()
    pdf.h1("Section 7 — Final Verdict")

    events         = (tl or {}).get("events", [])
    classification = (tl or {}).get("output_classification", "UNKNOWN")
    reason         = safe((tl or {}).get("output_classification_reason", ""), 300)
    avg_conf       = sum(e.get("confidence", 0) for e in events) / max(len(events), 1)
    n_conflict     = sum(1 for e in events if e.get("conflict_flag"))
    _, badge_color = conf_label(avg_conf)

    # Large verdict banner
    y0 = pdf.get_y()
    pdf.set_fill_color(*badge_color)
    pdf.rect(MARGIN, y0, CONTENT_W, 16, "F")
    pdf.set_xy(MARGIN, y0 + 2)
    pdf.set_font(FONT_NAME, "B", 17)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(0, 12, f"RECONSTRUCTION STATUS:  {classification}", align="C")
    pdf.set_text_color(*C_TEXT)
    pdf.set_y(y0 + 20)
    pdf.ln(3)

    # Plain English verdict
    if classification == "AMBIGUOUS":
        plain = (
            f"After analysing all available evidence and running the analysis three times, "
            f"the ForenSynth system produced an AMBIGUOUS reconstruction.\n\n"
            f"AMBIGUOUS means the AI found significant uncertainty. The average confidence "
            f"across all events is {avg_conf:.0%}, and {n_conflict} event(s) have "
            f"conflicting information between evidence sources.\n\n"
            f"This reconstruction should be used as a starting point for investigation only. "
            f"It must NOT be used as evidence in legal proceedings without further "
            f"human verification of the flagged items."
        )
    elif classification == "PARTIAL":
        plain = (
            f"The system produced a PARTIAL reconstruction with an average confidence "
            f"of {avg_conf:.0%}. {n_conflict} event(s) were flagged.\n\n"
            f"PARTIAL means the reconstruction is mostly reliable but has some gaps "
            f"or conflicts. Use it as a strong investigative lead, and verify the "
            f"flagged items before drawing conclusions."
        )
    else:
        plain = (
            f"The system produced a CLEAR reconstruction with an average confidence "
            f"of {avg_conf:.0%}. No unresolved conflicts were detected.\n\n"
            f"CLEAR means the reconstruction is well-supported by the evidence. "
            f"The timeline can be used with confidence as an investigative tool, "
            f"subject to normal forensic review procedures."
        )

    pdf.plain_box("Final Verdict — Plain English", plain)

    pdf.tech_box("Final Verdict — Technical detail", [
        f"output_classification              : {classification}",
        f"reason                             : {reason}",
        f"average_confidence                 : {avg_conf:.4f}  ({avg_conf:.0%})",
        f"total_events                       : {len(events)}",
        f"conflict_flagged_events            : {n_conflict}",
        f"er_classification                  : {(er or {}).get('output_classification', '?')}",
        f"timeline_agent_llm_backend         : {(tl or {}).get('llm_backend_used', '?')}",
        f"timeline_agent_llm_calls           : {(tl or {}).get('llm_calls_made', '?')}  "
        f"(calls made by the Timeline Agent during pipeline; this report uses no LLM)",
        f"total_pipeline_processing_time     : {(tl or {}).get('total_time_sec', '?')}s",
    ])

    pdf.ln(5)
    pdf.set_font(FONT_NAME, "I", 7.5)
    pdf.set_text_color(*C_MUTED)
    pdf.multi_cell(
        0, 4.5,
        "This report was generated automatically by the ForenSynth AI Pipeline. "
        "It does not constitute legal advice or a definitive forensic finding. "
        "All conclusions are probabilistic and must be reviewed by a qualified "
        "forensic investigator before use in legal proceedings.",
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_text_color(*C_TEXT)


# ──────────────────────────────────────────────────────────────────────────────
# Section 8 — Plain English Narrative Summary (final page)
# ──────────────────────────────────────────────────────────────────────────────

def build_narrative_summary(
    pdf: ExplainabilityPDF,
    er: dict, tl: dict, c3: dict, s3: dict,
):
    """
    A self-contained plain-English summary page. Written so that someone can
    read just this page and understand the case without reading anything else.
    Covers: who was identified, what happened and when, what evidence proves
    each event, what is uncertain, and what still needs investigation.
    """
    pdf.add_page()
    pdf.h1("Section 8 — Case Narrative: The Full Picture in Plain English")

    pdf.plain_box(
        "Read this if you want a quick summary of everything",
        "This page tells the complete story of what the ForenSynth AI system found, "
        "written in plain language. You can read just this page to understand the "
        "case — without going through the rest of the report. Every claim below "
        "is mapped to the specific piece of evidence that supports it.",
    )

    events   = (tl or {}).get("events", [])
    entities = (er or {}).get("canonical_entities", [])

    # ── Who was identified ────────────────────────────────────────────────────
    pdf.h2("The People Involved")

    for ent in entities:
        alias   = ent.get("primary_alias", "Unknown")
        aliases = ent.get("aliases", [])
        role    = ", ".join(ent.get("roles", ["unknown"]))
        conf    = ent.get("confidence_score", 0)
        srcs    = ent.get("sources", [])
        mods    = ent.get("modalities", [])
        locs    = ent.get("locations", [])
        label, color = conf_label(conf)

        # Inline confidence pill before description
        y0 = pdf.get_y()
        pdf.conf_badge(conf, MARGIN, y0, w=26, h=6)
        pdf.set_xy(MARGIN + 30, y0 + 1)
        pdf.set_font(FONT_NAME, "B", 9)
        pdf.set_text_color(*C_PRIMARY)
        extra = f"  (also referred to as: {', '.join(a for a in aliases if a != alias)})" \
                if len(aliases) > 1 else ""
        pdf.cell(0, 4, f"{alias}{extra}  —  {role.title()}")
        pdf.set_text_color(*C_TEXT)
        pdf.ln(8)

        pdf.set_font(FONT_NAME, "", 9)
        if "suspect" in role:
            desc = (
                f"{alias} is identified as the primary suspect. They appear in "
                f"{len(srcs)} piece(s) of {', '.join(mods)} evidence "
                f"({', '.join(srcs)}), recorded at the ATM across three different "
                f"camera angles. The system is {label} confidence ({conf:.0%}) that "
                f"all these sightings are the same person."
            )
        else:
            desc = (
                f"A witness appears in {len(srcs)} piece(s) of evidence "
                f"({', '.join(srcs)}) captured via {', '.join(mods)}. "
                f"These sources were grouped together because they all report the same "
                f"incident at the same time. However, the system flagged a conflict: "
                f"one of the evidence items (email_74) places this person at the local "
                f"police station, while another (Person_55 on video) shows them at the "
                f"ATM. This may mean the witness sent a text message to police from "
                f"their phone while at the ATM — but this needs a human to confirm. "
                f"Confidence in this grouping: {label} ({conf:.0%})."
            )
        pdf.multi_cell(0, 5.5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    pdf.divider()

    # ── What happened, event by event ─────────────────────────────────────────
    pdf.h2("What Happened — Event by Event, with Evidence")

    event_narrative = {
        # hand-crafted plain-English for each known event index
        # keyed by action_tags tuple; fallback to generic
    }

    for i, ev in enumerate(events, 1):
        conf     = ev.get("confidence", 0)
        label, _ = conf_label(conf)
        ts       = ev.get("timestamp", "")
        t        = time_str(ts)
        entity   = ev.get("primary_alias", "Unknown")
        location = ev.get("location", "")
        content  = ev.get("content", "")
        tags     = ev.get("action_tags", [])
        obs_ids  = ev.get("obs_ids", [])
        modality = ev.get("modality", "")
        conflict = ev.get("conflict_flag", False)

        action_desc = action_to_plain(tags)

        # Build evidence mapping string
        evidence_map = f"Supported by: {', '.join(obs_ids)} ({modality.title()} evidence)"

        y0 = pdf.get_y()
        if y0 > PAGE_H - 50:
            pdf.add_page()
            y0 = pdf.get_y()

        # Event number + time in left margin stripe
        pdf.set_fill_color(*C_ACCENT)
        pdf.rect(MARGIN, y0, 10, 14, "F")
        pdf.set_xy(MARGIN, y0 + 1)
        pdf.set_font(FONT_NAME, "B", 8)
        pdf.set_text_color(*C_WHITE)
        pdf.cell(10, 6, str(i), align="C")
        pdf.set_xy(MARGIN, y0 + 7)
        pdf.set_font(FONT_NAME, "", 6)
        pdf.cell(10, 4, t[:5], align="C")
        pdf.set_text_color(*C_TEXT)

        # Event body
        pdf.set_xy(MARGIN + 13, y0 + 2)
        pdf.set_font(FONT_NAME, "B", 9)
        pdf.cell(0, 5, f"{entity} — {action_desc.title()}")
        pdf.set_xy(MARGIN + 13, y0 + 8)
        pdf.set_font(FONT_NAME, "", 8.5)
        pdf.multi_cell(CONTENT_W - 13, 5, evidence_map, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        if content:
            pdf.set_xy(MARGIN + 13, pdf.get_y())
            pdf.set_font(FONT_NAME, "I", 8)
            pdf.set_text_color(*C_MUTED)
            pdf.multi_cell(CONTENT_W - 13, 4.5, f"\"{safe(content, 180)}\"",
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*C_TEXT)

        # Confidence badge for this event
        pdf.conf_badge(conf, PAGE_W - MARGIN - 28, y0, w=28, h=6)

        if conflict:
            pdf.set_font(FONT_NAME, "B", 7.5)
            pdf.set_text_color(*C_RED)
            pdf.cell(0, 4, "  ⚠ Conflicting evidence — needs investigator review")
            pdf.set_text_color(*C_TEXT)
        pdf.ln(6)

    pdf.divider()

    # ── What is confirmed vs uncertain ────────────────────────────────────────
    pdf.h2("What Is Confirmed vs. What Is Uncertain")

    high_evs   = [e for e in events if e.get("confidence", 0) >= HIGH_CONF]
    medium_evs = [e for e in events if LOW_CONF <= e.get("confidence", 0) < HIGH_CONF]
    low_evs    = [e for e in events if e.get("confidence", 0) < LOW_CONF]

    confirmed_text = (
        f"{len(high_evs)} event(s) are HIGH confidence — the evidence for these "
        f"is strong and consistent across multiple sources:\n"
        + "\n".join(
            f"  • Event {events.index(e)+1}: {e.get('primary_alias','?')} — "
            f"{action_to_plain(e.get('action_tags',[]))} at {time_str(e.get('timestamp',''))}"
            for e in high_evs
        )
    )

    uncertain_text = ""
    if medium_evs or low_evs:
        uncertain_text = (
            f"\n\n{len(medium_evs) + len(low_evs)} event(s) are MEDIUM or LOW confidence "
            f"and should be treated as investigative leads, not established facts:\n"
            + "\n".join(
                f"  • Event {events.index(e)+1}: {e.get('primary_alias','?')} — "
                f"{action_to_plain(e.get('action_tags',[]))} "
                f"({conf_label(e.get('confidence',0))[0]}, {e.get('confidence',0):.0%})"
                for e in medium_evs + low_evs
            )
        )

    pdf.plain_box("Confirmed vs. Uncertain", confirmed_text + uncertain_text)

    # ── What still needs investigation ────────────────────────────────────────
    pdf.h2("What Still Needs Investigation")

    remaining_gaps = (c3 or {}).get("gaps", [])
    s3_action      = (s3 or {}).get("action", "")
    unresolved     = (tl or {}).get("unresolved_entities", [])

    investigation_items = []

    if s3_action == "human_review":
        investigation_items.append(
            "The AI system itself escalated this case for human review after its "
            "third and final analysis round. It reached the limit of what it can "
            "determine automatically."
        )
    if unresolved:
        investigation_items.append(
            f"The system could not identify who {', '.join(safe(u,30) for u in unresolved[:4])} "
            f"are. These names appear in the evidence but could not be matched to real people."
        )
    conflict_evs = [e for e in events if e.get("conflict_flag")]
    if conflict_evs:
        investigation_items.append(
            f"{len(conflict_evs)} event(s) have conflicting information between evidence "
            f"sources. An investigator needs to determine which account is correct."
        )
    high_gaps = [
        g for g in remaining_gaps
        if isinstance(g.get("severity"), float) and g["severity"] >= 0.70
    ]
    for g in high_gaps[:3]:
        desc = g.get("description") or g.get("narrative_label") or g.get("detail") or ""
        if desc:
            investigation_items.append(safe(desc, 200))

    if investigation_items:
        text = "\n\n".join(f"  {chr(9679)}  {item}" for item in investigation_items)
        pdf.warn_box("⚠  Items requiring human investigation:\n\n" + text)
    else:
        pdf.body("No outstanding investigation items. The system reached a complete result.")

    # ── One-line bottom summary ───────────────────────────────────────────────
    pdf.ln(4)
    classification = (tl or {}).get("output_classification", "UNKNOWN")
    avg_conf = sum(e.get("confidence", 0) for e in events) / max(len(events), 1)
    pdf.set_fill_color(*C_PRIMARY)
    pdf.rect(MARGIN, pdf.get_y(), CONTENT_W, 10, "F")
    pdf.set_xy(MARGIN + 3, pdf.get_y() + 2.5)
    pdf.set_font(FONT_NAME, "B", 9)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(
        0, 5,
        f"ForenSynth overall verdict: {classification}  "
        f"·  Average confidence: {avg_conf:.0%}  "
        f"·  {len(events)} events reconstructed from {sum(len(e.get('obs_ids',[])) for e in events)} evidence items"
    )
    pdf.set_text_color(*C_TEXT)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def generate_explainability_report(
    pipeline_outputs: Dict[str, Any],
    output_dir: str,
    case_id: str,
) -> Optional[str]:
    """
    Build the dual-layer explainability PDF from in-memory pipeline output
    dicts and write it to `output_dir`. Returns the written path, or None
    if generation failed for any reason (never raises).

    pipeline_outputs keys (all optional, missing ones render as empty):
        er, timeline_v3, critique_c1, critique_c2, critique_c3,
        showrunner_c1, showrunner_c2, showrunner_c3
    """
    try:
        er = pipeline_outputs.get("er") or {}
        tl = pipeline_outputs.get("timeline_v3") or {}
        c1 = pipeline_outputs.get("critique_c1") or {}
        c2 = pipeline_outputs.get("critique_c2") or {}
        c3 = pipeline_outputs.get("critique_c3") or {}
        s1 = pipeline_outputs.get("showrunner_c1") or {}
        s2 = pipeline_outputs.get("showrunner_c2") or {}
        s3 = pipeline_outputs.get("showrunner_c3") or {}

        pdf = ExplainabilityPDF(case_id)

        build_cover(pdf, case_id, er, tl)
        build_guide(pdf)
        build_er_section(pdf, er)
        build_timeline_section(pdf, tl)
        build_critique_section(pdf, c1, c2, c3)
        build_showrunner_section(pdf, s1, s2, s3)
        build_human_review(pdf, er, tl, c3, s3)
        build_verdict(pdf, tl, er)
        build_narrative_summary(pdf, er, tl, c3, s3)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{case_id}_explainability_report_v2.pdf"
        pdf.output(str(out_path))
        log.info("Explainability report saved -> %s (%d pages)", out_path, pdf.page)
        return str(out_path)
    except Exception:
        log.exception("Explainability report generation failed for %s", case_id)
        return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Standalone test: build an explainability report from JSON files on disk."
    )
    ap.add_argument("--input-dir", required=True, help="Directory containing the pipeline JSON outputs")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--output-dir", default="./output/reports")
    args = ap.parse_args()

    base = Path(args.input_dir)
    outputs = {
        "er":            load_json(base / f"{args.case_id}_er_output.json"),
        "timeline_v3":   load_json(base / f"{args.case_id}_timeline_V3.json"),
        "critique_c1":   load_json(base / f"{args.case_id}_critique_C1.json"),
        "critique_c2":   load_json(base / f"{args.case_id}_critique_C2.json"),
        "critique_c3":   load_json(base / f"{args.case_id}_critique_C3.json"),
        "showrunner_c1": load_json(base / f"{args.case_id}_showrunner_C1.json"),
        "showrunner_c2": load_json(base / f"{args.case_id}_showrunner_C2.json"),
        "showrunner_c3": load_json(base / f"{args.case_id}_showrunner_C3.json"),
    }
    print("Files loaded:")
    for name, obj in outputs.items():
        print(f"  {name}: {'OK' if obj else 'MISSING'}")

    result = generate_explainability_report(outputs, args.output_dir, args.case_id)
    print(f"\nReport written -> {result}" if result else "\nReport generation FAILED.")