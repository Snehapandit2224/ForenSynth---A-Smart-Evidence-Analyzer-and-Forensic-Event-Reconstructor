#!/usr/bin/env python3
"""
ForenSynth – generate_comic.py
Generates a forensic comic strip SVG from a completed pipeline case.

Usage:
    python pipeline\generate_comic.py --case CASE_ATM_001
    python pipeline\generate_comic.py --case CASE_ATM_001 --tl-version V2
    python pipeline\generate_comic.py --all
    python pipeline\generate_comic.py --case CASE_ATM_001 --output output\comics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

_root = Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memory_store import ForenSynthMemory, _load

# ── Color palette ─────────────────────────────────────────────────────────────

ROLE_COLORS = {
    "suspect": {
        "body": "#3C3489", "bg": "#EEEDFE", "border": "#534AB7",
        "caption_bg": "#3C3489", "caption_text": "#CECBF6",
        "badge_bg": "#534AB7",
    },
    "witness": {
        "body": "#7F77DD", "bg": "#E1F5EE", "border": "#0F6E56",
        "caption_bg": "#085041", "caption_text": "#9FE1CB",
        "badge_bg": "#1D9E75",
    },
    "unknown": {
        "body": "#888780", "bg": "#F1EFE8", "border": "#5F5E5A",
        "caption_bg": "#444441", "caption_text": "#D3D1C7",
        "badge_bg": "#5F5E5A",
    },
}

ACTION_COLORS = {
    "APPROACH": "#1D9E75", "ENTER": "#1D9E75",
    "WITHDRAW": "#993C1D", "TAMPER": "#993C1D",
    "STEAL": "#993C1D",    "EXIT": "#0F6E56",
    "FLEE": "#534AB7",     "OBSERVE": "#534AB7",
    "REPORT": "#534AB7",   "WORK": "#BA7517",
    "NAVIGATE": "#BA7517", "LOITER": "#854F0B",
    "COMMUNICATE": "#185FA5", "CONFIRM": "#185FA5",
}


def _truncate(s, n):
    s = str(s)
    return s[:n-3] + "..." if len(s) > n else s

def _xml(s):
    """Escape special characters for safe SVG/XML embedding."""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\u2026", "...")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'"))

def _conf_color(conf):
    if conf >= 0.85: return "#1D9E75"
    if conf >= 0.70: return "#BA7517"
    return "#993C1D"

def _person(cx, cy, color, scale=1.0):
    r  = int(12 * scale)
    bw = int(12 * scale)
    bh = int(26 * scale)
    lw = int(2.5 * scale)
    return "\n".join([
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>',
        f'<rect x="{cx-bw//2}" y="{cy+r}" width="{bw}" height="{bh}" rx="{int(4*scale)}" fill="{color}"/>',
        f'<line x1="{cx-bw//2}" y1="{cy+r+8}" x2="{cx-bw//2-12}" y2="{cy+r+20}" stroke="{color}" stroke-width="{lw}" stroke-linecap="round"/>',
        f'<line x1="{cx+bw//2}" y1="{cy+r+6}" x2="{cx+bw//2+12}" y2="{cy+r+18}" stroke="{color}" stroke-width="{lw}" stroke-linecap="round"/>',
        f'<line x1="{cx-4}" y1="{cy+r+bh}" x2="{cx-10}" y2="{cy+r+bh+18}" stroke="{color}" stroke-width="{lw}" stroke-linecap="round"/>',
        f'<line x1="{cx+4}" y1="{cy+r+bh}" x2="{cx+10}" y2="{cy+r+bh+18}" stroke="{color}" stroke-width="{lw}" stroke-linecap="round"/>',
    ])

def _atm(x, y, w=55, h=75, opacity=1.0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="#9FE1CB" stroke="#0F6E56" stroke-width="1" opacity="{opacity}"/>'
        f'<rect x="{x+8}" y="{y+8}" width="{w-16}" height="{h//3}" rx="2" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5" opacity="{opacity}"/>'
        f'<text font-size="7" fill="#085041" x="{x+w//2}" y="{y+h//2+6}" text-anchor="middle" opacity="{opacity}">ATM</text>'
        f'<rect x="{x+w//4}" y="{y+h-14}" width="{w//2}" height="7" rx="1" fill="#0F6E56" opacity="{opacity}"/>'
    )

def _render_panel(ev, px, py, pw, ph, entity_roles, idx):
    entity_id = ev.get("entity_id", "")
    alias     = ev.get("primary_alias", entity_id)
    role      = entity_roles.get(entity_id, "unknown")
    colors    = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
    conf      = float(ev.get("confidence", 0.5))
    tags      = ev.get("action_tags") or []
    if isinstance(tags, str):
        import json
        try: tags = json.loads(tags)
        except: tags = []
    content   = ev.get("content", "")
    ts        = (ev.get("timestamp_str") or ev.get("timestamp", ""))
    ts_label  = ts[11:19] if len(ts) >= 19 else ts
    conflict  = bool(ev.get("conflict_flag", False))
    cx        = px + pw // 2
    mid_y     = py + ph // 2

    parts = []

    # Background
    parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="4" fill="{colors["bg"]}" stroke="{colors["border"]}" stroke-width="1"/>')

    # Ground
    parts.append(f'<line x1="{px}" y1="{py+ph-28}" x2="{px+pw}" y2="{py+ph-28}" stroke="#9FE1CB" stroke-width="1.5"/>')

    # ATM building
    has_atm = any(t in tags for t in ["APPROACH","ENTER","WITHDRAW","TAMPER","EXIT","STEAL","NAVIGATE"])
    if has_atm:
        atm_far = "EXIT" in tags or "FLEE" in tags
        atm_x = px + 8 if atm_far else px + pw - 70
        parts.append(_atm(atm_x, py + ph // 2 - 44, opacity=0.65 if atm_far else 1.0))

    # Person position
    movement = any(t in tags for t in ["APPROACH","ENTER","EXIT","FLEE"])
    person_x = cx - 20 if movement else cx
    parts.append(_person(person_x, mid_y - 14, colors["body"]))

    # Motion arrow
    if movement:
        ax1 = person_x + 22
        ax2 = px + pw - 20 if ("EXIT" in tags or "FLEE" in tags) else px + pw - 78
        if ax2 > ax1:
            parts.append(f'<line x1="{ax1}" y1="{mid_y}" x2="{ax2}" y2="{mid_y}" stroke="#1D9E75" stroke-width="1.5" stroke-dasharray="4 2"/>')
            parts.append(f'<polygon points="{ax2},{mid_y-3} {ax2+7},{mid_y} {ax2},{mid_y+3}" fill="#1D9E75"/>')

    # Speech bubble for content
    if content:
        words  = content.split()
        line1  = " ".join(words[:7])
        line2  = " ".join(words[7:14]) if len(words) > 7 else ""
        bh_bub = 44 if line2 else 30
        bub_y  = py + 10
        parts.append(f'<rect x="{px+10}" y="{bub_y}" width="{pw-20}" height="{bh_bub}" rx="7" fill="white" stroke="{colors["border"]}" stroke-width="0.8"/>')
        parts.append(f'<polygon points="{person_x-4},{bub_y+bh_bub} {person_x+4},{bub_y+bh_bub} {person_x},{bub_y+bh_bub+8}" fill="white" stroke="{colors["border"]}" stroke-width="0.8"/>')
        parts.append(f'<text font-size="9" fill="#3C3489" x="{px+pw//2}" y="{bub_y+12}" text-anchor="middle">{_xml(_truncate(line1, 34))}</text>')
        if line2:
            parts.append(f'<text font-size="9" fill="#3C3489" x="{px+pw//2}" y="{bub_y+26}" text-anchor="middle">{_xml(_truncate(line2, 34))}</text>')

    # Conflict flag
    if conflict:
        parts.append(f'<rect x="{px+4}" y="{py+4}" width="84" height="14" rx="3" fill="#E24B4A"/>')
        parts.append(f'<text font-size="7" fill="white" x="{px+46}" y="{py+13}" text-anchor="middle" dominant-baseline="central">conflict flagged</text>')

    # CCTV badge for video
    if ev.get("modality") == "video":
        parts.append(f'<rect x="{px+pw-46}" y="{py+4}" width="42" height="14" rx="3" fill="#2C2C2A" opacity="0.7"/>')
        parts.append(f'<text font-size="7" fill="white" x="{px+pw-25}" y="{py+13}" text-anchor="middle" dominant-baseline="central">CCTV</text>')

    # Caption
    cap_y = py + ph
    parts.append(f'<rect x="{px}" y="{cap_y}" width="{pw}" height="26" fill="{colors["caption_bg"]}"/>')
    caption_text = _xml(f"{ts_label} · {alias} · {_truncate(content, 28)}")
    parts.append(f'<text font-size="9" fill="{colors["caption_text"]}" x="{px+8}" y="{cap_y+13}" dominant-baseline="central">{caption_text}</text>')
    cc = _conf_color(conf)
    parts.append(f'<rect x="{px+pw-64}" y="{cap_y+3}" width="60" height="20" rx="10" fill="{cc}"/>')
    parts.append(f'<text font-size="8" fill="white" x="{px+pw-34}" y="{cap_y+13}" text-anchor="middle" dominant-baseline="central">conf {conf:.2f}</text>')

    # Panel label
    tag_str = tags[0] if tags else "—"
    parts.append(f'<text font-size="9" fill="{colors["border"]}" x="{px+8}" y="{py+16}" dominant-baseline="central">{idx+1}  {tag_str}</text>')

    return "\n".join(parts)


def generate_comic(case_id: str, tl_version: str = None, output_dir: str = "output/comics") -> str:
    mem = ForenSynthMemory()

    if not tl_version:
        versions = mem.get_latest_versions(case_id)
        tl_version = versions.get("tl_version") or "V1"

    events = mem.get_events(case_id, version=tl_version)
    if not events:
        print(f"[ERROR] No events found for {case_id} {tl_version}")
        return None

    er_row = mem._query_one(
        "SELECT full_json FROM er_runs WHERE case_id=? ORDER BY run_version DESC LIMIT 1",
        (case_id,)
    )
    er = _load(er_row["full_json"]) if er_row else {}
    entity_roles = {}
    for ent in er.get("canonical_entities", []):
        for role in (ent.get("roles") or []):
            entity_roles[ent["entity_id"]] = role

    crit_row = mem._query_one(
        "SELECT overall_score FROM critique_runs WHERE case_id=? ORDER BY generated_at DESC LIMIT 1",
        (case_id,)
    )
    crit_score = float(crit_row["overall_score"]) if crit_row else 0.0

    case_row = mem._query_one("SELECT domain, template FROM cases WHERE case_id=?", (case_id,))
    domain   = (case_row.get("domain") or "") if case_row else ""
    template = (case_row.get("template") or "") if case_row else ""

    causal = mem.get_causal_edges(case_id, version=tl_version)

    # Layout
    PW, PH  = 300, 170
    CAP_H   = 26
    GAP     = 20
    MARGIN  = 20

    rows = []
    i = 0
    while i < len(events):
        if i == len(events) - 1 and len(events) % 2 == 1:
            rows.append([events[i]])
            i += 1
        else:
            rows.append(events[i:i+2])
            i += 2

    header_h = 56
    row_h    = PH + CAP_H + GAP
    total_h  = header_h + len(rows) * row_h + 40 + 50 + 40 + 24 + MARGIN

    avg_conf = sum(float(e.get("confidence", 0)) for e in events) / max(len(events), 1)

    parts = [
        f'<svg width="100%" viewBox="0 0 680 {total_h}" role="img" xmlns="http://www.w3.org/2000/svg">',
        f'<title>ForenSynth forensic comic — {case_id}</title>',
        f'<desc>Panel-by-panel forensic reconstruction for {case_id} generated by ForenSynth.</desc>',
        '<style>text { font-family: "Helvetica Neue", Arial, sans-serif; fill: #2C2C2A; }</style>',
        f'<text font-size="14" font-weight="500" x="340" y="28" text-anchor="middle">{_xml(case_id)} · forensic reconstruction · {_xml(tl_version)}</text>',
        f'<text font-size="12" fill="#5F5E5A" x="340" y="46" text-anchor="middle">{_xml(domain)} · {_xml(template)} · critique {crit_score:.2f} · avg conf {avg_conf:.2f}</text>',
    ]

    y_cursor = header_h
    for row_idx, row_events in enumerate(rows):
        full_width = len(row_events) == 1
        for col_idx, ev in enumerate(row_events):
            global_idx = sum(len(r) for r in rows[:row_idx]) + col_idx
            pw = 640 if full_width else PW
            px = MARGIN if (full_width or col_idx == 0) else MARGIN + PW + GAP
            parts.append(_render_panel(ev, px, y_cursor, pw, PH, entity_roles, global_idx))
        y_cursor += PH + CAP_H + GAP

    # Timeline bar
    tb_y = y_cursor + 10
    parts.append(f'<rect x="{MARGIN}" y="{tb_y+10}" width="640" height="3" rx="1" fill="#D3D1C7"/>')
    if events:
        epochs = [float(e.get("ts_epoch") or 0) for e in events]
        ts_min, ts_max = min(epochs), max(epochs)
        ts_range = max(ts_max - ts_min, 1)
        for ev, ep in zip(events, epochs):
            ratio  = (ep - ts_min) / ts_range
            dot_x  = MARGIN + int(ratio * 640)
            role   = entity_roles.get(ev.get("entity_id",""), "unknown")
            col    = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])["border"]
            conf   = float(ev.get("confidence", 0.5))
            ts_lbl = (ev.get("timestamp_str") or ev.get("timestamp",""))
            ts_lbl = ts_lbl[11:16] if len(ts_lbl) >= 16 else ts_lbl
            parts.append(f'<circle cx="{dot_x}" cy="{tb_y+11}" r="{5 if conf >= 0.8 else 4}" fill="{col}"/>')
            parts.append(f'<text font-size="8" fill="#5F5E5A" x="{dot_x}" y="{tb_y+26}" text-anchor="middle">{_xml(ts_lbl)}</text>')

    y_cursor = tb_y + 40

    # Causal chain
    cc_y = y_cursor + 4
    parts.append(f'<rect x="{MARGIN}" y="{cc_y}" width="640" height="42" rx="6" fill="#F1EFE8" stroke="#D3D1C7" stroke-width="0.5"/>')
    parts.append(f'<text font-size="11" fill="#5F5E5A" x="36" y="{cc_y+14}" dominant-baseline="central">Causal chain:</text>')
    seen_tags = []
    for ev in events:
        tags = ev.get("action_tags") or []
        if isinstance(tags, str):
            import json
            try: tags = json.loads(tags)
            except: tags = []
        for t in tags:
            if t not in seen_tags and t in ACTION_COLORS:
                seen_tags.append(t)

    tag_x = 130
    for ti, tag in enumerate(seen_tags[:7]):
        col = ACTION_COLORS.get(tag, "#888780")
        tw  = len(tag) * 7 + 16
        parts.append(f'<rect x="{tag_x}" y="{cc_y+8}" width="{tw}" height="22" rx="4" fill="{col}"/>')
        parts.append(f'<text font-size="9" fill="white" x="{tag_x+tw//2}" y="{cc_y+21}" text-anchor="middle" dominant-baseline="central">{_xml(tag)}</text>')
        if ti < len(seen_tags) - 1 and tag_x + tw + 30 < 640:
            nx = tag_x + tw + 4
            parts.append(f'<line x1="{nx}" y1="{cc_y+19}" x2="{nx+12}" y2="{cc_y+19}" stroke="#888780" stroke-width="1.5"/>')
            parts.append(f'<polygon points="{nx+12},{cc_y+16} {nx+17},{cc_y+19} {nx+12},{cc_y+22}" fill="#888780"/>')
        tag_x += tw + 22

    llm_c = sum(1 for c in causal if "local LLM:" in c.get("label",""))
    parts.append(f'<text font-size="9" fill="#5F5E5A" x="36" y="{cc_y+35}" dominant-baseline="central">{len(causal)} causal links · {llm_c} LLM-derived · {len(causal)-llm_c} deterministic</text>')

    y_cursor = cc_y + 50

    # Legend
    leg_y = y_cursor
    parts.append(f'<rect x="{MARGIN}" y="{leg_y}" width="640" height="36" rx="6" fill="#F1EFE8" stroke="#D3D1C7" stroke-width="0.5"/>')
    parts.append(f'<circle cx="44" cy="{leg_y+18}" r="6" fill="{ROLE_COLORS["suspect"]["body"]}"/>')
    parts.append(f'<text font-size="10" fill="#5F5E5A" x="54" y="{leg_y+18}" dominant-baseline="central">Suspect</text>')
    parts.append(f'<circle cx="150" cy="{leg_y+18}" r="6" fill="{ROLE_COLORS["witness"]["body"]}"/>')
    parts.append(f'<text font-size="10" fill="#5F5E5A" x="160" y="{leg_y+18}" dominant-baseline="central">Witness</text>')
    parts.append(f'<rect x="256" y="{leg_y+10}" width="64" height="16" rx="3" fill="#E24B4A"/>')
    parts.append(f'<text font-size="8" fill="white" x="288" y="{leg_y+18}" text-anchor="middle" dominant-baseline="central">conflict flag</text>')
    for col, label, lx in [("#1D9E75","conf &gt;= 0.85", 360),("#BA7517","conf &gt;= 0.70", 450),("#993C1D","conf &lt; 0.70", 530)]:
        parts.append(f'<rect x="{lx}" y="{leg_y+11}" width="10" height="14" rx="2" fill="{col}"/>')
        parts.append(f'<text font-size="9" fill="#5F5E5A" x="{lx+14}" y="{leg_y+18}" dominant-baseline="central">{label}</text>')

    y_cursor += 40

    # Stamp
    parts.append(f'<text font-size="9" fill="#888780" x="340" y="{y_cursor+12}" text-anchor="middle">Generated by ForenSynth · Timeline {tl_version} · {len(events)} events · {len(causal)} causal links · conf avg {avg_conf:.2f}</text>')
    parts.append('</svg>')

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    svg_file = out_path / f"{case_id}_comic_{tl_version}.svg"
    with open(svg_file, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Saved → {svg_file}")
    return str(svg_file)


def main():
    p = argparse.ArgumentParser(description="ForenSynth Comic Generator")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", help="Case ID")
    group.add_argument("--all",  action="store_true", help="All cases in DB")
    p.add_argument("--tl-version", default=None)
    p.add_argument("--output",     default="./output/comics")
    args = p.parse_args()

    if args.all:
        mem   = ForenSynthMemory()
        cases = mem.list_cases()
        print(f"Generating comics for {len(cases)} cases...")
        for c in cases:
            generate_comic(c["case_id"], args.tl_version, args.output)
    else:
        generate_comic(args.case, args.tl_version, args.output)


if __name__ == "__main__":
    main()