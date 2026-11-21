#!/usr/bin/env python3
"""
scene_reconstruction_v3.py  —  Clean ForenSynth Scene Video
Simple layout:
  Top 530px  : floor plan + animated stick figures
  Bottom 190px: plain-English scene description
No confidence bars, no status badges, no gap IDs, no conflict banners.
"""
import logging, math, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageSequenceClip

log = logging.getLogger(__name__)

W, H  = 1280, 720
FP_H  = 530   # floor plan height
DESC_H= 190   # description panel height
FPS   = 15

# ── Fonts ─────────────────────────────────────────────────────────────────────
# Windows ships neither DejaVu font by default, so C:/Windows/Fonts is
# searched first for the named files, then for any TTF at all, and if
# nothing is found at all we fall back to PIL's built-in bitmap font.
WIN_FONTS_DIR = Path("C:/Windows/Fonts")

def _find_font(names):
    if WIN_FONTS_DIR.is_dir():
        for name in names:
            p = WIN_FONTS_DIR / name
            if p.exists():
                return str(p)
        any_ttf = sorted(WIN_FONTS_DIR.glob("*.ttf"))
        if any_ttf:
            return str(any_ttf[0])
    return None

MONO = _find_font(["DejaVuSansMono.ttf", "consola.ttf", "cour.ttf"])
BOLD = _find_font(["DejaVuSansMono-Bold.ttf", "consolab.ttf", "courbd.ttf"]) or MONO

def font(path, size):
    if not path:
        return ImageFont.load_default()
    try:    return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

F_SM  = font(MONO, 13)
F_MD  = font(BOLD, 16)
F_LG  = font(BOLD, 22)
F_XL  = font(BOLD, 32)
F_TIT = font(BOLD, 42)

# ── Colours ───────────────────────────────────────────────────────────────────
BG    = (8,  13,  24)
DARK  = (14, 22,  40)
PANEL = (12, 18,  36)
CYAN  = (0,  212, 255)
WHITE = (230,238, 248)
GRAY  = (100,120, 148)
LGRAY = (160,178, 200)

# ── Entity colours ────────────────────────────────────────────────────────────
# Assigned dynamically per case: each unique entity gets the next colour in
# this palette, in the order it first appears in the timeline.
PALETTE = [
    ((210, 50,  50),  (255, 110, 110)),   # red
    ((50,  130, 240), (110, 180, 255)),   # blue
    ((140, 90,  240), (185, 145, 255)),   # violet
    ((220, 155, 30),  (255, 200, 90)),    # amber
    ((30,  170, 140), (90,  225, 195)),   # teal
    ((200, 90,  180), (240, 150, 220)),   # magenta
    ((120, 170, 60),  (175, 220, 110)),   # olive-green
]
FALLBACK_COL = ((90, 100, 115), (150, 165, 185))
UNKNOWN_COL  = ((70, 78, 90), (130, 140, 155))   # unresolved subject (grey, unnamed)

def build_color_map(entity_keys):
    """Assign each unique entity key (first-seen order) the next palette colour."""
    return {key: PALETTE[i % len(PALETTE)] for i, key in enumerate(entity_keys)}

def ecol(alias, color_map):
    return color_map.get(alias, FALLBACK_COL)

# ── Location → pixel (within 1280×530 canvas) ─────────────────────────────────
# Keyword matches line up with landmarks actually drawn in build_floor_plan()
# (the kiosk, entrance, police station, ...). A location string that matches
# none of these is spread across a ring of generic anchors instead of all
# collapsing onto one fallback pixel.
LOC_KEYWORDS = {
    "main road":           (840,  50),
    "atm booth entrance":  (450, 290),
    "atm street frontage": (430, 410),
    "atm booth interior":  (270, 265),
    "atm kiosk":           (270, 265),
    "near atm":            (740, 305),
    "atm vicinity":        (740, 305),
    "outside":             (840,  50),
    "local police":        (1150, 70),
    "police":              (1150, 70),
}
GENERIC_ANCHORS = [(640, 300), (740, 130), (540, 400), (900, 250), (360, 130), (1000, 400)]

def build_location_map(location_keys):
    """Assign each unique location string a pixel anchor: keyword match first,
    then a rotating generic anchor so unmapped locations don't all overlap."""
    location_map, next_generic = {}, 0
    for loc in location_keys:
        s = (loc or "").lower()
        hit = next((px for kw, px in LOC_KEYWORDS.items() if kw in s), None)
        if hit is None:
            hit = GENERIC_ANCHORS[next_generic % len(GENERIC_ANCHORS)]
            next_generic += 1
        location_map[loc] = hit
    return location_map

def loc_px(loc_str, location_map):
    return location_map.get(loc_str, (640, 300))

# ── Floor plan ────────────────────────────────────────────────────────────────
def build_floor_plan():
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Road (right side)
    draw.rectangle([700, 0, W, FP_H], fill=(20, 28, 44))
    for y in range(50, FP_H, 65):
        draw.rectangle([800, y, 830, y+32], fill=(38,50,70))  # road markings

    # Pavement (left / centre)
    draw.rectangle([0, 0, 700, FP_H], fill=(15, 23, 42))
    draw.rectangle([0, 435, 700, 452], fill=(22, 36, 60))     # kerb line

    # ATM kiosk shell
    kx1,ky1,kx2,ky2 = 155, 165, 530, 460
    draw.rectangle([kx1,ky1,kx2,ky2], fill=(18,30,56))
    for side in [(kx1,ky1,kx2,ky1),(kx1,ky1,kx1,ky2),(kx2,ky1,kx2,ky2),(kx1,ky2,kx2,ky2)]:
        draw.line(side, fill=CYAN, width=2)

    # ATM machine inside kiosk
    draw.rectangle([188, 195, 318, 430], fill=(8,15,32))
    # Screen glow
    draw.rectangle([200, 208, 306, 318], fill=(0,25,50))
    draw.rectangle([204, 212, 302, 314], fill=(0,60,100))
    draw.text((218, 248), "ATM", fill=CYAN, font=F_MD)
    # Card slot
    draw.rectangle([224, 338, 284, 350], fill=(18,30,58))
    # Keypad
    draw.rectangle([224, 360, 284, 395], fill=(15,26,50))
    for ky in range(364, 394, 10):
        for kxi in range(228, 282, 14):
            draw.rectangle([kxi,ky,kxi+9,ky+6], fill=(25,40,72))

    # Entrance marker
    draw.line([390,452,530,452], fill=CYAN, width=2)
    draw.text((410, 456), "ENTRANCE", fill=GRAY, font=F_SM)

    # Trees
    for tx,ty in [(570,90),(600,215),(580,360),(635,445)]:
        draw.ellipse([tx-15,ty-15,tx+15,ty+15], fill=(12,38,22))
        draw.ellipse([tx-10,ty-10,tx+10,ty+10], fill=(18,52,28))

    # Police station
    draw.rectangle([1085,22,1210,95], fill=(16,25,46))
    draw.line([1085,22,1210,22], fill=CYAN, width=2)
    draw.text((1092, 30), "POLICE", fill=CYAN, font=F_SM)
    draw.text((1092, 50), "STATION", fill=GRAY, font=F_SM)

    # Subtle grid
    for gx in range(0, W, 80):
        draw.line([gx,0,gx,FP_H], fill=(18,27,46), width=1)
    for gy in range(0, FP_H, 80):
        draw.line([0,gy,W,gy], fill=(18,27,46), width=1)

    # Description panel
    draw.rectangle([0, FP_H, W, H], fill=PANEL)
    draw.line([0, FP_H, W, FP_H], fill=CYAN, width=2)

    return img

# ── Stick figure ──────────────────────────────────────────────────────────────
def draw_figure(draw, cx, cy, fill, hi, action="IDLE", scale=1.0):
    r  = int(11 * scale)
    lh = int(30 * scale)
    lw = max(int(2 * scale), 1)
    act = action.upper()

    # Head
    draw.ellipse([cx-r, cy-lh-2*r, cx+r, cy-lh], fill=fill, outline=hi, width=lw)
    # Torso
    draw.line([cx, cy-lh, cx, cy], fill=fill, width=lw)

    # Arms
    if "TAMPER" in act or "INTERACT" in act or "WITHDRAW" in act:
        draw.line([cx, cy-int(lh*.7), cx-int(20*scale), cy-int(lh*.45)], fill=fill, width=lw)
        draw.line([cx-int(20*scale), cy-int(lh*.45), cx-int(28*scale), cy-int(lh*.72)], fill=fill, width=lw)
        draw.line([cx, cy-int(lh*.7), cx+int(8*scale), cy-int(lh*.5)], fill=fill, width=lw)
    elif "OBSERVE" in act or "WATCH" in act:
        draw.line([cx, cy-int(lh*.7), cx-int(18*scale), cy-int(lh*.9)], fill=fill, width=lw)
        draw.line([cx, cy-int(lh*.7), cx+int(18*scale), cy-int(lh*.9)], fill=fill, width=lw)
    elif any(a in act for a in ["FLEE","EXIT","APPROACH","WALK"]):
        draw.line([cx, cy-int(lh*.7), cx-int(15*scale), cy-int(lh*.42)], fill=fill, width=lw)
        draw.line([cx, cy-int(lh*.7), cx+int(15*scale), cy-int(lh*.5)],  fill=fill, width=lw)
    else:
        draw.line([cx, cy-int(lh*.7), cx-int(13*scale), cy-int(lh*.5)], fill=fill, width=lw)
        draw.line([cx, cy-int(lh*.7), cx+int(13*scale), cy-int(lh*.5)], fill=fill, width=lw)

    # Legs
    if any(a in act for a in ["FLEE","EXIT","APPROACH","WALK"]):
        draw.line([cx, cy, cx-int(14*scale), cy+int(24*scale)], fill=fill, width=lw)
        draw.line([cx, cy, cx+int(12*scale), cy+int(20*scale)], fill=fill, width=lw)
    else:
        draw.line([cx, cy, cx-int(11*scale), cy+int(24*scale)], fill=fill, width=lw)
        draw.line([cx, cy, cx+int(11*scale), cy+int(24*scale)], fill=fill, width=lw)

def draw_name(draw, cx, cy, scale, name, hi):
    f = font(MONO, int(11*scale)) if scale < 1 else F_SM
    bb = draw.textbbox((0,0), name, font=f)
    tw = bb[2]-bb[0]
    draw.text((cx-tw//2, cy+int(28*scale)), name, fill=hi, font=f)

MODALITY_ICON = {"text": "TEXT", "audio": "AUDIO"}

def draw_report_card(draw, x, y, alias, modality, content, hi):
    """Distinct visual treatment for an evidence source (witness statement,
    audio call, text/email report): a bordered card, NOT a stick figure at
    a floor-plan location -- placing this entity as an actor would assert
    it was directly, physically observed there, which a report is not."""
    w, h = 300, 68
    draw.rectangle([x, y, x + w, y + h], fill=(16, 18, 24), outline=hi, width=2)
    icon  = MODALITY_ICON.get(modality, "RPT")
    label = f"[{icon}] {alias.replace('_',' ')} -- REPORTED, not observed"
    draw.text((x + 8, y + 7), label[:44], fill=hi, font=F_SM)
    draw.text((x + 8, y + 29), f'"{(content or "").strip()[:46]}"', fill=LGRAY, font=F_SM)

def draw_dashed_path(draw, start, end, col):
    dx,dy = end[0]-start[0], end[1]-start[1]
    dist  = math.hypot(dx,dy) or 1
    steps = max(int(dist//16),2)
    for i in range(steps):
        if i%2==0:
            t = i/steps
            x=int(start[0]+dx*t); y=int(start[1]+dy*t)
            draw.ellipse([x-2,y-2,x+2,y+2], fill=col)

# ── Description panel ─────────────────────────────────────────────────────────
def draw_desc(img, ev_num, total, timestamp, title_line, detail_line=""):
    draw = ImageDraw.Draw(img)
    py   = FP_H + 12

    # Event counter + timestamp  (right side)
    ts = timestamp.replace("T"," ").replace("Z","")
    draw.text((W-260, py), f"EVENT {ev_num} / {total}", fill=CYAN, font=F_MD)
    draw.text((W-260, py+22), ts, fill=GRAY, font=F_SM)

    # Main description line
    draw.text((24, py+4), title_line[:88], fill=WHITE, font=F_LG)

    # Detail line (smaller, dimmer)
    if detail_line:
        draw.text((24, py+34), detail_line[:110], fill=LGRAY, font=F_SM)

# ── Scene descriptions ───────────────────────────────────────────────────────
# Built from scene_planner.build_scene_specs()'s output, not raw timeline
# events -- so a scene's participants (physically placed) and evidence
# sources (reports, never placed) are already correctly separated by the
# time this module sees them; it never re-derives that distinction itself.
def _named(p):
    """A participant is 'named' if scene_planner resolved a real alias for
    it; an extracted-but-unidentified subject (UNKNOWN_SUBJECT_1, from a
    report like "two men fled") has no primary_alias."""
    return bool(p.get("primary_alias"))

def _p_key(p, scene_id):
    """Stable colour/position key. Unresolved subjects are suffixed by
    scene_id so two unrelated anonymous people from different reports never
    get spuriously drawn as the same person moving between scenes."""
    base = p.get("primary_alias") or p.get("entity_id") or "Unknown"
    return base if _named(p) else f"{base}::{scene_id}"

def describe_scene_spec(sc):
    """Return (title_line, detail_line) for one scene_planner scene."""
    participants = sc.get("participants", [])
    evidence     = sc.get("evidence_sources", [])
    is_group     = len(sc.get("event_ids", [])) > 1

    if not is_group:
        if participants:
            p      = participants[0]
            alias  = (p.get("primary_alias") or p.get("entity_id", "?")).replace("_", " ")
            role   = p.get("entity_role") or p.get("role", "unknown")
            action = p.get("action", "OBSERVED")
            loc    = p.get("location") or sc.get("location") or "an unspecified location"
            label  = "Unknown subject" if not _named(p) else alias
            title  = f"{label} ({role}) — {action.title()} at {loc}."
            detail = sc.get("raw_content", "") or ""
        elif evidence:
            e      = evidence[0]
            alias  = e.get("source_id", "?").replace("_", " ")
            title  = f"{alias} ({e.get('role','reporter')}) filed a report — not directly observed."
            detail = e.get("content", "") or ""
        else:
            title, detail = "Unclassified scene.", ""
        if sc.get("location_conflicts"):
            detail = (detail + "  [LOCATION CONFLICT FLAGGED]").strip()
        return title, detail

    names = [(p.get("primary_alias") or p.get("entity_id", "?")).replace("_", " ")
             for p in participants]
    names += [e.get("source_id", "?").replace("_", " ") for e in evidence]
    title  = f"{len(names)} simultaneous item(s): {', '.join(names) or 'none'}."
    detail = "  |  ".join((e.get("content", "") or "")[:60] for e in evidence[:3])
    if not detail:
        detail = "  |  ".join(p.get("action", "") for p in participants[:3])
    return title, detail

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE / SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def render_title(case_id, stats):
    img  = Image.new("RGB", (W,H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0,0,W,5], fill=CYAN)
    draw.rectangle([0,H-5,W,H], fill=CYAN)
    draw.text((60,100), "FORENSYNTH", fill=GRAY, font=F_LG)
    draw.text((60,140), "FORENSIC SCENE RECONSTRUCTION", fill=CYAN, font=F_TIT)
    draw.text((60,210), case_id, fill=WHITE, font=F_XL)
    draw.line([60,260,W-60,260], fill=DARK, width=1)
    draw.text((60,285),
        f"{stats['n_scenes']} scenes reconstructed from {stats['n_obs']} observations", fill=LGRAY, font=F_MD)
    draw.text((60,335), stats["entity_line"], fill=GRAY, font=F_SM)
    draw.text((60,365),
        f"Pipeline: Entity Resolution  ->  Timeline {stats['tl_version']}  ->  Critique  ->  Showrunner",
        fill=GRAY, font=F_SM)
    return [np.array(img)] * int(3.5 * FPS)

def render_summary(case_id, stats):
    img  = Image.new("RGB", (W,H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0,0,W,5], fill=CYAN)
    draw.text((60,80),  "RECONSTRUCTION COMPLETE", fill=CYAN, font=F_TIT)
    draw.text((60,145), case_id, fill=WHITE, font=F_XL)
    draw.text((60,200), f"Classification: {stats['classification']}", fill=(255,100,100), font=F_LG)
    draw.line([60,245,W-60,245], fill=DARK, width=1)
    draw.text((60,270), f"{stats['n_scenes']} scenes reconstructed from {stats['n_obs']} raw pipeline events.", fill=LGRAY, font=F_MD)
    draw.text((60,305), stats["unresolved_line"], fill=GRAY, font=F_SM)
    draw.text((60,335), f"{stats['n_simultaneous']} simultaneous-observation scene(s) consolidated.", fill=GRAY, font=F_SM)
    draw.text((60,420), "This video is a forensic working hypothesis — not confirmed evidence.", fill=GRAY, font=F_SM)
    return [np.array(img)] * int(4.0 * FPS)

# ═══════════════════════════════════════════════════════════════════════════════
# SCENE RENDERER
# ═══════════════════════════════════════════════════════════════════════════════
def render_event(base_fp, ev_num, total, timestamp, actors, title, detail, prev_pos,
                  color_map, location_map, report_entries=None):
    """
    Animate actors moving from prev_pos to their new position,
    then hold for 3s.  Returns list of numpy frames.

    actors: list of (key, action, loc_str, scale, unresolved) -- "key" is
    the colour/position identity (see _p_key); "unresolved" marks a
    subject scene_planner extracted from a report's text but could not
    identify, drawn grey/unnamed rather than with a real assigned colour.

    report_entries: list of (alias, modality, content) for this
    timestamp's evidence sources -- entities who produced a piece of
    evidence but were never physically placed (see draw_report_card).
    """
    report_entries = report_entries or []
    N_MOVE = 10
    N_HOLD = int(3.5 * FPS)
    frames = []

    # Compute target positions
    targets = {}
    for i,(key, action, loc_str, scale, unresolved) in enumerate(actors):
        bx,by = loc_px(loc_str, location_map)
        # spread multiple actors sideways
        bx += (i - (len(actors)-1)/2) * 65
        by  = min(by, FP_H - 80)
        targets[key] = (int(bx), int(by), action, scale, unresolved)

    for fi in range(N_MOVE + N_HOLD):
        img  = base_fp.copy()
        draw = ImageDraw.Draw(img)
        t    = min(fi / max(N_MOVE-1,1), 1.0)

        for key,(tx,ty,action,scale,unresolved) in targets.items():
            fill, hi = UNKNOWN_COL if unresolved else ecol(key, color_map)
            # Interpolate from previous position if we have one
            if key in prev_pos and fi < N_MOVE:
                px0,py0 = prev_pos[key]
                cx = int(px0 + (tx-px0)*t)
                cy = int(py0 + (ty-py0)*t)
            else:
                cx, cy = tx, ty

            # Draw dotted trail on last hold frame
            if fi == N_MOVE + N_HOLD - 1 and key in prev_pos:
                draw_dashed_path(draw, prev_pos[key], (tx,ty), tuple(int(c*0.5) for c in hi))

            draw_figure(draw, cx, cy, fill, hi, action, scale)
            name = "Unknown (reported)" if unresolved else key.replace("_"," ").title()
            draw_name(draw, cx, cy, scale, name, hi)

        # Evidence sources at this timestamp: drawn as cards, never as
        # floor-plan actors -- a card does not assert "observed here".
        for ci, (r_alias, r_modality, r_content) in enumerate(report_entries):
            _, r_hi = ecol(r_alias, color_map)
            draw_report_card(draw, 16, FP_H - 90 - ci * 78, r_alias, r_modality, r_content, r_hi)

        # Label the simultaneous badge for event 5
        if len(actors) > 1:
            draw.rectangle([8,8,340,30], fill=(10,16,32))
            draw.line([8,8,340,8], fill=CYAN, width=1)
            draw.line([8,30,340,30], fill=CYAN, width=1)
            draw.text((14,12), f"SIMULTANEOUS  |  {timestamp.replace('T',' ').replace('Z','')}", fill=CYAN, font=F_SM)

        draw_desc(img, ev_num, total, timestamp, title, detail)
        frames.append(np.array(img))

    prev_pos.update({a:(targets[a][0],targets[a][1]) for a in targets})
    return frames

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def generate_scene_video(scene_spec: dict, output_dir: str, case_id: str):
    """
    Render the forensic scene-reconstruction video for one case and write
    it to `output_dir`. Returns the written path, or None if generation
    failed for any reason (never raises).

    scene_spec: the dict returned by scene_planner.build_scene_specs(), NOT
    a raw timeline dict -- physically-observed participants and
    report-only evidence sources are already separated by that point, so
    this module never has to re-derive the physical-vs-reported
    distinction itself (see classify_visualizability... i.e.
    scene_planner.classify_event).
    """
    try:
        scenes = scene_spec.get("scenes", [])

        # Dynamic colour map: every named participant AND every evidence
        # source gets a stable colour (first-seen order), so a witness's
        # report card and that same witness's own resolved sighting (if
        # any) read as the same person.
        entity_keys = list(dict.fromkeys(
            [_p_key(p, sc["scene_id"]) for sc in scenes for p in sc.get("participants", [])]
            + [e.get("source_id", "?") for sc in scenes for e in sc.get("evidence_sources", [])]
        ))
        color_map = build_color_map(entity_keys)

        location_keys = list(dict.fromkeys(
            p.get("location", "?") for sc in scenes for p in sc.get("participants", [])
        ))
        location_map = build_location_map(location_keys)

        # Aggregate stats for the title/summary cards
        n_events = sum(len(sc.get("event_ids", [])) for sc in scenes) or len(scenes)
        roles = {}
        for sc in scenes:
            for p in sc.get("participants", []):
                label = "Unknown subject" if not _named(p) else (p.get("primary_alias") or "?").replace("_", " ")
                roles.setdefault(p.get("entity_role") or p.get("role", "unknown"), set()).add(label)
        entity_line = "  |  ".join(
            f"{', '.join(sorted(names))} ({role})" for role, names in roles.items()
        ) or "No participants resolved."
        unresolved = scene_spec.get("unresolved_entities", []) or []
        stats = {
            "n_scenes":       len(scenes),
            "n_obs":          n_events,
            "entity_line":    entity_line,
            "tl_version":     scene_spec.get("tl_version", "V?"),
            "classification": scene_spec.get("classification", "UNKNOWN"),
            "unresolved_line": (
                f"{len(unresolved)} entit{'y' if len(unresolved) == 1 else 'ies'} remain unresolved "
                "pending identity verification." if unresolved
                else "All entities resolved with high confidence."
            ),
            "n_simultaneous": sum(1 for sc in scenes if sc.get("scene_type") == "simultaneous_group"),
        }

        base_fp  = build_floor_plan()
        prev_pos = {}
        frames   = []

        log.info("Rendering title card for %s", case_id)
        frames.extend(render_title(case_id, stats))

        total = len(scenes)
        for i, sc in enumerate(scenes):
            ts           = sc.get("timestamp", "")
            participants = sc.get("participants", [])
            evidence     = sc.get("evidence_sources", [])

            actors = [
                (_p_key(p, sc["scene_id"]), p.get("action", "OBSERVE"), p.get("location", "?"),
                 1.0 if len(participants) == 1 else 0.9, not _named(p))
                for p in participants
            ]
            report_entries = [
                (e.get("source_id", "?"), e.get("modality", "?"), e.get("content", ""))
                for e in evidence
            ]
            title, detail = describe_scene_spec(sc)
            log.info("Rendering scene %d/%d [%s] (%d physical, %d reported)",
                      i + 1, total, ts, len(participants), len(evidence))
            frames.extend(render_event(base_fp, i + 1, total, ts, actors, title, detail,
                                        prev_pos, color_map, location_map,
                                        report_entries=report_entries))

        log.info("Rendering summary card for %s", case_id)
        frames.extend(render_summary(case_id, stats))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{case_id}_scene_reconstruction_v3.mp4"

        log.info("Encoding %d frames -> %s", len(frames), out_path)
        clip = ImageSequenceClip(frames, fps=FPS)
        clip.write_videofile(str(out_path), codec="libx264", audio=False,
                             ffmpeg_params=["-crf","18","-preset","fast"], logger=None)
        sz  = Path(out_path).stat().st_size / 1024
        dur = len(frames) / FPS
        log.info("Scene video done: %s (%.1fs, %.0fKB)", out_path, dur, sz)
        return str(out_path)
    except Exception:
        log.exception("Scene video generation failed for %s", case_id)
        return None


if __name__ == "__main__":
    import argparse
    from scene_planner import build_scene_specs, DEFAULT_DB_PATH

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(
        description="Standalone test: render a scene video for a case in forensynth.db."
    )
    ap.add_argument("--case", required=True, help="Case ID, e.g. CASE_ATM_009")
    ap.add_argument("--db-path", default=None, help="Override the forensynth.db path")
    ap.add_argument("--output-dir", default="./output/videos")
    args = ap.parse_args()

    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH
    spec = build_scene_specs(args.case, db_path)
    result = generate_scene_video(spec, args.output_dir, args.case)
    print(f"Video written -> {result}" if result else "Video generation FAILED.")