# ForenSynth — Video, Explainability & Evaluation Layer

This branch (`feature/video-explainability-integration`) adds three post-pipeline
modules on top of the existing ER → Timeline → Critique → Showrunner pipeline
(`pipeline/run_case.py`): a scene-reconstruction video generator, a dual-layer
explainability PDF report, and a ground-truth evaluation layer — plus a
data-driven rewrite of the scene planner, deterministic evaluation tooling,
and six bug fixes discovered while building and testing them.

## What's new

### 1. Scene reconstruction video — `agents/scene_reconstruction_v3.py`

```python
generate_scene_video(timeline_dict: dict, output_dir: str, case_id: str) -> str | None
```

Renders an animated floor-plan video (stick figures moving between locations,
with a plain-English description panel) for a case's final timeline. Fully
data-driven from the timeline JSON:

- **Entity colors** are assigned dynamically from a fixed palette, in the order
  entities first appear in the timeline — not hardcoded to specific alias names.
- **Locations** are matched to floor-plan landmarks by keyword where possible,
  and fall back to a rotating set of generic anchor points otherwise, so
  cases with unfamiliar location strings don't all collapse onto one pixel.
- **Scene descriptions** are built from each event's real `primary_alias`,
  `role`, `action_tags`, `location`, and `content` — including a
  "N simultaneous observations" summary when multiple events share a timestamp.
- **Title/summary cards** report the case's actual entity roster, scene count,
  observation count, timeline version, and classification, instead of static text.
- The floor-plan artwork itself (ATM kiosk, entrance, police station) is a
  shared visual "set" reused across every case, since this dataset is
  exclusively ATM-robbery scenarios.
- Wrapped in try/except — returns `None` on failure, never raises.

`agents/scene_planner.py` is a semantic-transform helper alongside it (groups
simultaneous events, classifies event type, flags action/location conflicts).

### 2. Explainability report (PDF) — `agents/explainability_report_v2.py`

```python
generate_explainability_report(pipeline_outputs: dict, output_dir: str, case_id: str) -> str | None
```

Generates a dual-layer PDF (plain-English summary + technical detail per
section) covering Entity Resolution, Timeline, Critique rounds (C1→C3), and
Showrunner decisions. `pipeline_outputs` keys: `er`, `timeline_v3`,
`critique_c1..c3`, `showrunner_c1..c3` (missing rounds render as empty
sections — most cases never reach C3). Font resolution: searches
`C:/Windows/Fonts/` for a usable TTF, falls back to any TTF found, falls back
to fpdf's built-in core font if nothing is found — no hardcoded Linux paths.

### 3. Ground-truth evaluation layer — `pipeline/evaluate.py` + `pipeline/evaluate_all.py`

```python
evaluate_case(case_id: str, ground_truth_path: str, timeline_v3_path: str,
              output_dir: str = "output/evaluation") -> dict
```

Compares a pipeline-reconstructed timeline against the case generator's
ground truth (`GENERATOR_FIXED/cases_atm/{case_id}.json`). Matching algorithm:

1. Each ground-truth event is matched to a pipeline event via **fuzzy entity
   resolution** (through `ground_truth.entity_mapping`, exact match then
   substring fallback) plus nearest timestamp.
2. **Action-tag accuracy** and **entity accuracy** are computed independently
   on top of that match set — entity accuracy re-checks with a *strict*
   (non-fuzzy) resolver, so a pair that only matched via a loose alias
   overlap can still be marked entity-incorrect.
3. Reports: event recall, action-tag accuracy, temporal accuracy (avg
   timestamp delta in matched pairs), entity accuracy, precision, F1, and a
   confidence-calibration correlation (does higher pipeline confidence
   correlate with correct reconstruction?).

`PASS` is F1 ≥ 0.75, `PARTIAL` is F1 ≥ 0.55, else `FAIL`.

`evaluate_all.py` discovers every ground-truth file in
`GENERATOR_FIXED/cases_atm/` dynamically (no hardcoded case IDs), resolves
each case's actual final timeline **by file modification time** (not
version-number priority — see Bug Fixes below), and prints a summary table.

### 4. Pipeline integration — `pipeline/run_case.py`

- Video, report, and evaluation all run automatically after each case
  finishes (`no_action`, `human_review`, or the C3 version ceiling), fed
  directly from in-memory results — nothing is re-read from disk.
- Evaluation only runs if a matching `GENERATOR_FIXED/cases_atm/{case_id}.json`
  ground-truth file exists; otherwise it's skipped silently.
- New CLI flags: `--skip-video`, `--skip-report`, `--skip-eval`.

### 5. Dashboard — `forensynth_dashboard.html`

Added an "Outputs" tab (this dashboard is a fully static/mock demo with no
backend, so this just adds two link cards showing where the video and report
would land: `output/videos/{case_id}_scene_reconstruction_v3.mp4` and
`output/reports/{case_id}_explainability_report_v2.pdf`).

### 6. Data-driven scene planner — `agents/scene_planner.py`

Rewritten to read timeline and critique data directly from `forensynth.db`
(`timeline_runs`, `critique_runs` — latest row per case by `id`) instead of
JSON files on disk, and to resolve case context (currently `ATM` or
`UNKNOWN`) from the `cases.domain` column instead of a hardcoded value.
`classify_event`, `detect_action_issues`, and `resolve_participants` are
unchanged. CLI: `python agents/scene_planner.py --case CASE_ID [--db-path]
[--output-dir]`.

### 7. Deterministic evaluation flags — `pipeline/evaluate_all.py`

Added `--no-llm` (wipes each case's DB rows and re-runs the pipeline with
`GROQ_API_KEY`/`Timeline_Key`/`TIMELINE_LOCAL_LLM_BACKEND` cleared, so
Timeline/ER/Critique take their deterministic fallback path — Showrunner is
unaffected either way, since it makes zero LLM calls) and `--stable-check
--runs N` (re-runs each case N times and reports F1 mean/stdev, to
distinguish genuine pipeline non-determinism from a one-off bad run).

## Bug fixes

1. **`agents/entity_resolution.py`** — `EntityResolutionPipeline` expected
   `human_constraints` as a `HumanConstraints` dataclass (attribute access:
   `self.constraints.must_not_merge`), but `run_case.py`'s `re_run_er` path
   passes a plain `dict` (loaded back out of the DB). This crashed
   internally (`AttributeError: 'dict' object has no attribute
   'must_not_merge'`), was silently caught, and produced a degraded ER v2
   with **0 entities** — which then fed a corrupted, inflated timeline that
   the critique agent still scored as a perfect 1.00. Fixed by normalizing
   a dict into `HumanConstraints` at the point it enters the pipeline.
   Confirmed on `CASE_ATM_007` and `CASE_ATM_011` (both went from
   "failed — 0 entities" to correctly re-resolving their real entity count).

2. **`pipeline/evaluate_all.py`'s timeline resolution** — originally picked
   a case's "final" timeline by version-number priority (V3 > V2 > V1).
   `output/timelines/` carries stale V2/V3 files left over from an earlier,
   unrelated dataset, so a stale higher-version file was silently
   outranking the correct, fresh lower-version file (this produced a flat
   0.0 across every metric for `CASE_ATM_004` — a genuine bug signature,
   not real performance). Fixed to pick by file modification time instead.

3. **`agents/shared.py`'s `extract_action_tags()` fragment bank** —
   `"insertion area"` was listed under `WITHDRAW`, and `"fiddling"` had no
   fragment coverage anywhere, so content describing tampering (e.g.
   "fiddling with the ATM card insertion area") was mistagged `WITHDRAW`
   instead of `TAMPER`. Fixed by moving `"insertion area"` to `TAMPER` and
   adding `"fiddling"`, `"fiddling with"`, `"card slot"`, `"manipulating"`,
   `"messing with"`. (Note: measured to have zero effect on the current
   13-case evaluation set, since that exact phrasing doesn't occur in it —
   see Known Limitations.)

A hybrid fragment+semantic scoring approach for `extract_action_tags()` was
also prototyped and tested with real numbers, then deliberately **not**
adopted — the proposed conflict-detection and per-tag threshold override
were verified not to fire on the motivating example (the embedding model
itself isn't confident enough to disagree with the wrong fragment match),
so only the direct fragment-bank fix (#3 above) was kept.

4. **`agents/entity_resolution.py`'s suspect over-merging** — this is the fix
   the root-cause analysis below identified as the real lever.
   `EntityCoreferenceAgent.heuristic_coreference` was merging two distinct
   suspects into one entity whenever their observations shared enough
   circumstantial signal (timing/location/context), even when one
   observation's own text said something like "a second suspect" or "two
   individuals" — content that should have blocked the merge rather than
   fed it. Fixed by adding `_mentions_multiple_actors()`, checked only when
   `role == "suspect"` on either side (an earlier, role-agnostic version of
   this check broke legitimate witness cross-modal merging, since a
   witness's own statement routinely says "two individuals" *about the
   suspects*, not about the witness). Confirmed on `CASE_ATM_004`: before
   the fix, `entity_1` wrongly absorbed `['Person_50', 'Speaker_X',
   'Person_80']`; after, it correctly splits into `entity_1: ['Person_50',
   'Speaker_X']` and `entity_2: ['Person_80']`, with the legitimate witness
   merge (`['Person_55', 'Speaker_Q']`) still intact.

   **Known remaining gap**: the suppression is per observation-*pair*, not
   per-entity. If suspect A has one observation containing marker language
   ("two suspects") and a *different* observation with no marker phrase,
   the marker-free pair can still merge through Union-Find even though the
   marked pair correctly didn't. Seen on `CASE_ATM_012` (its 50% F1 /
   FAIL result below is a mix of this and a separate, unfixable data gap —
   3 of its 5 ground-truth entities have zero raw observations in the
   generated file). Not fixed on this branch.

5. **`pipeline/evaluate.py`'s recall undercounted legitimately-clustered
   events** — the Timeline agent's clustering (see root cause analysis
   below) merges what the generator recorded as separate ground-truth
   moments into one reconstructed event; the evaluator counted every
   ground-truth event that lost that 1:1 match as a flat miss, even when
   its content was still present, just folded into an event matched to a
   different ground-truth event. Added `_find_covered_events()`, which uses
   each observation's `obs_id → event_ref` ground-truth linkage to check
   whether an "unmatched" ground-truth event's own observations actually
   ended up inside an already-matched pipeline event, and gives it 0.5
   partial credit (`n_covered`) instead of zero. Genuinely-missing events
   (no observations at all, or observations that never made it into any
   matched pipeline event) still score zero, unchanged.

6. **`agents/explainability_report_v2.py`'s "What Happened — Event by
   Event, with Evidence" section rendered with overlapping rows.** Two
   separate bugs, both in `build_narrative_summary()`:
   - The left accent stripe was a fixed 14mm tall regardless of how much
     text actually wrapped next to it (the evidence-source line and quoted
     content routinely wrap to 2–3 lines each). Fixed by measuring real
     wrapped-line counts via `pdf.multi_cell(..., dry_run=True,
     output="LINES")` before drawing the stripe.
   - Separately — and this was the one actually causing visible text
     overlap — `conf_badge()`'s internal `cell()` call snaps fpdf2's
     running cursor back up near the top of the row it draws. The old code
     then advanced with a flat `pdf.ln(6)` from that yanked-back position,
     so row *N+1* started rendering before row *N*'s wrapped text had
     actually finished. Fixed by capturing the real bottom-of-content y
     position before the badge draws, and advancing to
     `max(that, y0 + stripe_h) + 3` afterward. Also replaced `★`/`⚠`/`→`
     (which the Windows-resolved report font doesn't have glyphs for, and
     were logged as "Font ... is missing the following glyphs" warnings on
     every report) with `(Start Here)`/`[!]`/`->` respectively; the em-dash
     `—` was left untouched since it renders fine in that font.

## Requirements

Added to `requirements.txt`: `moviepy>=2.1.2`, `Pillow>=10.0.0`,
`fpdf2>=2.7.0`, `numpy>=1.24.0`.

## Usage

```bash
# Full pipeline for one case, including video/report/eval
python pipeline/run_case.py --input path/to/CASE_obs_only.json

# Skip the expensive extras
python pipeline/run_case.py --input path/to/CASE_obs_only.json --skip-video --skip-report --skip-eval

# Evaluate all discovered GENERATOR_FIXED cases against ground truth
python pipeline/evaluate_all.py

# Deterministic re-run: wipe + re-run every case with Groq disabled, then evaluate
python pipeline/evaluate_all.py --no-llm

# Stability check: re-run each case 5x under --no-llm, report F1 mean/stdev
python pipeline/evaluate_all.py --no-llm --stable-check --runs 5

# Rebuild scene spec for one case directly from forensynth.db (no file re-run needed)
python agents/scene_planner.py --case CASE_ATM_013
```

## Evaluation results (13 GENERATOR_FIXED ATM cases)

The dataset was fully regenerated (`GENERATOR_FIXED/main.py --batch 13`,
confirmed genuinely diverse — see below) after the entity-resolution and
evaluation fixes above, replacing the old 13-case set (which included the
retired `CASE_DEMO_001`). Numbers below are from a fresh
`python pipeline/evaluate_all.py` run against that regenerated set.

| Result | Count |
|---|---|
| PASS (F1 ≥ 0.75) | 11 |
| PARTIAL (F1 ≥ 0.55) | 1 |
| FAIL | 1 |

**Mean F1 across all 13 cases: 0.8142**

| Case ID | Recall | Action Acc | Temporal Acc | Entity Acc | Precision | F1 | Result |
|---|---|---|---|---|---|---|---|
| CASE_ATM_001 | 92% | 60% | 8.0s avg | 100% | 71% | 0.80 | PASS |
| CASE_ATM_002 | 83% | 50% | 12.0s avg | 100% | 50% | 0.62 | PARTIAL |
| CASE_ATM_003 | 100% | 89% | 3.3s avg | 100% | 64% | 0.78 | PASS |
| CASE_ATM_004 | 94% | 29% | 18.0s avg | 100% | 64% | 0.76 | PASS |
| CASE_ATM_005 | 100% | 29% | 7.1s avg | 100% | 64% | 0.78 | PASS |
| CASE_ATM_006 | 92% | 20% | 34.4s avg | 100% | 83% | 0.87 | PASS |
| CASE_ATM_007 | 100% | 43% | 8.6s avg | 100% | 88% | 0.93 | PASS |
| CASE_ATM_008 | 79% | 60% | 33.4s avg | 100% | 100% | 0.88 | PASS |
| CASE_ATM_009 | 100% | 67% | 9.2s avg | 100% | 86% | 0.92 | PASS |
| CASE_ATM_010 | 93% | 83% | 6.0s avg | 100% | 86% | 0.89 | PASS |
| CASE_ATM_011 | 100% | 71% | 5.4s avg | 100% | 78% | 0.88 | PASS |
| CASE_ATM_012 | 50% | 75% | 8.5s avg | 100% | 50% | 0.50 | FAIL |
| CASE_ATM_013 | 93% | 67% | 12.5s avg | 100% | 100% | 0.96 | PASS |

Best-performing: `CASE_ATM_013` (0.96), `CASE_ATM_007` (0.93), `CASE_ATM_009`
(0.92) — video and explainability report have been regenerated for these
three post-fix (`output/videos/`, `output/reports/`, both gitignored).

Entity accuracy is still 100% across every case — see "Why entity accuracy
stays 100%" below for why that's an expected property of the metric, not
evidence that ER has no bugs.

## Root cause analysis — why the results look the way they do

**The old "13 cases are really ~4 templates" finding no longer applies.**
The dataset shown above was regenerated in a single `--batch 13` call
against the *same persistent RNG* the generator seeds once at construction
(verified directly: `ForenSynthGenerator.__init__` seeds `self._rng` once,
`generate_batch()` reuses it across all 13 `generate_case()` calls), which
produces genuinely distinct scenarios — confirmed by inspecting the
generated files (varying suspect counts, templates, and event counts per
case) rather than the near-duplicate metrics the old set showed. The old
dataset's 4-way repetition was concluded to be a one-time artifact of how
those specific static files were originally produced (most likely separate
generator invocations reusing a fixed seed), not a bug in
`generator.py`/`templates.py` itself.

**`CASE_ATM_012` is the one FAIL, and it's two separate, distinguishable
problems, not one:**
- A generator data gap: 3 of its 5 ground-truth entities (`suspect_3`,
  `witness_1`, `witness_2`) have zero raw observations in the generated
  case file at all — nothing recoverable pipeline-side, since there's no
  evidence to reconstruct from.
- The narrower per-pair (not per-entity) gap in the suspect-merge fix
  described in Bug Fixes #4 above — an unmarked observation from a suspect
  can still merge through Union-Find even when a different observation from
  that same suspect correctly triggered the marker-language block.

**Action-tag accuracy still varies case to case (20–89%), but it is no
longer explained by the same "entity merge scrambles the pairing" cause
documented in earlier analysis on the old dataset** — that was root-caused
against `CASE_ATM_004` in the old 4-template set, which no longer exists in
its old form. Re-diagnosing action-tag accuracy against the current,
genuinely-diverse dataset (rather than assuming the old explanation still
applies) is flagged as follow-up work, not carried over here as fact.

**Why entity accuracy stays 100% even where ER clustering has known bugs**
(this was asked directly and is worth stating plainly): `entity_accuracy`
checks each event's `primary_alias` — preserved verbatim from the source
observation — against ground truth, not the pipeline's internal
`entity_id`/cluster label. A clustering mistake (two people merged into one
entity) doesn't corrupt any individual event's alias field, so this
specific check is largely blind to it by construction. The real
manifestation of a merging bug is lost **recall**: when two people's
observations get folded into one synthetic pipeline event, only one alias
survives that event, and the other person's ground-truth event becomes
unmatchable. This is a genuine limitation of the current metric design
(not something silently patched over) — recall and F1 are where a
merging bug actually shows up, not entity accuracy.

**Entity resolution is not fully deterministic run-to-run when Groq is
enabled.** Whether the cloud LLM path or the heuristic fallback fires
depends on API availability/rate limits at the time, and that can shift
clustering decisions between identical re-runs of the same input. Verified
directly with `evaluate_all.py --stable-check --runs N` under `--no-llm`
(stdev 0.0 across repeated runs — fully deterministic once Groq is out of
the loop); a Groq-enabled re-run of the same case is not guaranteed to
reproduce the exact numbers above.

## Known limitations

- **The video's visual content is still partly generic**: entity colors,
  locations, and scene text are dynamic, but the floor-plan artwork itself
  is one fixed ATM-kiosk backdrop (reasonable here since every case in this
  dataset is an ATM robbery, but wouldn't generalize to other domains
  without adding more floor-plan variants).
- **Not fixed on this branch** (found but out of scope for this pass):
  - The per-pair (not per-entity) gap in the suspect-merge suppression —
    see Bug Fixes #4 above and the `CASE_ATM_012` discussion in root cause
    analysis.
  - `pipeline/pipeline` (the older orchestrator, superseded by
    `pipeline/run_case.py`) has two bugs — a wrong keyword argument name
    calling the critique agent, and reading a Showrunner result key
    (`action_taken`) that doesn't exist (the real key is `action`).
  - `agents/timeline_agent.py`'s `TimelineAgent.run()` hardcodes
    `TimelineVersion(version="V1", ...)` internally, so the agent's own
    self-written timeline file is always `_V1.json` regardless of the
    actual revision; `run_case.py` works around this by writing its own
    correctly-versioned copy afterward.
  - `pipeline/run_case.py` prints a `→` (U+2192) character, which raises
    `UnicodeEncodeError` when stdout is captured under a non-UTF-8 codepage
    (e.g. invoked as a subprocess from certain shells). Worked around at
    the invocation level (`PYTHONUTF8=1`) rather than patched in source.
