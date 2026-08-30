# ForenSynth — Video, Explainability & Evaluation Layer

This branch (`feature/video-explainability-integration`) adds three post-pipeline
modules on top of the existing ER → Timeline → Critique → Showrunner pipeline
(`pipeline/run_case.py`): a scene-reconstruction video generator, a dual-layer
explainability PDF report, and a ground-truth evaluation layer — plus three bug
fixes discovered while building and testing them.

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
```

## Evaluation results (13 GENERATOR_FIXED ATM cases)

| Result | Count |
|---|---|
| PASS (F1 ≥ 0.75) | 3 |
| PARTIAL (F1 ≥ 0.55) | 9 |
| FAIL | 1 |

| Case ID | Recall | Action Acc | Temporal Acc | Entity Acc | Precision | F1 | Result |
|---|---|---|---|---|---|---|---|
| CASE_ATM_001 | 83% | 60% | 8.0s avg | 100% | 71% | 0.77 | PASS |
| CASE_ATM_002 | 67% | 50% | 12.0s avg | 100% | 50% | 0.57 | PARTIAL |
| CASE_ATM_003 | 67% | 100% | 3.0s avg | 100% | 55% | 0.60 | PARTIAL |
| CASE_ATM_004 | 62% | 40% | 5.0s avg | 100% | 50% | 0.56 | PARTIAL |
| CASE_ATM_005 | 83% | 60% | 8.0s avg | 100% | 71% | 0.77 | PASS |
| CASE_ATM_006 | 67% | 50% | 12.0s avg | 100% | 50% | 0.57 | PARTIAL |
| CASE_ATM_007 | 67% | 100% | 3.0s avg | 100% | 55% | 0.60 | PARTIAL |
| CASE_ATM_008 | 62% | 40% | 5.0s avg | 100% | 50% | 0.56 | PARTIAL |
| CASE_ATM_009 | 83% | 60% | 8.0s avg | 100% | 71% | 0.77 | PASS |
| CASE_ATM_010 | 67% | 50% | 12.0s avg | 100% | 50% | 0.57 | PARTIAL |
| CASE_ATM_011 | 67% | 100% | 3.0s avg | 100% | 55% | 0.60 | PARTIAL |
| CASE_ATM_012 | 62% | 40% | 5.0s avg | 100% | 50% | 0.56 | PARTIAL |
| CASE_DEMO_001 | 64% | 100% | 3.1s avg | 100% | 70% | 0.67 | PARTIAL |

Entity accuracy is 100% across every case (the fuzzy-vs-strict entity
resolver never actually diverged on this dataset); the real bottleneck is
event recall (62–83%) and action-tag accuracy (40–100%, bimodal — see below).

## Root cause analysis — why the results look the way they do

**The 13 cases are really only ~4 distinct scenarios.** Look closely at the
table above: `{001, 005, 009}`, `{002, 006, 010}`, `{003, 007, 011}`, and
`{004, 008, 012}` each report *identical* metrics down to the decimal, and
their generated videos are frame-for-frame identical too. `GENERATOR_FIXED`'s
case generator appears to cycle through the same underlying template every
4 case IDs rather than producing 12 unique scenarios. This means "3 PASS, 9
PARTIAL, 1 FAIL" is really "3 pass/fail outcomes across ~4 distinct
scenarios, each counted 3 times" — worth knowing before treating these
counts as evidence of broad pipeline coverage.

**Recall is capped at 62–83%, never higher, for a structural reason:**
ground truth always contains more fine-grained events than the pipeline
reconstructs. The Timeline agent clusters raw observations into events by
entity + temporal proximity, which legitimately merges what the generator
recorded as two separate moments (e.g. "tamper" then "exit" from the same
observation) into one reconstructed event. Every case has at least one
ground-truth event that has no dedicated pipeline event to match against —
that's a real recall ceiling from the clustering step, not noise.

**Action-tag accuracy is bimodal — 40–60% in three of the four clusters,
100% in the fourth — and the reason is not tagging quality, it's entity
resolution.** Investigated `CASE_ATM_004` (from the `{004,008,012}` cluster,
40% action accuracy) event by event: every "wrong" tag was actually correct
for its own content —

| Ground truth expects | Pipeline event's real content | Tag pipeline gave | Tag correct for that content? |
|---|---|---|---|
| `withdraw_cash` | "Both persons leave the ATM location at pace" | `EXIT` | Yes |
| `exit_atm` | "Card's in, processing." | `WITHDRAW` | Yes |

The tags are right. What's wrong is *which ground-truth event got compared
to which pipeline event*. Entity Resolution merged two distinct
ground-truth suspects (`suspect_1`, `suspect_2`) into a single pipeline
entity, so the evaluator's entity-based matcher — which pairs a ground-truth
event to the nearest-in-time pipeline event *for the same entity* — ends up
pairing `suspect_1`'s real "withdraw" moment against `suspect_2`'s real
"exit" event, because the pipeline no longer distinguishes them as two
people. The `{003,007,011}` and `DEMO_001` clusters score 100% action
accuracy because their entity resolution happened not to merge anyone in
that run, so events line up with the right person and the tags — which were
never the problem — read as fully correct.

This is the real lever for improving both recall and action-tag accuracy on
this dataset: **fix the suspect-merging behavior in `agents/entity_resolution.py`**,
not `extract_action_tags()`. That fragment-bank fix (the WITHDRAW/TAMPER
one, above) was correct and is kept, but it measurably changed nothing on
this specific dataset, because the phrasing it targets ("fiddling",
"insertion area") doesn't occur in `GENERATOR_FIXED`'s observation content
at all — confirmed by grepping every case file for it. The bug it fixed was
only ever observed against a stale timeline built from a different, older
dataset earlier in development.

**Temporal accuracy also splits by cluster (3s / 5s / 8s / 12s avg), for
the same underlying reason as recall**: it's an average over whichever
pairs the entity-based matcher manages to form, so a cluster with more
successful entity-based matches (denser matched-pair set) naturally
averages over more (and often closer) timestamp pairs than one where fewer,
more scattered pairs got matched.

**Entity resolution is not fully deterministic run-to-run.** Re-running the
same case can shift these numbers even with no code changes: whether the
Groq LLM path or the heuristic fallback fires depends on API
availability/rate limits at the time, and that can change clustering
decisions. This was observed directly — `CASE_ATM_003` scored 100% recall
in one run and 67% in a later re-run of the identical input, purely from
this variance.

## Known limitations

- **The video's visual content is still partly generic**: entity colors,
  locations, and scene text are dynamic, but the floor-plan artwork itself
  is one fixed ATM-kiosk backdrop (reasonable here since every case in this
  dataset is an ATM robbery, but wouldn't generalize to other domains
  without adding more floor-plan variants).
- **Not fixed on this branch** (found but out of scope for this pass):
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
