"""Locates ground-truth case files and pipeline output artifacts for eval."""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GROUND_TRUTH_DIRS = [
    REPO_ROOT / "GENERATOR_FIXED" / "cases_atm",
    REPO_ROOT / "GENERATOR_FIXED" / "cases_office",
    REPO_ROOT / "cases" / "cases_atm",
    REPO_ROOT / "cases" / "cases_office",
]

OUTPUT_DIR = REPO_ROOT / "output"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_ground_truth_case(case_id):
    """Return the parsed full case JSON (with a ground_truth block) for case_id,
    or None if no case with ground truth is found."""
    for d in GROUND_TRUTH_DIRS:
        p = d / f"{case_id}.json"
        if p.exists():
            data = load_json(p)
            if "ground_truth" in data:
                return data
    return None


def _latest_versioned(dir_path, pattern):
    """Return the path with the highest captured version number matching pattern."""
    if not dir_path.exists():
        return None
    best_version, best_path = -1, None
    for p in dir_path.iterdir():
        m = pattern.match(p.name)
        if m:
            v = int(m.group(1))
            if v > best_version:
                best_version, best_path = v, p
    return best_path


def find_all_er_outputs(case_id):
    """All ER output attempts for case_id, in ascending version order (each a
    parsed JSON dict). Re-runs triggered by the showrunner can fail outright
    (status == 'failed'), so callers that want reconstruction quality should
    use find_latest_successful_er_output instead of just the newest file."""
    d = OUTPUT_DIR / "er"
    pattern = re.compile(rf"^{re.escape(case_id)}_er_v(\d+)_output\.json$")
    versioned = []
    if d.exists():
        for p in d.iterdir():
            m = pattern.match(p.name)
            if m:
                versioned.append((int(m.group(1)), p))
    versioned.sort(key=lambda t: t[0])
    if not versioned:
        fallback = d / f"{case_id}_er_output.json"
        if fallback.exists():
            versioned = [(0, fallback)]
    return [load_json(p) for _, p in versioned]


def find_latest_er_output(case_id):
    """The most recent ER attempt, regardless of whether it succeeded."""
    outputs = find_all_er_outputs(case_id)
    return outputs[-1] if outputs else None


def find_latest_successful_er_output(case_id):
    """The most recent ER attempt whose status isn't 'failed' (for scoring
    reconstruction quality; a crashed run has no entities to score)."""
    for data in reversed(find_all_er_outputs(case_id)):
        if data.get("status") != "failed":
            return data
    return None


def find_latest_timeline_output(case_id):
    d = OUTPUT_DIR / "timelines"
    pattern = re.compile(rf"^{re.escape(case_id)}_timeline_V(\d+)\.json$")
    p = _latest_versioned(d, pattern)
    return load_json(p) if p else None


def find_latest_critique_output(case_id):
    d = OUTPUT_DIR / "critiques"
    pattern = re.compile(rf"^{re.escape(case_id)}_critique_C(\d+)\.json$")
    p = _latest_versioned(d, pattern)
    return load_json(p) if p else None


def find_latest_showrunner_output(case_id):
    d = OUTPUT_DIR / "showrunner"
    pattern = re.compile(rf"^{re.escape(case_id)}_showrunner_C(\d+)\.json$")
    p = _latest_versioned(d, pattern)
    return load_json(p) if p else None


def discover_case_ids():
    """All case_ids that have at least an ER output on disk."""
    d = OUTPUT_DIR / "er"
    ids = set()
    if d.exists():
        for p in d.glob("*_er_*output.json"):
            m = re.match(r"^(CASE_[A-Z0-9]+_\d+)_er_", p.name)
            if m:
                ids.add(m.group(1))
    return sorted(ids)
