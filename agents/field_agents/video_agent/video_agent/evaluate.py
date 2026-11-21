"""
Compare extracted video observations with the case's *_obs_only.json (observation fields only,
never ground_truth).

  python -m video_agent.evaluate --cases cases/cases_atm --runs runs
  python -m video_agent.evaluate --cases cases/cases_atm --runs runs --calibrate

Per field: source, location, role, action (tagged from content), timestamp / time_offset error,
entity consistency (one-to-one label mapping), confidence error. --calibrate fits a linear map
from extracted to reference confidence and saves runs/calibration.json, which the agent applies
on the next run.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import config as C
from .agent import parse_ts

VOCAB = C.load_vocabulary()
SENTENCE_TO_ACTION = {s.lower(): a for a, v in VOCAB.items() for s in v["templates"]}
QUALITY = {s.lower() for s in C.VIDEO_QUALITY_BANK.values()}

RULES = [   # order matters: witness phrasing first, since it often mentions exits
    ("report_incident", r"officer|police|security personnel|on (the )?phone|typing on phone|writing|report"),
    ("observe_exit", r"bystander|passerby|eyewitness|onlooker|notices|watch(es|ing) .*(leave|depart|exit)|sees someone|stares"),
    ("tamper_atm", r"tamper|fiddl|card insertion|card slot|card reader|attach|device"),
    ("withdraw_cash", r"transaction|operat|pressing keys|interacts with|conducting activity|cash machine|atm (panel|screen|terminal)"),
    ("flee_scene", r"\bran\b|\brun|rush|hurr|brisk|flee|sprint|quickly"),
    ("wait_outside", r"linger|wait|remains|stands guard|lookout|threshold|watching the area|outside the booth|stands near"),
    ("loiter_near_atm", r"loiter|pac(es|ing)|hang(s|ing) around"),
    ("exit_atm", r"exit|leav|walks out|depart|left the"),
    ("enter_atm", r"enter|steps (into|inside)|pushes|pulls open|proceeds in|crossing the threshold|inside the atm"),
    ("approach_atm", r"approach|nearing|walk(s|ing)? toward|heading|arriv|moving toward|in the direction"),
]


def action_of(content):
    c = re.sub(r"\s*\((captured on footage|captured on camera|on footage|cctv footage)\)", "", (content or "")).strip().lower()
    if c in SENTENCE_TO_ACTION:
        return SENTENCE_TO_ACTION[c]
    if c in QUALITY:
        return "footage_issue"
    if re.match(r"^i('m| am| was| did| never| had| have| only| can|\b)", c):
        return "denial"
    for action, pattern in RULES:
        if re.search(pattern, c):
            return action
    return "unknown"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pair(ref_obs, pred_obs):
    """
    Pass 1: clips named after an obs_id (O4.mp4 -> O4, O4b ...) are matched to that observation.
    Pass 2: remaining reference observations are matched by nearest timestamp (<= 60 s), but only
            to extracted observations whose clip was not named after some other observation.
    """
    base = lambda p: re.match(r"^(O\d+|V\d+)", p["obs_id"]).group(1)
    ref_ids = {r["obs_id"] for r in ref_obs}
    used, result = set(), {}

    def best_of(r, cands):
        ra = action_of(r["content"])
        return max(cands, key=lambda p: (p["role"] == r["role"], action_of(p["content"]) == ra,
                                         -abs(p["time_offset"] - r["time_offset"]) if p["time_offset"] is not None else -1e9))

    for r in ref_obs:
        cands = [p for p in pred_obs if base(p) == r["obs_id"]]
        if cands:
            b = best_of(r, cands)
            used.add(id(b))
            result[r["obs_id"]] = b
    for r in ref_obs:
        if r["obs_id"] in result:
            continue
        rt = parse_ts(r["timestamp"])
        cands = [p for p in pred_obs if id(p) not in used and base(p) not in ref_ids and p["timestamp"] and rt
                 and abs((parse_ts(p["timestamp"]) - rt).total_seconds()) <= 60]
        if cands:
            b = best_of(r, cands)
            used.add(id(b))
            result[r["obs_id"]] = b
    return [(r, result.get(r["obs_id"])) for r in ref_obs]


def evaluate_case(ref_path, pred_path):
    ref = load(ref_path)
    pred = load(pred_path)
    ref_video = [o for o in ref["observations"] if o["modality"] == "video"]
    pairs = pair(ref_video, pred["observations"])
    matched = [(r, p) for r, p in pairs if p]

    mapping = {}
    votes = Counter((p["entity"], r["entity"]) for r, p in matched)
    taken = set()
    for (pe, re_), _ in votes.most_common():
        if pe not in mapping and re_ not in taken:
            mapping[pe] = re_
            taken.add(re_)

    rows = []
    for r, p in pairs:
        row = {"ref_obs": r["obs_id"], "ref_content": r["content"], "ref_action": action_of(r["content"])}
        if p is None:
            row["status"] = "missing (no clip or nothing extracted)"
        else:
            dt = abs((parse_ts(p["timestamp"]) - parse_ts(r["timestamp"])).total_seconds()) if p["timestamp"] else None
            row.update({
                "pred_obs": p["obs_id"], "pred_content": p["content"], "pred_action": action_of(p["content"]),
                "source_ok": p["source"] == r["source"], "location_ok": p["location"] == r["location"],
                "role_ok": p["role"] == r["role"], "entity_ok": mapping.get(p["entity"]) == r["entity"],
                "timestamp_err_s": dt,
                "time_offset_err_s": abs(p["time_offset"] - r["time_offset"]) if p["time_offset"] is not None else None,
                "confidence_ref": r["confidence"], "confidence_pred": p["confidence"],
            })
            row["action_ok"] = row["pred_action"] == row["ref_action"]
            row["status"] = "matched"
        rows.append(row)

    extra = [p["obs_id"] for p in pred["observations"] if all(p is not q for _, q in matched)]
    return {"case_id": ref["case_id"], "rows": rows, "entity_mapping": mapping, "extra_pred_obs": extra}


def summarize(results):
    rows = [r for res in results for r in res["rows"]]
    m = [r for r in rows if r["status"] == "matched"]

    def rate(k):
        return round(sum(r[k] for r in m) / len(m), 3) if m else None

    def mean(k):
        v = [r[k] for r in m if r.get(k) is not None]
        return round(sum(v) / len(v), 2) if v else None

    conf = [(r["confidence_pred"], r["confidence_ref"]) for r in m]
    return {
        "reference_video_obs": len(rows), "matched": len(m),
        "unrendered_or_missing": [f"{res['case_id']}:{r['ref_obs']}" for res in results for r in res["rows"] if r["status"] != "matched"],
        "extra_extracted_obs": sum(len(res["extra_pred_obs"]) for res in results),
        "source_acc": rate("source_ok"), "location_acc": rate("location_ok"), "role_acc": rate("role_ok"),
        "action_acc": rate("action_ok"), "entity_consistency": rate("entity_ok"),
        "timestamp_mae_s": mean("timestamp_err_s"), "time_offset_mae_s": mean("time_offset_err_s"),
        "confidence_mae": round(sum(abs(a - b) for a, b in conf) / len(conf), 3) if conf else None,
        "confidence_pairs": conf,
    }


def fit_calibration(pairs, existing):
    n = len(pairs)
    if n < 3:
        raise SystemExit("need at least 3 matched observations to calibrate")
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    var = sum((x - mx) ** 2 for x, _ in pairs)
    a2 = sum((x - mx) * (y - my) for x, y in pairs) / var if var else 1.0
    b2 = my - a2 * mx
    a0, b0 = (existing["a"], existing["b"]) if existing else (1.0, 0.0)
    return {"a": round(a2 * a0, 4), "b": round(a2 * b0 + b2, 4), "fitted_on": n}   # compose with any previous map


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="folder with *_obs_only.json files")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    results = []
    for pred in sorted(Path(a.runs).glob("*/*_video_obs.json")):
        case_id = pred.parent.name
        ref = Path(a.cases) / f"{case_id}_obs_only.json"
        if not ref.exists():
            print(f"[eval] {case_id}: no {ref.name}, skipped")
            continue
        res = evaluate_case(ref, pred)
        results.append(res)
        print(f"\n## {case_id}   entity mapping: {res['entity_mapping']}")
        for r in res["rows"]:
            if r["status"] != "matched":
                print(f"  {r['ref_obs']:<4} MISSING   ref: {r['ref_content']}")
                continue
            ok = lambda b: "ok " if b else "XX "
            print(f"  {r['ref_obs']:<4}<-{r['pred_obs']:<5} src {ok(r['source_ok'])}role {ok(r['role_ok'])}"
                  f"act {ok(r['action_ok'])}ent {ok(r['entity_ok'])}dt {r['timestamp_err_s']}s  "
                  f"conf {r['confidence_pred']} vs {r['confidence_ref']}")
            if not r["action_ok"]:
                print(f"        ref : [{r['ref_action']}] {r['ref_content']}\n        pred: [{r['pred_action']}] {r['pred_content']}")
        if res["extra_pred_obs"]:
            print(f"  extra extracted (not in case file): {res['extra_pred_obs']}")

    if not results:
        raise SystemExit("no runs to evaluate")
    summary = summarize(results)
    pairs = summary.pop("confidence_pairs")
    print("\n## SUMMARY")
    for k, v in summary.items():
        print(f"  {k:<24} {v}")
    out = Path(a.runs) / "evaluation.json"
    out.write_text(json.dumps({"summary": summary, "cases": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval] details -> {out}")

    if a.calibrate:
        cal_path = Path(a.runs) / "calibration.json"
        existing = load(cal_path) if cal_path.exists() else None
        cal = fit_calibration(pairs, existing)
        cal_path.write_text(json.dumps(cal, indent=2))
        print(f"[eval] calibration {cal} -> {cal_path} (applied on the next agent run; "
              f"cached model answers are reused and only confidence is recomputed)")


if __name__ == "__main__":
    main()
