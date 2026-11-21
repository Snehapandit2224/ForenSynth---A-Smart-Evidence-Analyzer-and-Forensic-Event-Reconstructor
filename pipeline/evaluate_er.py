#!/usr/bin/env python3
"""
ForenSynth - Entity Resolution evaluation

Scores ER directly against the generator's ground truth, separately from
the timeline: for every pair of observations, did ER put them in the same
entity exactly when the ground truth says they are the same person?

  precision  - of the pairs ER merged, how many are really one person.
               A wrong merge fuses two people (e.g. suspect + witness) -
               the expensive error for an investigator.
  recall     - of the pairs that really are one person, how many ER merged.
               A wrong split leaves one person as two entities.
  wrong_merges - which ground-truth people were fused together.

The timeline F1 in evaluate.py can't show this: it resolves each event's
alias through the ground-truth mapping, so it barely notices a wrong merge.

Run on every ground-truth case, both with the generator's roles and with
every role "unknown" - what real field agents produce. The second column
is the one that shows whether ER keeps people apart on real evidence:

    python pipeline/evaluate_er.py                     # both role modes
    python pipeline/evaluate_er.py --roles unknown --legacy
        # --legacy: config require_identity_evidence=False (the old
        # co-occurrence + equal-role merging), for comparison
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _name_tokens(alias: str) -> Optional[set]:
    """Name parts of an alias that is a person's name, else None (labels
    like Person_14, cam_SPEAKER_00, unknown_man_1@doc, sms_74)."""
    tokens = [t for t in re.split(r"[^a-z]+", (alias or "").lower()) if t]
    if not tokens or any(ch.isdigit() for ch in alias or "") or tokens[0] in ("unknown", "email")             or {"speaker", "person"} & set(tokens):
        return None   # agent labels (rec_Speaker_Unknown), not names
    return set(tokens)


def _evidence_supported_pairs(observations: List[Dict[str, Any]], truth: Dict[str, str]) -> set:
    """Same-person observation pairs the EVIDENCE connects - directly or
    through a chain - by: the same label (not an undiarized recording's
    shared one), voice prints on both, appearance on both, or names sharing a
    word. An upper bound on what ER can justify merging: e.g. a suspect's own
    phone call and their CCTV clip share none of these, and no rule should
    merge them. (Appearance and surnames can also mislead - this counts
    them as available evidence, not as proof.)"""
    by_person: Dict[str, List[Dict[str, Any]]] = {}
    for o in observations:
        if truth.get(str(o.get("obs_id"))):
            by_person.setdefault(truth[str(o["obs_id"])], []).append(o)
    supported = set()
    for obs in by_person.values():
        parent = {str(o["obs_id"]): str(o["obs_id"]) for o in obs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in itertools.combinations(obs, 2):
            undiarized = "undiarized" in (a.get("extraction_status"), b.get("extraction_status"))
            names_a, names_b = _name_tokens(a.get("entity", "")), _name_tokens(b.get("entity", ""))
            if undiarized:
                names_a = names_b = None
            linked = (
                (a.get("entity") == b.get("entity")
                 and "undiarized" not in (a.get("extraction_status"), b.get("extraction_status")))
                or (a.get("identity_embedding") and b.get("identity_embedding"))
                or (a.get("appearance") and b.get("appearance"))
                or (names_a and names_b and names_a & names_b)
            )
            if linked:
                parent[find(str(a["obs_id"]))] = find(str(b["obs_id"]))
        for a, b in itertools.combinations(sorted(parent), 2):
            if find(a) == find(b):
                supported.add((a, b))
    return supported


def er_pairwise_metrics(er_result: Dict[str, Any], observations: List[Dict[str, Any]],
                        entity_mapping: Dict[str, str],
                        obs_entity: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Pairwise precision / recall of ER's entities against ground truth.
    obs_entity (obs_id -> person, written by the generator's realism flags)
    is used when given - there an alias alone may name several people (an
    undiarized recording). Otherwise each observation's alias is looked up
    in entity_mapping; observations it doesn't cover are left out."""
    alias_of = {str(o.get("obs_id")): o.get("entity") for o in observations}
    truth = {oid: (obs_entity or {}).get(oid) or entity_mapping.get(alias)
             for oid, alias in alias_of.items()}
    predicted = {str(oid): e["entity_id"]
                 for e in er_result.get("canonical_entities", []) for oid in e.get("sources", [])}
    ids = sorted(o for o in predicted if truth.get(o))
    tp = fp = fn = 0
    wrong_merges, wrong_splits = set(), set()
    for a, b in itertools.combinations(ids, 2):
        same_pred = predicted[a] == predicted[b]
        ta, tb = truth[a], truth[b]
        if same_pred and ta == tb:
            tp += 1
        elif same_pred:
            fp += 1
            wrong_merges.add(tuple(sorted((ta, tb))))
        elif ta == tb:
            fn += 1
            wrong_splits.add(ta)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # recall against what the evidence supports, not against every true link
    supported = _evidence_supported_pairs(
        [o for o in observations if str(o.get("obs_id")) in predicted], truth)
    found = sum(1 for a, b in supported if predicted.get(a) == predicted.get(b))
    return {
        "er_precision": round(precision, 4),
        "er_recall": round(recall, 4),
        "er_f1": round(f1, 4),
        "er_recall_supported": round(found / len(supported), 4) if supported else 1.0,
        "same_person_pairs": tp + fn,
        "evidence_supported_pairs": len(supported),
        "observations_scored": len(ids),
        "entities_predicted": len({predicted[o] for o in ids}),
        "entities_true": len({truth[o] for o in ids}),
        "wrong_merges": [list(p) for p in sorted(wrong_merges)],
        "wrong_splits": sorted(wrong_splits),
    }


def evaluate_er_file(er_path: str, ground_truth_path: str) -> Optional[Dict[str, Any]]:
    """Score a saved ER output (output/er/<case>_er_vN_output.json) against
    a ground-truth case file, which holds the observations and the mapping."""
    try:
        er = json.loads(Path(er_path).read_text(encoding="utf-8"))
        gt = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    g = gt.get("ground_truth", {})
    return er_pairwise_metrics(er, gt.get("observations", []), g.get("entity_mapping", {}),
                               g.get("obs_entity"))


def main() -> None:
    ap = argparse.ArgumentParser(description="ER precision/recall against ground truth")
    ap.add_argument("--cases", default=str(_root / "cases"), help="Folder of ground-truth case files")
    ap.add_argument("--roles", choices=["real", "unknown", "both"], default="both")
    ap.add_argument("--legacy", action="store_true",
                    help="Old merging (require_identity_evidence=False) for comparison")
    ap.add_argument("--llm", action="store_true", help="Let ER use the LLM (default: offline, reproducible)")
    ap.add_argument("--output", help="Write the per-case results to this JSON file")
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    from entity_resolution import resolve_entities

    modes = ["real", "unknown"] if args.roles == "both" else [args.roles]
    config = {"require_identity_evidence": not args.legacy}
    results: Dict[str, Dict[str, Any]] = {}
    gt_files = sorted(p for p in Path(args.cases).glob("*.json")
                      if not p.stem.endswith("_obs_only") and not p.stem.startswith("_"))
    for gt_path in gt_files:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        mapping = gt.get("ground_truth", {}).get("entity_mapping")
        if not mapping:
            continue
        for mode in modes:
            case = {k: v for k, v in gt.items() if k != "ground_truth"}
            case["observations"] = [
                {k: v for k, v in o.items() if k not in ("event_ref", "noise_tags")}
                | ({"role": "unknown"} if mode == "unknown" else {})
                for o in gt["observations"]
            ]
            er = resolve_entities(case, config=config, llm_enabled=args.llm)
            m = er_pairwise_metrics(er, gt["observations"], mapping,
                                    gt["ground_truth"].get("obs_entity"))
            results.setdefault(gt_path.stem, {})[mode] = m
            print(f"{gt_path.stem:<14} roles={mode:<7} P={m['er_precision']:.2f} R={m['er_recall']:.2f} "
                  f"entities {m['entities_predicted']}/{m['entities_true']}"
                  + (f"  wrong merges: {m['wrong_merges']}" if m["wrong_merges"] else ""), flush=True)

    for mode in modes:
        rows = [r[mode] for r in results.values()]
        if rows:
            p = sum(r["er_precision"] for r in rows) / len(rows)
            r_ = sum(r["er_recall"] for r in rows) / len(rows)
            merged = sum(1 for r in rows if r["wrong_merges"])
            print(f"MEAN roles={mode}: precision={p:.3f} recall={r_:.3f}  "
                  f"cases with a wrong merge: {merged}/{len(rows)}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
