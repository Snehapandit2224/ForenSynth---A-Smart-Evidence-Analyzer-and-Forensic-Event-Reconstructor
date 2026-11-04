"""
Store the pipeline's ground-truth scores after an evaluation run.

evaluate.py / evaluate_all.py compute precision, recall, F1, action / temporal / entity accuracy
and confidence calibration, but only write them to output/evaluation/*.json. That folder is
git-ignored and is overwritten by the next run, so scores were never kept. This script reads
those files and appends one row per case (tagged with a shared run id) to:

  * forensynth.db, table evaluation_scores    (queryable history, next to the pipeline runs)
  * eval/scores_history.csv                   (plain-text history that can be committed)

  python pipeline/evaluate_all.py
  python pipeline/store_scores.py                      # defaults: output/evaluation, forensynth.db
  python pipeline/store_scores.py --label "after ER fix" --eval-dir output/evaluation
"""
import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COLUMNS = [   # (column, key in the per-case evaluation JSON)
    ("precision", "precision"),
    ("recall", "event_recall"),
    ("f1_score", "f1_score"),
    ("action_tag_accuracy", "action_tag_accuracy"),
    ("temporal_accuracy_sec", "temporal_accuracy_sec"),
    ("entity_accuracy", "entity_accuracy"),
    ("n_ground_truth_events", "n_ground_truth_events"),
    ("n_pipeline_events", "n_pipeline_events"),
    ("n_matched", "n_matched"),
    ("n_covered", "n_covered"),
    ("n_missed", "n_missed"),
    ("result", "result"),
]
HEADER = ["run_id", "stored_utc", "label", "case_id", *(c for c, _ in COLUMNS), "confidence_correlation"]


def load_rows(eval_dir: Path, run_id: str, stored: str, label: str) -> list[dict]:
    rows = []
    for path in sorted(eval_dir.glob("*_evaluation.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        row = {"run_id": run_id, "stored_utc": stored, "label": label, "case_id": r.get("case_id", path.stem)}
        row.update({col: r.get(key) for col, key in COLUMNS})
        row["confidence_correlation"] = (r.get("confidence_calibration") or {}).get("correlation")
        rows.append(row)
    return rows


def store_db(db_path: Path, rows: list[dict]) -> None:
    with sqlite3.connect(db_path) as conn:
        cols = ", ".join(f"{c} REAL" if c not in ("run_id", "stored_utc", "label", "case_id", "result") else f"{c} TEXT"
                         for c in HEADER)
        conn.execute(f"CREATE TABLE IF NOT EXISTS evaluation_scores (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        conn.executemany(f"INSERT INTO evaluation_scores ({', '.join(HEADER)}) VALUES ({', '.join('?' * len(HEADER))})",
                         [[row[c] for c in HEADER] for row in rows])
        conn.commit()


def store_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=str(ROOT / "output" / "evaluation"),
                    help="folder with <case>_evaluation.json files (evaluate_all.py --output-dir)")
    ap.add_argument("--db", default=str(ROOT / "forensynth.db"))
    ap.add_argument("--csv", default=str(ROOT / "eval" / "scores_history.csv"))
    ap.add_argument("--label", default="", help="optional note for this run, e.g. a commit or change name")
    a = ap.parse_args()

    eval_dir = Path(a.eval_dir)
    stored = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = load_rows(eval_dir, run_id, stored, a.label)
    if not rows:
        raise SystemExit(f"no *_evaluation.json in {eval_dir}: run pipeline/evaluate_all.py first")

    store_db(Path(a.db), rows)
    store_csv(Path(a.csv), rows)

    f1 = [r["f1_score"] for r in rows if r["f1_score"] is not None]
    mean = lambda k: (lambda v: round(sum(v) / len(v), 4) if v else None)([r[k] for r in rows if r[k] is not None])
    print(f"[scores] run {run_id}: {len(rows)} cases  mean precision={mean('precision')}  "
          f"recall={mean('recall')}  F1={mean('f1_score')}  ({len(f1)} with F1)")
    print(f"[scores] -> {a.db} (table evaluation_scores) and {a.csv}")


if __name__ == "__main__":
    main()
