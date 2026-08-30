"""End-to-end evaluation harness for the ForenSynth pipeline.

Compares each stage's output (output/er, output/timelines, output/critiques,
output/showrunner) against the synthetic generator's ground truth and prints
per-case plus aggregate metrics.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --case CASE_ATM_001
    python eval/run_eval.py --report eval/report.json
"""
import argparse
import json

from eval import loader, metrics


def evaluate_case(case_id):
    result = {"case_id": case_id}

    case_full = loader.find_ground_truth_case(case_id)
    if case_full is None:
        result["error"] = "no ground truth found"
        return result

    latest_er = loader.find_latest_er_output(case_id)
    er_output = loader.find_latest_successful_er_output(case_id)
    if latest_er is not None:
        result["er_run_status"] = latest_er.get("status")
    if er_output is not None:
        true_clusters = metrics.true_alias_clusters(case_full["ground_truth"])
        pred_clusters = metrics.pred_alias_clusters(er_output)
        result["er"] = {
            "alias_overlap_ratio": metrics.alias_overlap_ratio(true_clusters, pred_clusters),
            "pairwise": metrics.pairwise_prf1(true_clusters, pred_clusters),
            "b_cubed": metrics.b_cubed(true_clusters, pred_clusters),
            "conflict_signal": metrics.er_conflict_signal(case_full, er_output),
            "calibration": metrics.er_confidence_calibration(true_clusters, er_output),
            "output_classification": er_output.get("output_classification"),
        }

    timeline_output = loader.find_latest_timeline_output(case_id)
    if timeline_output is not None:
        result["timeline"] = {
            "segmentation_purity": metrics.event_segmentation_purity(case_full, timeline_output),
            "temporal_edge_accuracy": metrics.temporal_edge_accuracy(case_full, timeline_output),
            "output_classification": timeline_output.get("output_classification"),
        }

    critique_output = loader.find_latest_critique_output(case_id)
    if critique_output is not None:
        result["critique"] = {
            "gap_detection": metrics.critique_gap_detection_signal(case_full, critique_output),
            "verdict": critique_output.get("verdict"),
            "overall_score": critique_output.get("overall_score"),
        }

    showrunner_output = loader.find_latest_showrunner_output(case_id)
    if showrunner_output is not None:
        result["showrunner"] = {
            "action": showrunner_output.get("action"),
            "output_case": showrunner_output.get("output_case"),
            "loops": len(showrunner_output.get("iter_log", [])),
        }

    return result


def _mean(values):
    return sum(values) / len(values) if values else None


def aggregate(results):
    agg = {}

    agg["er_pairwise_f1_mean"] = _mean([r["er"]["pairwise"]["f1"] for r in results if "er" in r])
    agg["er_b_cubed_f1_mean"] = _mean([r["er"]["b_cubed"]["f1"] for r in results if "er" in r])
    agg["er_conflict_signal_accuracy"] = _mean(
        [r["er"]["conflict_signal"]["correct"] for r in results if "er" in r])
    agg["er_run_failure_rate"] = _mean(
        [r.get("er_run_status") == "failed" for r in results if "er_run_status" in r])

    agg["timeline_segmentation_purity_mean"] = _mean([
        r["timeline"]["segmentation_purity"]["purity"] for r in results
        if "timeline" in r and r["timeline"]["segmentation_purity"]["purity"] is not None])
    agg["timeline_temporal_edge_accuracy_mean"] = _mean([
        r["timeline"]["temporal_edge_accuracy"]["accuracy"] for r in results
        if "timeline" in r and r["timeline"]["temporal_edge_accuracy"]["accuracy"] is not None])

    agg["critique_gap_detection_accuracy"] = _mean(
        [r["critique"]["gap_detection"]["correct"] for r in results if "critique" in r])

    agg["showrunner_mean_loops_to_convergence"] = _mean(
        [r["showrunner"]["loops"] for r in results if "showrunner" in r])
    agg["showrunner_escalation_rate"] = _mean(
        [r["showrunner"]["action"] == "human_review" for r in results if "showrunner" in r])

    return {k: v for k, v in agg.items() if v is not None}


def print_report(results, agg):
    print(f"\n{'=' * 70}\nForenSynth Pipeline Evaluation - {len(results)} case(s)\n{'=' * 70}")
    for r in results:
        print(f"\n[{r['case_id']}]")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        if r.get("er_run_status") == "failed":
            print("  ER        WARNING: latest ER run status=failed (scoring the last successful attempt instead)")
        if "er" in r:
            if r["er"]["alias_overlap_ratio"] < 0.5:
                print(f"  ER        WARNING: only {r['er']['alias_overlap_ratio']:.0%} of resolved aliases appear "
                      f"in current ground truth - stored output looks stale vs. a regenerated case; re-run the "
                      f"pipeline on this case before trusting the score below")
            p, b = r["er"]["pairwise"], r["er"]["b_cubed"]
            print(f"  ER        pairwise F1={p['f1']:.3f} (P={p['precision']:.3f} R={p['recall']:.3f})"
                  f"  b-cubed F1={b['f1']:.3f}  classification={r['er']['output_classification']}")
        if "timeline" in r:
            sp = r["timeline"]["segmentation_purity"]["purity"]
            ea = r["timeline"]["temporal_edge_accuracy"]["accuracy"]
            sp_s = f"{sp:.3f}" if sp is not None else "n/a"
            ea_s = f"{ea:.3f}" if ea is not None else "n/a"
            print(f"  Timeline  segmentation_purity={sp_s}  temporal_edge_accuracy={ea_s}"
                  f"  classification={r['timeline']['output_classification']}")
        if "critique" in r:
            print(f"  Critique  verdict={r['critique']['verdict']}  score={r['critique']['overall_score']}"
                  f"  gap_detection_correct={r['critique']['gap_detection']['correct']}")
        if "showrunner" in r:
            print(f"  Showrunner action={r['showrunner']['action']}  output_case={r['showrunner']['output_case']}"
                  f"  loops={r['showrunner']['loops']}")

    print(f"\n{'-' * 70}\nAggregate (n={len(results)})\n{'-' * 70}")
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ForenSynth pipeline output against ground truth")
    parser.add_argument("--case", help="Evaluate a single case_id (default: all cases with output)")
    parser.add_argument("--report", help="Write full JSON report to this path")
    args = parser.parse_args()

    case_ids = [args.case] if args.case else loader.discover_case_ids()
    if not case_ids:
        print("No case outputs found under output/er/. Run the pipeline first.")
        return

    results = [evaluate_case(cid) for cid in case_ids]
    agg = aggregate(results)
    print_report(results, agg)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"cases": results, "aggregate": agg}, f, indent=2)
        print(f"Full report written to {args.report}")


if __name__ == "__main__":
    main()
