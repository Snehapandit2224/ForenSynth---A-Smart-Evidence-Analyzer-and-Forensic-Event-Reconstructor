"""Evaluation metrics for the ForenSynth pipeline, scored against the synthetic
generator's ground truth (cases/**/*.json 'ground_truth' block and 'noise_tags').
"""
from itertools import combinations
from collections import defaultdict


# ---------------------------------------------------------------------------
# Entity Resolution
# ---------------------------------------------------------------------------

def true_alias_clusters(ground_truth):
    """alias -> true canonical entity_id, from ground_truth.entity_mapping."""
    return dict(ground_truth["entity_mapping"])


def pred_alias_clusters(er_output):
    """alias -> predicted canonical entity_id, from er_output.canonical_entities."""
    pred = {}
    for ent in er_output.get("canonical_entities", []):
        for alias in ent.get("aliases", []):
            pred[alias] = ent["entity_id"]
    return pred


def alias_overlap_ratio(true_clusters, pred_clusters):
    """Fraction of aliases the ER output actually resolved that appear in this
    case's current ground truth. The generator assigns aliases (Person_NN,
    Speaker_X, ...) randomly on each run, so a stored pipeline output scored
    against a *regenerated* case file will share almost no aliases with the
    ground truth even though nothing is wrong with either file on its own -
    this catches that mismatch before it's misread as a bad reconstruction."""
    if not pred_clusters:
        return 1.0
    overlap = sum(1 for a in pred_clusters if a in true_clusters)
    return overlap / len(pred_clusters)


def pairwise_prf1(true_clusters, pred_clusters):
    """Pairwise precision/recall/F1 over aliases present in true_clusters.

    An alias the pipeline never assigned to any canonical entity is treated as
    its own singleton cluster (i.e. the pipeline failed to resolve it).
    """
    aliases = sorted(true_clusters.keys())
    tp = fp = fn = 0
    for a, b in combinations(aliases, 2):
        same_true = true_clusters[a] == true_clusters[b]
        same_pred = pred_clusters.get(a, f"__singleton_{a}") == pred_clusters.get(b, f"__singleton_{b}")
        if same_true and same_pred:
            tp += 1
        elif same_pred and not same_true:
            fp += 1
        elif same_true and not same_pred:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def b_cubed(true_clusters, pred_clusters):
    """B-cubed precision/recall/F1, averaged per-alias."""
    aliases = sorted(true_clusters.keys())
    true_groups = defaultdict(set)
    for a in aliases:
        true_groups[true_clusters[a]].add(a)
    pred_groups = defaultdict(set)
    for a in aliases:
        pred_groups[pred_clusters.get(a, f"__singleton_{a}")].add(a)

    precisions, recalls = [], []
    for a in aliases:
        true_g = true_groups[true_clusters[a]]
        pred_g = pred_groups[pred_clusters.get(a, f"__singleton_{a}")]
        overlap = len(true_g & pred_g)
        precisions.append(overlap / len(pred_g))
        recalls.append(overlap / len(true_g))
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def er_conflict_signal(case_full, er_output):
    """Did ER flag a conflict on a case where the generator actually injected a
    contradiction (and vice versa)?"""
    noise_injected = any(
        "contradiction" in obs.get("noise_tags", [])
        for obs in case_full.get("observations", [])
    )
    flagged = er_output.get("conflicts_detected", 0) > 0
    return {"noise_injected": noise_injected, "flagged": flagged, "correct": noise_injected == flagged}


def er_confidence_calibration(true_clusters, er_output):
    """Brier score of each canonical entity's confidence_score against whether
    that cluster is actually 'pure' (all its known-alias members share one true
    entity id)."""
    scored = []
    for ent in er_output.get("canonical_entities", []):
        aliases = [a for a in ent.get("aliases", []) if a in true_clusters]
        if not aliases:
            continue
        true_ids = {true_clusters[a] for a in aliases}
        correctness = 1.0 if len(true_ids) == 1 else 0.0
        confidence = ent.get("confidence_score", 0.0)
        scored.append((confidence, correctness))
    if not scored:
        return {"brier_score": None, "n": 0}
    brier = sum((c - y) ** 2 for c, y in scored) / len(scored)
    return {"brier_score": brier, "n": len(scored)}


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def obs_to_true_event(case_full):
    """obs_id -> (true_event_id, true_timestamp), from the full case's
    observations (event_ref) and ground_truth.events (timestamp)."""
    event_ts = {e["event_id"]: e["timestamp"] for e in case_full["ground_truth"]["events"]}
    mapping = {}
    for obs in case_full["observations"]:
        ref = obs.get("event_ref")
        if ref in event_ts:
            mapping[obs["obs_id"]] = (ref, event_ts[ref])
    return mapping


def event_segmentation_purity(case_full, timeline_output):
    """For each predicted event (which bundles one or more obs_ids), the fraction
    of its obs_ids that share the majority true event_ref among that group.
    1.0 = every predicted event corresponds to exactly one true event (no
    incorrect merging of unrelated observations)."""
    obs_map = obs_to_true_event(case_full)
    events = timeline_output.get("events", [])
    total_obs = correct_obs = 0
    for ev in events:
        refs = [obs_map[o][0] for o in ev.get("obs_ids", []) if o in obs_map]
        if not refs:
            continue
        majority = max(set(refs), key=refs.count)
        total_obs += len(refs)
        correct_obs += refs.count(majority)
    purity = correct_obs / total_obs if total_obs else None
    return {"purity": purity, "events_scored": len(events), "obs_scored": total_obs}


def temporal_edge_accuracy(case_full, timeline_output):
    """Fraction of graph edges (TEMPORAL or CAUSAL) whose predicted BEFORE/AFTER
    relation agrees with true event order (derived from each endpoint event's
    obs_ids' true timestamps). Edges between obs sharing one true event, or
    tagged SIMULTANEOUS/UNKNOWN, are excluded (no ground-truth order to check)."""
    obs_map = obs_to_true_event(case_full)
    events_by_id = {e["event_id"]: e for e in timeline_output.get("events", [])}

    def true_time_for(event_id):
        ev = events_by_id.get(event_id)
        if not ev:
            return None
        times = [obs_map[o][1] for o in ev.get("obs_ids", []) if o in obs_map]
        return min(times) if times else None

    graph = timeline_output.get("timeline_graph", {})
    edges = graph.get("edges") or timeline_output.get("causal_links", [])
    scored = correct = 0
    for edge in edges:
        if edge.get("relation") not in ("BEFORE", "AFTER"):
            continue
        t_src = true_time_for(edge["source"])
        t_tgt = true_time_for(edge["target"])
        if t_src is None or t_tgt is None or t_src == t_tgt:
            continue
        scored += 1
        true_before = t_src < t_tgt
        pred_before = edge["relation"] == "BEFORE"
        if true_before == pred_before:
            correct += 1
    accuracy = correct / scored if scored else None
    return {"accuracy": accuracy, "edges_scored": scored}


# ---------------------------------------------------------------------------
# Critique
# ---------------------------------------------------------------------------

def critique_gap_detection_signal(case_full, critique_output):
    """Did the critique agent flag something (REVISE verdict, or any issue/gap)
    on a case where the generator injected noise that should be catchable
    (contradiction or missing_modality), and stay quiet otherwise?"""
    detectable_tags = {"contradiction", "missing_modality"}
    noise_injected = any(
        set(obs.get("noise_tags", [])) & detectable_tags
        for obs in case_full.get("observations", [])
    )
    flagged = (
        critique_output.get("verdict") == "REVISE"
        or len(critique_output.get("issues", [])) > 0
        or len(critique_output.get("gaps", [])) > 0
    )
    return {"noise_injected": noise_injected, "flagged": flagged, "correct": noise_injected == flagged}
