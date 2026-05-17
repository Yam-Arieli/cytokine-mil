"""
Aggregate and analyze results from the two-stage cascade detection pipeline.

Reads top_pairs.json (geo detection) and ablation_scores_shard_*.pkl (direction)
from each experiment directory; computes cross-experiment consistency, recall vs
KNOWN_CASCADES, ablation direction accuracy, and threshold sensitivity.

Outputs (to --output_dir):
  triplet_consistency.csv      - per pair: n_exps detected, direction call votes
  recall_specificity.csv       - sweep top_pct → recall / precision vs known pairs
  direction_accuracy.csv       - per known pair: detected? direction correct?
  precision_recall_curve.png
  consistency_heatmap.png      - (pair) × (exp) detection matrix

Usage:
    python scripts/analyze_two_stage_results.py \
        --exp_dirs results/two_stage_pipeline/exp_0_seed42 \
                   results/two_stage_pipeline/exp_1_seed123 \
                   ... \
        --output_dir results/two_stage_pipeline/analysis \
        --n_shards 4
"""

import argparse
import glob
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KNOWN_CASCADES = [
    ("IL-12",     "IFN-gamma"),
    ("IL-1-beta", "IL-6"),
    ("IL-2",      "IL-15"),
    ("IL-33",     "IL-13"),
    ("IL-18",     "IFN-gamma"),
    ("IL-21",     "IL-10"),
    ("TNF-alpha", "IL-6"),
    ("IFN-alpha1","IFN-gamma"),
    ("IL-10",     "IL-6"),
    ("IL-4",      "IL-13"),
    ("IL-27",     "IFN-gamma"),
]
KNOWN_SET  = {(a, b) for a, b in KNOWN_CASCADES}
KNOWN_REVS = {(b, a) for a, b in KNOWN_CASCADES}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dirs",   nargs="+", required=True,
                   help="List of experiment output directories.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_shards",   type=int, default=4,
                   help="Number of ablation shards per experiment.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_top_pairs(exp_dir: Path) -> list:
    path = exp_dir / "top_pairs.json"
    if not path.exists():
        print(f"  MISSING top_pairs.json in {exp_dir.name}")
        return []
    with open(path) as f:
        return json.load(f)


def load_ablation_shards(exp_dir: Path, n_shards: int) -> dict:
    """
    Aggregate ablation score pooled dicts from all shards.
    Returns combined {(source, target, cell_type): [scores]} dict.
    """
    combined: dict = defaultdict(list)
    for shard_idx in range(n_shards):
        path = exp_dir / f"ablation_scores_shard_{shard_idx}.pkl"
        if not path.exists():
            # Fallback: maybe n_shards=1 run (no shard suffix)
            path = exp_dir / "ablation_scores.pkl"
            if not path.exists():
                print(f"  MISSING ablation shard {shard_idx} in {exp_dir.name}")
                continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        pooled = data.get("pooled", {})
        for key, vals in pooled.items():
            combined[key].extend(vals)

    return dict(combined)


# ---------------------------------------------------------------------------
# Direction call from pooled ablation scores
# ---------------------------------------------------------------------------

def direction_call_from_pooled(pooled: dict, a: str, b: str) -> str:
    """
    For pair (A, B): compare mean relay score in fwd vs rev direction.
    Returns 'A→B', 'B→A', 'shared', or 'no_data'.
    """
    fwd_scores = {ct: np.mean(v) for (src, tgt, ct), v in pooled.items()
                  if src == a and tgt == b}
    rev_scores = {ct: np.mean(v) for (src, tgt, ct), v in pooled.items()
                  if src == b and tgt == a}
    if not fwd_scores and not rev_scores:
        return "no_data"
    best_fwd = max(fwd_scores.values()) if fwd_scores else -np.inf
    best_rev = max(rev_scores.values()) if rev_scores else -np.inf
    if best_fwd > best_rev:
        return "A→B"
    elif best_rev > best_fwd:
        return "B→A"
    return "shared"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_exps  = len(args.exp_dirs)

    exp_dirs  = [Path(d) for d in args.exp_dirs]
    exp_names = [d.name for d in exp_dirs]

    # ── Load geo top-pairs and ablation per experiment ─────────────────────
    geo_pairs_per_exp:  list[list] = []   # list of top_pairs lists
    ablation_per_exp:   list[dict] = []   # list of pooled score dicts
    all_detected_pairs: set        = set()

    for exp_dir in exp_dirs:
        top = load_top_pairs(exp_dir)
        abl = load_ablation_shards(exp_dir, args.n_shards)
        geo_pairs_per_exp.append(top)
        ablation_per_exp.append(abl)
        for entry in top:
            all_detected_pairs.add((entry["A"], entry["B"]))

    all_pairs_list = sorted(all_detected_pairs)
    print(f"Unique ordered pairs detected across all exps: {len(all_pairs_list)}", flush=True)

    # ── Cross-experiment consistency table ─────────────────────────────────
    # For each pair: how many experiments detected it? What direction calls?
    detect_matrix = np.zeros((len(all_pairs_list), n_exps), dtype=np.int8)
    for j, exp_pairs in enumerate(geo_pairs_per_exp):
        detected_this_exp = {(e["A"], e["B"]) for e in exp_pairs}
        for i, (a, b) in enumerate(all_pairs_list):
            if (a, b) in detected_this_exp:
                detect_matrix[i, j] = 1

    n_detected = detect_matrix.sum(axis=1)

    # Ablation direction call per pair per experiment
    direction_votes = []
    for (a, b) in all_pairs_list:
        votes = []
        for abl in ablation_per_exp:
            call = direction_call_from_pooled(abl, a, b)
            votes.append(call)
        direction_votes.append(votes)

    def majority_call(votes):
        counts = defaultdict(int)
        for v in votes:
            if v != "no_data":
                counts[v] += 1
        return max(counts, key=counts.get) if counts else "no_data"

    consistency_rows = []
    for i, (a, b) in enumerate(all_pairs_list):
        votes      = direction_votes[i]
        n_det      = int(n_detected[i])
        maj_call   = majority_call(votes)
        is_known   = (a, b) in KNOWN_SET
        is_rev_known = (b, a) in KNOWN_SET
        consistency_rows.append({
            "A": a, "B": b,
            "n_exps_detected": n_det,
            "majority_call": maj_call,
            "call_A_to_B": votes.count("A→B"),
            "call_B_to_A": votes.count("B→A"),
            "call_shared": votes.count("shared"),
            "call_no_data": votes.count("no_data"),
            "is_known_fwd": is_known,
            "is_known_rev": is_rev_known,
        })

    df_cons = pd.DataFrame(consistency_rows).sort_values(
        "n_exps_detected", ascending=False
    )
    df_cons.to_csv(out_dir / "triplet_consistency.csv", index=False)
    print(f"Saved: triplet_consistency.csv  ({len(df_cons)} pairs)", flush=True)

    # ── Direction accuracy on known cascade pairs ──────────────────────────
    dir_acc_rows = []
    for a, b in KNOWN_CASCADES:
        in_geo_any = any((a, b) in {(e["A"], e["B"]) for e in ep}
                         for ep in geo_pairs_per_exp)
        n_exps_det = int(detect_matrix[
            all_pairs_list.index((a, b)), :
        ].sum()) if (a, b) in all_pairs_list else 0

        if (a, b) in all_pairs_list:
            votes    = direction_votes[all_pairs_list.index((a, b))]
            maj_call = majority_call(votes)
            correct  = maj_call == "A→B"
        else:
            votes    = ["no_data"] * n_exps
            maj_call = "not_detected"
            correct  = False

        dir_acc_rows.append({
            "A": a, "B": b,
            "detected_any_exp": in_geo_any,
            "n_exps_detected": n_exps_det,
            "majority_call": maj_call,
            "direction_correct": correct,
        })

    df_dir = pd.DataFrame(dir_acc_rows)
    df_dir.to_csv(out_dir / "direction_accuracy.csv", index=False)

    n_detected_known  = df_dir.detected_any_exp.sum()
    n_correct_dir     = df_dir.direction_correct.sum()
    n_total_known     = len(KNOWN_CASCADES)
    print(f"\nKnown cascade recovery:", flush=True)
    print(f"  Detected in ≥1 exp : {n_detected_known}/{n_total_known}", flush=True)
    print(f"  Direction correct  : {n_correct_dir}/{n_total_known}", flush=True)
    print(df_dir[["A","B","n_exps_detected","majority_call","direction_correct"]].to_string(), flush=True)

    # ── Threshold sweep: recall & precision vs KNOWN_CASCADES ─────────────
    # For each experiment, rank pairs by p_bonf (stored in top_pairs.json);
    # sweep threshold to get recall/precision curves.
    # "Detected" = pair appears in top_pairs; "Correct direction" = A→B call.
    thresholds = np.arange(0.01, 0.51, 0.01)

    # Aggregate p_bonf scores: for each (A, B), take the min p_bonf across exps
    pair_min_p: dict = {}
    for exp_pairs in geo_pairs_per_exp:
        for entry in exp_pairs:
            key = (entry["A"], entry["B"])
            p   = float(entry["p_bonf"])
            pair_min_p[key] = min(pair_min_p.get(key, 1.0), p)

    ranked = sorted(pair_min_p.items(), key=lambda x: x[1])   # ascending p = most significant

    recall_rows = []
    for frac in thresholds:
        n_top    = max(1, int(round(len(ranked) * frac)))
        top_set  = {k for k, _ in ranked[:n_top]}
        tp_fwd   = sum(1 for p in KNOWN_SET if p in top_set)
        fp       = n_top - tp_fwd
        recall   = tp_fwd / len(KNOWN_CASCADES) if KNOWN_CASCADES else 0.0
        precision= tp_fwd / n_top if n_top > 0 else 0.0
        recall_rows.append({
            "top_pct": round(frac, 2),
            "n_top": n_top,
            "tp_known": tp_fwd,
            "fp": fp,
            "recall": recall,
            "precision": precision,
        })

    df_rc = pd.DataFrame(recall_rows)
    df_rc.to_csv(out_dir / "recall_specificity.csv", index=False)

    # ── Precision-recall plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df_rc.top_pct * 100, df_rc.recall * 100, "-o", ms=3, color="steelblue")
    ax.axvline(5, color="gray", ls="--", lw=0.8, label="5% threshold")
    ax.set_xlabel("Top X% ordered pairs (by min p_bonf)")
    ax.set_ylabel("Recall of KNOWN_CASCADES (%)")
    ax.set_title("Geo detection recall\n(aggregated across experiments)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    ax = axes[1]
    ax.plot(df_rc.recall * 100, df_rc.precision * 100, "-o", ms=3, color="darkred")
    ax.set_xlabel("Recall of KNOWN_CASCADES (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision-Recall curve\n(geo detection stage)")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 105); ax.set_ylim(0, 105)

    fig.suptitle(
        f"Two-Stage Pipeline — Geo Detection Analysis\n"
        f"{n_exps} experiments, {len(KNOWN_CASCADES)} known cascade pairs",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: precision_recall_curve.png", flush=True)

    # ── Consistency heatmap: pair × experiment ─────────────────────────────
    # Show top-50 most consistently detected pairs
    top_n  = min(50, len(all_pairs_list))
    order  = np.argsort(n_detected)[::-1][:top_n]
    mat_s  = detect_matrix[order]
    pairs_s = [all_pairs_list[i] for i in order]
    labels  = [f"{'★ ' if p in KNOWN_SET else ''}{p[0]}→{p[1]}" for p in pairs_s]

    fig, ax = plt.subplots(figsize=(max(8, n_exps * 0.9), max(8, top_n * 0.35)))
    im = ax.imshow(mat_s, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_exps))
    ax.set_xticklabels([n.replace("exp_","") for n in exp_names], rotation=45, ha="right")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(
        f"Geo detection consistency  |  top-{top_n} pairs by n_exps_detected\n"
        f"★ = known cascade  |  Blue = detected",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "consistency_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: consistency_heatmap.png", flush=True)

    # ── Final summary printout ─────────────────────────────────────────────
    at5pct = df_rc[df_rc.top_pct == 0.05].iloc[0]
    print(f"\n=== Summary at 5% threshold ===")
    print(f"  Pairs in top-5% : {int(at5pct.n_top)}")
    print(f"  Known recall    : {at5pct.tp_known}/{len(KNOWN_CASCADES)}  "
          f"({at5pct.recall*100:.1f}%)")
    print(f"  Precision       : {at5pct.precision*100:.1f}%")
    print(f"  Direction acc   : {n_correct_dir}/{n_total_known}  "
          f"({100*n_correct_dir/max(n_total_known,1):.1f}%)")
    print(f"\nResults saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
