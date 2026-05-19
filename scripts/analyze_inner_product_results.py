"""
Cross-experiment analysis of inner-product / alignment pair detection.

Aggregates `top_pairs_{metric}_{dim}D.json` across all experiments and computes:
  - Recall@K vs KNOWN_CASCADES (unordered: (A, B) matches (B, A))
  - Detection consistency across experiments
  - Recall curves comparing all (metric, dim) variants

Outputs (to --output_dir):
  recall_table.csv                          - recall sweep per (metric, dim, top_pct)
  recall_curves.png                         - 6-line recall vs top_pct plot
  dim_comparison.png                        - recall@5%, @10%, @20% bar chart
  triplet_consistency_{metric}_{dim}D.csv  - top-100 most consistent pairs per variant

Usage:
    python scripts/analyze_inner_product_results.py \\
        --exp_dirs results/two_stage_pipeline/exp_0_seed42 ... \\
        --output_dir results/two_stage_pipeline/inner_product_analysis
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Same ground truth as the directional pipeline (scripts/analyze_two_stage_results.py).
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
# Unordered set: a frozenset per pair so matching is symmetric.
KNOWN_UNORDERED = [frozenset({a, b}) for a, b in KNOWN_CASCADES]

METRICS = ("cosine", "inner_product")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dirs", nargs="+", required=True,
                   help="Experiment dirs to aggregate.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dims", type=int, nargs="*", default=None,
                   help="Dim variants to analyze. If omitted, autodetect from "
                        "top_pairs filenames in the first exp_dir.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _autodetect_dims(exp_dirs):
    """Discover (metric, dim) variants by scanning top_pairs_*_*D.json names."""
    variants = set()
    for d in exp_dirs:
        for p in Path(d).glob("top_pairs_*_*D.json"):
            stem = p.stem  # top_pairs_cosine_6D
            parts = stem.split("_")
            # top_pairs_<metric...>_<dim>D
            dim_part = parts[-1]
            if not dim_part.endswith("D"):
                continue
            try:
                dim = int(dim_part[:-1])
            except ValueError:
                continue
            metric = "_".join(parts[2:-1])
            if metric in METRICS:
                variants.add((metric, dim))
    return variants


def load_top_pairs(exp_dir: Path, metric: str, dim: int) -> list:
    path = exp_dir / f"top_pairs_{metric}_{dim}D.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def _pair_set_topk(pairs: list, top_k_pct: float, total_estimated: int) -> set:
    """Return frozenset(A, B) for the top top_k_pct fraction of `pairs`.

    `total_estimated` is the total scored-pair count used as the denominator
    for top_k_pct. We pass it explicitly because `top_pairs.json` only stores
    the truncated head.
    """
    n_top = max(1, int(round(total_estimated * top_k_pct)))
    head = pairs[:n_top]
    return {frozenset({e["A"], e["B"]}) for e in head}


def recall_curve_for_variant(
    exp_top_pairs_per_exp: list,
    top_pcts: list,
) -> pd.DataFrame:
    """Compute recall @ each top_pct for one variant, aggregating across exps.

    For each top_pct k:
      - Per experiment, take its top-k% pairs (unordered).
      - Aggregate: a known cascade is "recalled" if detected in ANY experiment.

    Returns a DataFrame with columns top_pct, n_top_per_exp_mean, tp_known,
    fp, recall, precision.
    """
    # Each exp's pair list is already sorted; full scored count is unknown but
    # roughly n_cytokines * (n_cytokines - 1) / 2 ≈ 4005 for 90 cytokines.
    # We estimate the total scored count from the saved list (it equals
    # total_pairs * stored_top_pct of the runner). Use 4005 unordered as the
    # canonical denominator since runner saved top_pct of that.
    # However, the saved list IS already top_pct=top_pct_saved of the total.
    # To compute recall @ user-supplied top_pct, we slice the saved list
    # treating it as the top of an implicit total. We just slice proportionally.
    rows = []
    if not exp_top_pairs_per_exp:
        return pd.DataFrame(rows)

    # The saved list represents the highest-scoring pairs only. We assume the
    # user-supplied top_pct is <= the saved top_pct. The saved list length is
    # n_saved = saved_top_pct * n_total_pairs, so the slice for user top_pct
    # is (user_top_pct / saved_top_pct) * n_saved. We can't recover the saved
    # top_pct precisely, so we approximate by assuming the user wants top_k
    # PAIRS where k = user_top_pct * 4005 (canonical unordered count).
    CANONICAL_TOTAL = 4005   # 90 cytokines × 89 / 2

    for k in top_pcts:
        n_top_target = max(1, int(round(CANONICAL_TOTAL * k)))
        per_exp_recalled = []
        union_recalled = set()
        n_top_per_exp = []
        for pairs in exp_top_pairs_per_exp:
            head = pairs[:n_top_target]
            n_top_per_exp.append(len(head))
            top_set = {frozenset({e["A"], e["B"]}) for e in head}
            recalled = sum(1 for kp in KNOWN_UNORDERED if kp in top_set)
            per_exp_recalled.append(recalled)
            for kp in KNOWN_UNORDERED:
                if kp in top_set:
                    union_recalled.add(kp)

        rows.append({
            "top_pct":            k,
            "n_top_target":       n_top_target,
            "n_top_per_exp_mean": float(np.mean(n_top_per_exp)),
            "recall_any_exp":     len(union_recalled) / len(KNOWN_UNORDERED),
            "tp_any_exp":         len(union_recalled),
            "recall_per_exp_mean": float(np.mean(per_exp_recalled)) /
                                   len(KNOWN_UNORDERED),
            "recall_per_exp_max":  float(np.max(per_exp_recalled)) /
                                   len(KNOWN_UNORDERED),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-experiment consistency
# ---------------------------------------------------------------------------

def consistency_table(exp_top_pairs_per_exp: list, top_k_per_exp: int) -> pd.DataFrame:
    """For each unordered pair, count in how many experiments it appears in
    the top-`top_k_per_exp` slots.
    """
    n_exps = len(exp_top_pairs_per_exp)
    detect_count: dict = defaultdict(int)
    pair_to_relays: dict = defaultdict(list)
    pair_to_scores: dict = defaultdict(list)
    canonical_ab: dict = {}  # frozenset → (A, B) of the first occurrence

    for j, pairs in enumerate(exp_top_pairs_per_exp):
        head = pairs[:top_k_per_exp]
        seen_in_this_exp = set()
        for e in head:
            key = frozenset({e["A"], e["B"]})
            if key in seen_in_this_exp:
                continue
            seen_in_this_exp.add(key)
            detect_count[key] += 1
            pair_to_relays[key].append(e.get("relay_cell_type"))
            pair_to_scores[key].append(e["score"])
            canonical_ab.setdefault(key, (e["A"], e["B"]))

    rows = []
    for key, n_det in detect_count.items():
        a, b = canonical_ab[key]
        is_known = key in KNOWN_UNORDERED
        rows.append({
            "A": a, "B": b,
            "n_exps_detected": n_det,
            "mean_score": float(np.mean(pair_to_scores[key])),
            "n_unique_relay_cts": len({r for r in pair_to_relays[key] if r}),
            "is_known": is_known,
        })
    df = pd.DataFrame(rows).sort_values(
        ["n_exps_detected", "mean_score"], ascending=[False, False],
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dirs = [Path(d) for d in args.exp_dirs]

    # Discover (metric, dim) variants
    if args.dims:
        variants = sorted([(m, d) for m in METRICS for d in args.dims])
    else:
        variants = sorted(_autodetect_dims(exp_dirs))
    print(f"Variants found: {variants}", flush=True)
    if not variants:
        raise SystemExit("No top_pairs_*.json files found.")

    # Top-pct sweep — these are the K values used for recall/precision.
    top_pcts = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    recall_rows = []
    for metric, dim in variants:
        exp_pairs = [load_top_pairs(d, metric, dim) for d in exp_dirs]
        n_present = sum(1 for p in exp_pairs if p)
        print(f"\n=== {metric:>13} {dim:>3}D ({n_present}/{len(exp_dirs)} exps present) ===",
              flush=True)
        if n_present == 0:
            continue

        df_rc = recall_curve_for_variant(exp_pairs, top_pcts)
        df_rc.insert(0, "dim", dim)
        df_rc.insert(0, "metric", metric)
        recall_rows.append(df_rc)
        print(df_rc.round(3).to_string(index=False), flush=True)

        # Consistency at canonical top_k = top 10% of canonical total ≈ 400
        df_cons = consistency_table(exp_pairs, top_k_per_exp=400)
        cons_path = out_dir / f"triplet_consistency_{metric}_{dim}D.csv"
        df_cons.head(100).to_csv(cons_path, index=False)
        print(f"  Saved: {cons_path.name}", flush=True)

    if not recall_rows:
        raise SystemExit("No variants had any data.")

    df_all = pd.concat(recall_rows, ignore_index=True)
    df_all.to_csv(out_dir / "recall_table.csv", index=False)
    print(f"\nSaved: recall_table.csv  ({len(df_all)} rows)", flush=True)

    # ── Plot: recall vs top_pct ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for (metric, dim), grp in df_all.groupby(["metric", "dim"]):
        label = f"{metric}, {dim}D"
        ax.plot(grp.top_pct * 100, grp.recall_any_exp * 100, "-o",
                ms=4, label=label)
    ax.axvline(5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Top X% of unordered pairs")
    ax.set_ylabel("Recall (any experiment) (%)")
    ax.set_title("Known cascade recall — union across 8 experiments")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    ax = axes[1]
    for (metric, dim), grp in df_all.groupby(["metric", "dim"]):
        label = f"{metric}, {dim}D"
        ax.plot(grp.top_pct * 100, grp.recall_per_exp_mean * 100, "-o",
                ms=4, label=label)
    ax.axvline(5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Top X% of unordered pairs")
    ax.set_ylabel("Recall per-experiment mean (%)")
    ax.set_title("Known cascade recall — mean across 8 experiments")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    fig.suptitle(
        f"Alignment-based pair detection — {len(variants)} variants, "
        f"{len(KNOWN_UNORDERED)} known pairs",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: recall_curves.png", flush=True)

    # ── Plot: bar chart at top_pct=5%, 10%, 20% ───────────────────────────
    targets = [0.05, 0.10, 0.20]
    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 1.0), 5))
    width = 0.25
    xpos = np.arange(len(variants))
    variant_labels = [f"{m}\n{d}D" for (m, d) in variants]

    for i, k in enumerate(targets):
        recalls = []
        for metric, dim in variants:
            row = df_all[(df_all.metric == metric) & (df_all.dim == dim) &
                         (np.isclose(df_all.top_pct, k))]
            if len(row) == 0:
                recalls.append(0.0)
            else:
                recalls.append(float(row.iloc[0].recall_any_exp))
        ax.bar(xpos + i * width, np.array(recalls) * 100, width=width,
               label=f"top {int(k*100)}%")

    ax.set_xticks(xpos + width)
    ax.set_xticklabels(variant_labels)
    ax.set_ylabel("Known cascade recall (union across exps) %")
    ax.set_title("Recall@K by (metric, dim) — alignment pair detection")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "dim_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: dim_comparison.png", flush=True)

    # ── Summary printout: best variant ────────────────────────────────────
    print("\n=== BEST VARIANT BY recall_any_exp at top_pct=0.05 ===")
    df5 = df_all[np.isclose(df_all.top_pct, 0.05)].copy()
    df5 = df5.sort_values("recall_any_exp", ascending=False)
    print(df5[["metric", "dim", "n_top_target", "recall_any_exp",
               "recall_per_exp_mean"]].to_string(index=False), flush=True)

    print(f"\nResults saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
