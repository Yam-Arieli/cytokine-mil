"""
Post-processing: load existing latent_geo_results.pkl and produce a new
top_pairs.json using per-source top-K filtering instead of global top-X%.

This avoids re-running the expensive geo computation (embedding cache +
Wilcoxon tests). Useful after the initial geo run has completed.

Filtering logic:
  For each source cytokine A (excluding PBS), collect all (A, B) pairs with
  W_stat >= --min_w_stat, sort by W descending, keep top --top_k_per_source.
  Final list is globally re-ranked by W descending.

This ensures every cytokine gets its best cascade candidates tested, preventing
high-signal cytokines (e.g. IFN-gamma) from dominating the ablation budget.

Usage:
    python scripts/filter_top_pairs.py \\
        --exp_dir results/two_stage_pipeline/exp_0_seed42 \\
        --top_k_per_source 3 \\
        --min_w_stat 0

    # All 8 experiments at once:
    for d in results/two_stage_pipeline/exp_*/; do
        python scripts/filter_top_pairs.py --exp_dir $d --top_k_per_source 3
    done
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-source top-K filtering of geo candidate pairs."
    )
    p.add_argument("--exp_dir", required=True,
                   help="Experiment directory containing latent_geo_results.pkl.")
    p.add_argument("--top_k_per_source", type=int, default=3,
                   help="Number of best target cytokines to keep per source cytokine. "
                        "Total pairs ≈ n_cytokines × top_k_per_source.")
    p.add_argument("--min_w_stat", type=float, default=0.0,
                   help="Minimum W statistic to include a pair (range 0–55 for n=10 donors). "
                        "E.g. 30 ≈ 9/10 donors agree in the forward direction.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write top_pairs.json (default: same as --exp_dir).")
    return p.parse_args()


def main():
    args    = parse_args()
    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = exp_dir / "latent_geo_results.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"latent_geo_results.pkl not found in {exp_dir}")

    with open(pkl_path, "rb") as f:
        geo = pickle.load(f)

    sig_result = geo["sig_result"]
    p_pair_fwd = sig_result["p_pair_fwd"]   # {(A, B) -> min Bonferroni p across cell types}
    W_pair_fwd = sig_result["W_pair_fwd"]   # {(A, B) -> max W across cell types}
    relay_T    = sig_result["relay_T"]       # {(A, B) -> relay cell type (argmax W)}

    n_total_pairs = len(p_pair_fwd)

    # Group by source cytokine, excluding PBS on either side
    by_source: dict = defaultdict(list)
    n_below_floor = 0
    for (a, b), p in p_pair_fwd.items():
        if a == "PBS" or b == "PBS":
            continue
        w = float(W_pair_fwd.get((a, b), 0.0))
        if w < args.min_w_stat:
            n_below_floor += 1
            continue
        by_source[a].append({
            "A": a, "B": b,
            "relay_cell_type": relay_T.get((a, b)),
            "p_bonf": float(p),
            "W_stat": w,
        })

    # Per-source top-K by W descending, p ascending as tiebreak
    top_pairs = []
    sources_with_no_pairs = []
    for a in sorted(by_source):
        candidates = sorted(by_source[a], key=lambda x: (-x["W_stat"], x["p_bonf"]))
        selected   = candidates[:args.top_k_per_source]
        top_pairs.extend(selected)
        if not selected:
            sources_with_no_pairs.append(a)

    # Global re-rank by W descending
    top_pairs.sort(key=lambda x: (-x["W_stat"], x["p_bonf"]))
    for rank, entry in enumerate(top_pairs, 1):
        entry["rank"] = rank

    with open(out_dir / "top_pairs.json", "w") as f:
        json.dump(top_pairs, f, indent=2)

    # Summary
    n_sources = len(by_source)
    print(f"Experiment  : {exp_dir.name}", flush=True)
    print(f"Total ordered pairs in geo : {n_total_pairs}", flush=True)
    print(f"Pairs below W floor (<{args.min_w_stat:.0f}) : {n_below_floor}", flush=True)
    print(f"Sources with ≥1 candidate  : {n_sources}", flush=True)
    print(f"Per-source top-{args.top_k_per_source} selected  : {len(top_pairs)} pairs", flush=True)
    print(f"Saved to    : {out_dir / 'top_pairs.json'}", flush=True)

    if sources_with_no_pairs:
        print(f"WARNING: {len(sources_with_no_pairs)} sources had no pairs after floor filter: "
              f"{sources_with_no_pairs[:5]}", flush=True)

    # Print top-25 table
    header = (f"{'Rank':>4}  {'A':<18}  {'B':<18}  "
              f"{'relay_T':<22}  {'W_stat':>7}  {'p_bonf':>10}")
    print(f"\n{header}")
    print("-" * len(header))
    for entry in top_pairs[:25]:
        rt = str(entry["relay_cell_type"] or "?")
        print(f"{entry['rank']:>4}  {entry['A']:<18}  {entry['B']:<18}  "
              f"{rt:<22}  {entry['W_stat']:>7.1f}  {entry['p_bonf']:>10.4f}",
              flush=True)
    if len(top_pairs) > 25:
        print(f"  ... ({len(top_pairs) - 25} more pairs not shown)")


if __name__ == "__main__":
    main()
