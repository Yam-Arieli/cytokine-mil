"""
Triple synthesis: geo ∧ ablation conjunction.

For each seed directory, applies the conjunction defined in
.claude/plans/suggest-how-to-continue-effervescent-stonebraker.md (Step 3):

A triple (A, B, T) is reported by a single seed iff:

 1. Geo refined readout (latent_geo_results.pkl / sig_result):
      - cascade_call[(A, B)] == "A->B"
      - p_fwd_bonf[(A, B, T)] <= alpha   (default 0.05)
 2. Ablation (ablation_scores_shard_*.pkl, key "pooled"):
      - direction call from pooled = "A→B"
        (max over T of mean(pooled[(A,B,T)]) > max over T of mean(pooled[(B,A,T)]))
      - T == argmax_T mean(pooled[(A,B,T)])   (same T as geo)
      - mean(pooled[(A,B,T)]) > 0

A triple is reported in the final CSV iff it appears in >= --min_seeds seeds
(default 5 of 8).

Output: cascade_triples.csv with columns:
    A, B, T,
    n_seeds,
    p_fwd_bonf_median, p_fwd_bonf_min, p_fwd_bonf_max,
    ablation_relay_median, ablation_relay_min, ablation_relay_max,
    geo_cascade_calls,     # comma-separated per-seed cascade_call values
    ablation_direction_calls,
    seeds_supporting        # space-separated seed_dir names
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Geo loading
# ---------------------------------------------------------------------------

def _load_geo_sig(geo_pkl_path: Path) -> Optional[dict]:
    if not geo_pkl_path.exists():
        return None
    with open(geo_pkl_path, "rb") as f:
        geo = pickle.load(f)
    return geo.get("sig_result")


def _geo_significant_T(
    sig: dict,
    a: str,
    b: str,
    alpha: float,
) -> List[Tuple[str, float]]:
    """Return [(T, p_fwd_bonf)] for cell types T where geo declares (A->B) at T.

    Only triples where cascade_call[(A, B)] == "A->B" are considered.
    """
    cascade_call = sig["cascade_call"]
    if cascade_call.get((a, b)) != "A->B":
        return []
    p_fwd_bonf = sig["p_fwd_bonf"]
    out = []
    for (aa, bb, ct), p in p_fwd_bonf.items():
        if aa == a and bb == b and p <= alpha:
            out.append((ct, float(p)))
    return out


# ---------------------------------------------------------------------------
# Ablation loading
# ---------------------------------------------------------------------------

def _load_ablation_pooled(ablation_dir: Path) -> Dict[Tuple[str, str, str], list]:
    """Combine all ablation_scores_shard_*.pkl in ablation_dir into one pooled dict.

    Falls back to ablation_scores.pkl (n_shards=1) if no shards present.
    """
    pooled: Dict[Tuple[str, str, str], list] = defaultdict(list)
    shard_paths = sorted(ablation_dir.glob("ablation_scores_shard_*.pkl"))
    if not shard_paths:
        single = ablation_dir / "ablation_scores.pkl"
        if single.exists():
            shard_paths = [single]
    for p in shard_paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        for key, vals in data.get("pooled", {}).items():
            pooled[key].extend(vals)
    return dict(pooled)


def _ablation_call(
    pooled: Dict[Tuple[str, str, str], list],
    a: str,
    b: str,
) -> Tuple[str, Optional[str], Optional[float]]:
    """Return (direction_call, T_argmax_for_called_direction, mean_relay_at_T).

    direction_call ∈ {"A→B", "B→A", "shared", "no_data"}.
    If A→B: T_argmax and mean_relay refer to forward direction.
    If B→A: they refer to reverse direction (swapped to be informative for B→A).
    """
    fwd = {ct: float(np.mean(v))
           for (src, tgt, ct), v in pooled.items()
           if src == a and tgt == b and len(v) > 0}
    rev = {ct: float(np.mean(v))
           for (src, tgt, ct), v in pooled.items()
           if src == b and tgt == a and len(v) > 0}
    if not fwd and not rev:
        return ("no_data", None, None)

    best_fwd = max(fwd.values()) if fwd else -np.inf
    best_rev = max(rev.values()) if rev else -np.inf
    if best_fwd > best_rev:
        T_star = max(fwd, key=fwd.get)
        return ("A→B", T_star, best_fwd)
    elif best_rev > best_fwd:
        T_star = max(rev, key=rev.get)
        return ("B→A", T_star, best_rev)
    return ("shared", None, None)


# ---------------------------------------------------------------------------
# Per-seed triple extraction
# ---------------------------------------------------------------------------

def _seed_triples(
    exp_dir: Path,
    alpha: float,
    ablation_subdir: str = ".",
) -> List[dict]:
    """Apply the geo ∧ ablation conjunction within one seed.

    Geo result is read from exp_dir/latent_geo_results.pkl.
    Ablation shards are read from exp_dir/<ablation_subdir>/.

    Returns a list of {A, B, T, p_fwd_bonf, ablation_relay,
                      geo_cascade_call, ablation_direction_call, seed}.
    """
    sig = _load_geo_sig(exp_dir / "latent_geo_results.pkl")
    if sig is None:
        return []
    pooled = _load_ablation_pooled(exp_dir / ablation_subdir)
    if not pooled:
        return []

    out: List[dict] = []
    cascade_call = sig["cascade_call"]
    # Iterate every ordered pair geo called A->B
    for (a, b), call in cascade_call.items():
        if call != "A->B":
            continue

        # Geo-significant T set for this ordered pair
        geo_TS = _geo_significant_T(sig, a, b, alpha)
        if not geo_TS:
            continue

        # Ablation direction call + T*
        abl_dir, abl_T, abl_relay = _ablation_call(pooled, a, b)
        if abl_dir != "A→B":
            continue
        if abl_T is None or abl_relay is None or abl_relay <= 0:
            continue

        # Conjunction: ablation argmax T must be in the geo-significant T set
        geo_T_map = dict(geo_TS)
        if abl_T not in geo_T_map:
            continue

        out.append({
            "A": a, "B": b, "T": abl_T,
            "p_fwd_bonf":               geo_T_map[abl_T],
            "ablation_relay":           abl_relay,
            "geo_cascade_call":         call,
            "ablation_direction_call":  abl_dir,
            "seed":                     exp_dir.name,
        })
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(
    per_seed_triples: List[dict],
    min_seeds: int,
) -> pd.DataFrame:
    by_triple: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in per_seed_triples:
        by_triple[(r["A"], r["B"], r["T"])].append(r)

    rows = []
    for (a, b, t), entries in by_triple.items():
        n = len(entries)
        if n < min_seeds:
            continue
        p_arr  = np.array([e["p_fwd_bonf"]    for e in entries])
        ab_arr = np.array([e["ablation_relay"] for e in entries])
        rows.append({
            "A": a, "B": b, "T": t,
            "n_seeds":                n,
            "p_fwd_bonf_median":      float(np.median(p_arr)),
            "p_fwd_bonf_min":         float(np.min(p_arr)),
            "p_fwd_bonf_max":         float(np.max(p_arr)),
            "ablation_relay_median":  float(np.median(ab_arr)),
            "ablation_relay_min":     float(np.min(ab_arr)),
            "ablation_relay_max":     float(np.max(ab_arr)),
            "geo_cascade_calls":      ",".join(e["geo_cascade_call"] for e in entries),
            "ablation_direction_calls": ",".join(e["ablation_direction_call"] for e in entries),
            "seeds_supporting":       " ".join(sorted(e["seed"] for e in entries)),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["n_seeds", "ablation_relay_median"],
                       ascending=[False, False], inplace=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Synthesize cascade triples from geo + ablation outputs."
    )
    p.add_argument("--exp_dirs", nargs="+", required=True,
                   help="Seed experiment directories (must each contain "
                        "latent_geo_results.pkl and ablation_scores_shard_*.pkl).")
    p.add_argument("--output", required=True,
                   help="Path to write cascade_triples.csv.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Geo p_fwd_bonf threshold (default 0.05).")
    p.add_argument("--min_seeds", type=int, default=5,
                   help="Minimum seeds in which a triple must appear (default 5 of 8).")
    p.add_argument("--ablation_subdir", type=str, default=".",
                   help="Subdirectory of each exp_dir holding "
                        "ablation_scores_shard_*.pkl (default '.': same dir as "
                        "latent_geo_results.pkl). Use 'ablation_union' for "
                        "union-pair-list ablations.")
    p.add_argument("--per_seed_dump", type=str, default=None,
                   help="Optional path to dump the raw per-seed triple list as JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    all_seed_triples: List[dict] = []
    for d in args.exp_dirs:
        seed_rows = _seed_triples(Path(d), alpha=args.alpha,
                                  ablation_subdir=args.ablation_subdir)
        print(f"  {Path(d).name:<30s} {len(seed_rows):>4d} triples (seed-level)",
              flush=True)
        all_seed_triples.extend(seed_rows)

    if args.per_seed_dump:
        Path(args.per_seed_dump).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_seed_dump, "w") as f:
            json.dump(all_seed_triples, f, indent=2)

    df = _aggregate(all_seed_triples, min_seeds=args.min_seeds)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nTotal seed-level triples       : {len(all_seed_triples)}", flush=True)
    print(f"Triples passing >= {args.min_seeds}/{len(args.exp_dirs)} seeds : {len(df)}",
          flush=True)
    print(f"Saved to                       : {out}", flush=True)
    if not df.empty:
        print("\nTop reportable triples:")
        cols = ["A", "B", "T", "n_seeds",
                "p_fwd_bonf_median", "ablation_relay_median"]
        print(df[cols].head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
