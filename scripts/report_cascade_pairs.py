"""
Pair-level cascade reporting (the natural unit when the relay T floats across
seeds but the pair direction is stable).

Two aggregation strategies are produced side-by-side:

per_seed_then_count
    For each (A, B), count the seeds in which ablation's direction call is
    "A→B" with a positive argmax relay. Report pairs with `n_seeds >=
    --min_seeds`. Each seed contributes its own argmax T → emit a frequency
    distribution over the relay cell types observed in those seeds.

pool_then_call
    Concatenate ablation relay scores across all seeds first (so each
    `(src, tgt, ct)` key gets one long list of per-tube relay scores pooled
    over seeds), then call direction by `max_T mean(pooled_fwd) vs
    max_T mean(pooled_rev)`. Report pairs whose pooled forward mean is
    positive and strictly greater than the pooled reverse mean. Emit one
    pooled-argmax T per pair plus the per-seed T-distribution for context.

Both strategies are emitted independently to two CSVs so they can be
compared. They are intentionally orthogonal: the first measures
reproducibility of seed-level calls; the second measures the strength of
the cascade signal in the pooled data.

Inputs
------
--exp_dirs        : list of seed directories. Each must contain
                    ablation_scores_shard_*.pkl (or fallback ablation_scores.pkl)
                    inside `--ablation_subdir` (default '.': same dir).
--output_dir      : where to write cascade_pairs_*.csv.
--ablation_subdir : subdir under each exp_dir. Use 'ablation_union' for
                    union-pair-list ablations.
--min_seeds       : minimum seeds with an A→B call for the per-seed strategy
                    (default 5).
--known_csv       : optional path that, if provided, emits a marker column
                    `known_cascade=True/False` against the 11 KNOWN_CASCADES.

Outputs
-------
cascade_pairs_per_seed_then_count.csv
cascade_pairs_pool_then_call.csv

Columns (common):
  A, B
  n_seeds_a_to_b : seeds where direction call was A→B with positive relay
  n_seeds_b_to_a : seeds where the reverse direction was called
  n_seeds_total  : seeds with any data for this (A, B)
  T_distribution_a_to_b : "NK:5,MAIT:2,CD8 T:1" — sorted by frequency desc

Additional in per_seed_then_count.csv:
  best_T_by_freq : top entry from T_distribution_a_to_b
  median_relay   : median of per-seed argmax mean relays where call was A→B

Additional in pool_then_call.csv:
  pooled_best_T  : argmax T of mean across pooled per-tube relay scores
  pooled_relay   : value of that argmax mean
  pooled_reverse_relay : max mean for the reverse direction
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


KNOWN_CASCADES = [
    ("IL-12",      "IFN-gamma"),
    ("IL-1-beta",  "IL-6"),
    ("IL-2",       "IL-15"),
    ("IL-33",      "IL-13"),
    ("IL-18",      "IFN-gamma"),
    ("IL-21",      "IL-10"),
    ("TNF-alpha",  "IL-6"),
    ("IFN-alpha1", "IFN-gamma"),
    ("IL-10",      "IL-6"),
    ("IL-4",       "IL-13"),
    ("IL-27",      "IFN-gamma"),
]
KNOWN_UNORDERED = {frozenset(p) for p in KNOWN_CASCADES}


# ---------------------------------------------------------------------------
# Ablation loaders
# ---------------------------------------------------------------------------

def _load_pooled(exp_dir: Path) -> Dict[Tuple[str, str, str], list]:
    """Combine all ablation_scores_shard_*.pkl in `exp_dir` into one pooled dict."""
    pooled: Dict[Tuple[str, str, str], list] = defaultdict(list)
    shard_paths = sorted(exp_dir.glob("ablation_scores_shard_*.pkl"))
    if not shard_paths:
        single = exp_dir / "ablation_scores.pkl"
        if single.exists():
            shard_paths = [single]
    for p in shard_paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        for key, vals in data.get("pooled", {}).items():
            pooled[key].extend(vals)
    return dict(pooled)


def _ordered_pairs(pooled: Dict[Tuple[str, str, str], list]) -> List[Tuple[str, str]]:
    return sorted({(s, t) for (s, t, _ct) in pooled})


def _seed_call(pooled: Dict[Tuple[str, str, str], list],
               a: str, b: str) -> Tuple[str, Optional[str], Optional[float]]:
    """(direction, argmax_T, relay_at_argmax) for one seed.

    direction ∈ {'A→B', 'B→A', 'shared', 'no_data'}.
    """
    fwd = {ct: float(np.mean(v))
           for (s, t, ct), v in pooled.items()
           if s == a and t == b and len(v) > 0}
    rev = {ct: float(np.mean(v))
           for (s, t, ct), v in pooled.items()
           if s == b and t == a and len(v) > 0}
    if not fwd and not rev:
        return ("no_data", None, None)
    best_fwd = max(fwd.values()) if fwd else -np.inf
    best_rev = max(rev.values()) if rev else -np.inf
    if best_fwd > best_rev:
        return ("A→B", max(fwd, key=fwd.get), best_fwd)
    if best_rev > best_fwd:
        return ("B→A", max(rev, key=rev.get), best_rev)
    return ("shared", None, None)


# ---------------------------------------------------------------------------
# Strategy 1: per_seed_then_count
# ---------------------------------------------------------------------------

def _per_seed_then_count(
    per_seed_pooled: Dict[str, Dict[Tuple[str, str, str], list]],
    min_seeds: int,
) -> pd.DataFrame:
    """For each (A, B), count seeds calling A→B (with positive relay) and
    aggregate the argmax T values observed in those seeds.
    """
    # Use the union of ordered pairs seen across all seeds.
    all_pairs: set = set()
    for pooled in per_seed_pooled.values():
        all_pairs.update(_ordered_pairs(pooled))

    # Canonicalize: report each unordered pair once. Pick the direction that
    # has more seeds calling it as A→B.
    canon: Dict[Tuple[str, str], dict] = {}
    for (a, b) in all_pairs:
        key = (a, b) if a <= b else (b, a)
        if key not in canon:
            canon[key] = {
                "n_fwd_calls": Counter(),  # A=key[0], B=key[1]: seeds calling A→B
                "n_rev_calls": Counter(),
                "T_fwd": Counter(),
                "T_rev": Counter(),
                "fwd_relays": [],
                "rev_relays": [],
                "seeds_total": 0,
            }

    for seed_name, pooled in per_seed_pooled.items():
        seen_canon = set()
        for (a, b) in _ordered_pairs(pooled):
            key = (a, b) if a <= b else (b, a)
            if key in seen_canon:
                continue
            seen_canon.add(key)
            # Use the canonical (A, B) for the per-seed call so direction is
            # always measured in the canonical orientation.
            d, T, relay = _seed_call(pooled, key[0], key[1])
            canon[key]["seeds_total"] += 1
            if d == "A→B" and relay is not None and relay > 0 and T is not None:
                canon[key]["n_fwd_calls"][seed_name] = 1
                canon[key]["T_fwd"][T] += 1
                canon[key]["fwd_relays"].append(relay)
            elif d == "B→A" and relay is not None and relay > 0 and T is not None:
                canon[key]["n_rev_calls"][seed_name] = 1
                canon[key]["T_rev"][T] += 1
                canon[key]["rev_relays"].append(relay)

    rows = []
    for (a, b), data in canon.items():
        n_fwd = sum(data["n_fwd_calls"].values())
        n_rev = sum(data["n_rev_calls"].values())
        # Report a row only if at least one direction reaches min_seeds.
        if n_fwd < min_seeds and n_rev < min_seeds:
            continue
        # Choose the dominant direction; emit both for transparency.
        if n_fwd >= n_rev:
            T_dist = data["T_fwd"]
            relays = data["fwd_relays"]
            direction = "A→B"
            primary_a, primary_b = a, b
        else:
            T_dist = data["T_rev"]
            relays = data["rev_relays"]
            direction = "B→A"
            primary_a, primary_b = b, a

        T_str = ",".join(f"{ct}:{n}" for ct, n in T_dist.most_common())
        rows.append({
            "A": primary_a,
            "B": primary_b,
            "n_seeds_a_to_b": (n_fwd if direction == "A→B" else n_rev),
            "n_seeds_b_to_a": (n_rev if direction == "A→B" else n_fwd),
            "n_seeds_total":  data["seeds_total"],
            "T_distribution_a_to_b": T_str,
            "best_T_by_freq": T_dist.most_common(1)[0][0] if T_dist else None,
            "median_relay":   float(np.median(relays)) if relays else None,
            "known_cascade":  frozenset({a, b}) in KNOWN_UNORDERED,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["n_seeds_a_to_b", "median_relay"],
                       ascending=[False, False], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Strategy 2: pool_then_call
# ---------------------------------------------------------------------------

def _pool_across_seeds(
    per_seed_pooled: Dict[str, Dict[Tuple[str, str, str], list]],
) -> Dict[Tuple[str, str, str], list]:
    """Concatenate per-tube relay-score lists across seeds, keyed by
    (src, tgt, ct)."""
    big: Dict[Tuple[str, str, str], list] = defaultdict(list)
    for pooled in per_seed_pooled.values():
        for key, vals in pooled.items():
            big[key].extend(vals)
    return dict(big)


def _pool_then_call(
    per_seed_pooled: Dict[str, Dict[Tuple[str, str, str], list]],
    min_seeds: int,
) -> pd.DataFrame:
    """Pool relay scores across seeds first, then call direction + argmax T
    on the pooled distribution.

    Per-seed T-distribution is reported alongside as context.
    """
    big = _pool_across_seeds(per_seed_pooled)
    all_pairs = _ordered_pairs(big)
    seen_canon = set()
    rows = []
    for (a, b) in all_pairs:
        key = (a, b) if a <= b else (b, a)
        if key in seen_canon:
            continue
        seen_canon.add(key)

        d_pool, T_pool, relay_pool = _seed_call(big, key[0], key[1])
        if d_pool == "no_data":
            continue
        # Reverse relay for reporting symmetry
        rev_pool = {ct: float(np.mean(v))
                    for (s, t, ct), v in big.items()
                    if s == key[1] and t == key[0] and len(v) > 0}
        rev_pool_best = max(rev_pool.values()) if rev_pool else -np.inf

        # Per-seed T-distribution in the canonical-A→B direction
        T_fwd_dist: Counter = Counter()
        T_rev_dist: Counter = Counter()
        n_fwd = n_rev = 0
        total_seeds = 0
        for pooled in per_seed_pooled.values():
            seed_pairs = {(s, t) for (s, t, _) in pooled}
            if (key[0], key[1]) not in seed_pairs and (key[1], key[0]) not in seed_pairs:
                continue
            total_seeds += 1
            ds, Ts, _ = _seed_call(pooled, key[0], key[1])
            if ds == "A→B" and Ts is not None:
                T_fwd_dist[Ts] += 1
                n_fwd += 1
            elif ds == "B→A" and Ts is not None:
                T_rev_dist[Ts] += 1
                n_rev += 1

        if d_pool == "A→B":
            primary_a, primary_b = key[0], key[1]
            T_str = ",".join(f"{ct}:{n}" for ct, n in T_fwd_dist.most_common())
            n_fwd_out, n_rev_out = n_fwd, n_rev
            forward_relay = relay_pool
            reverse_relay = rev_pool_best
        elif d_pool == "B→A":
            primary_a, primary_b = key[1], key[0]
            T_str = ",".join(f"{ct}:{n}" for ct, n in T_rev_dist.most_common())
            n_fwd_out, n_rev_out = n_rev, n_fwd
            forward_relay = relay_pool
            reverse_relay = rev_pool_best
        else:  # shared
            continue

        # Gate on pooled-positive forward relay
        if forward_relay is None or forward_relay <= 0:
            continue
        # Optional sanity: at least min_seeds had relevant data
        if total_seeds < min_seeds:
            continue

        rows.append({
            "A": primary_a,
            "B": primary_b,
            "pooled_best_T":         T_pool,
            "pooled_relay":          float(forward_relay),
            "pooled_reverse_relay":  float(reverse_relay) if np.isfinite(reverse_relay) else None,
            "n_seeds_a_to_b":        n_fwd_out,
            "n_seeds_b_to_a":        n_rev_out,
            "n_seeds_total":         total_seeds,
            "T_distribution_a_to_b": T_str,
            "known_cascade":         frozenset({primary_a, primary_b}) in KNOWN_UNORDERED,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["pooled_relay", "n_seeds_a_to_b"],
                       ascending=[False, False], inplace=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--exp_dirs", nargs="+", required=True,
                   help="Seed experiment directories.")
    p.add_argument("--output_dir", required=True,
                   help="Where to write the two cascade_pairs_*.csv files.")
    p.add_argument("--ablation_subdir", type=str, default=".",
                   help="Subdir under each exp_dir holding "
                        "ablation_scores_shard_*.pkl (default '.').")
    p.add_argument("--min_seeds", type=int, default=5,
                   help="Minimum supporting seeds for a row to be emitted "
                        "(default 5).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_pooled: Dict[str, Dict[Tuple[str, str, str], list]] = {}
    for d in args.exp_dirs:
        seed_dir = Path(d) / args.ablation_subdir
        pooled = _load_pooled(seed_dir)
        if not pooled:
            print(f"  WARN: empty pooled for {seed_dir}", flush=True)
            continue
        per_seed_pooled[Path(d).name] = pooled

    if not per_seed_pooled:
        raise SystemExit("No ablation data loaded from any seed.")

    print(f"Loaded {len(per_seed_pooled)} seeds.", flush=True)

    df_count = _per_seed_then_count(per_seed_pooled, min_seeds=args.min_seeds)
    df_pool  = _pool_then_call(per_seed_pooled,     min_seeds=args.min_seeds)

    df_count.to_csv(out_dir / "cascade_pairs_per_seed_then_count.csv", index=False)
    df_pool.to_csv (out_dir / "cascade_pairs_pool_then_call.csv",     index=False)

    print(f"\nper_seed_then_count : {len(df_count)} pairs >= {args.min_seeds}/8 seeds",
          flush=True)
    if not df_count.empty:
        print(df_count.head(20).to_string(index=False), flush=True)
        n_known = int(df_count["known_cascade"].sum())
        print(f"  -> {n_known} KNOWN_CASCADES recovered", flush=True)

    print(f"\npool_then_call      : {len(df_pool)} pairs with positive pooled forward relay",
          flush=True)
    if not df_pool.empty:
        print(df_pool.head(20).to_string(index=False), flush=True)
        n_known = int(df_pool["known_cascade"].sum())
        print(f"  -> {n_known} KNOWN_CASCADES recovered", flush=True)


if __name__ == "__main__":
    main()
