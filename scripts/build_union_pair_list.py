"""
Merge alignment and geo pair-detection outputs into a single union pair list
for the ablation pipeline.

A "pair" here is *unordered*: the ablation script
(scripts/analyze_cell_type_ablation.py, line 196) tests both (A→B) and (B→A)
internally per entry, so each unordered pair should appear exactly once in the
union list. We canonicalize as (A, B) with A < B lexicographically.

Inputs
------
--alignment_pairs_file : path to top_pairs_<metric>_<dim>.json emitted by
    scripts/run_inner_product_pairs.py. Schema: list of
    {A, B, relay_cell_type, score, rank}. May be omitted (geo-only union).

--alignment_top_k : optional cap on the number of alignment pairs to include
    (sorted by rank ascending). Useful because run_inner_product_pairs.py
    defaults to top_pct=0.20 (~800 pairs of 4005), but ablation budget on
    "short" partition only fits ~200-300 unordered pairs across 4 shards.
    See recall_table.csv to pick a top_pct that matches your budget.

--geo_results : path to latent_geo_results.pkl emitted by
    scripts/run_geo_extract.py. Reads sig_result["cascade_call"] and
    sig_result["relay_T"]. May be omitted (alignment-only union).

--output : path to write the merged top_pairs_union.json.

Output schema
-------------
List of entries with keys:
    A, B                  : canonicalized lexicographic order
    relay_cell_type       : best relay (geo if available, else alignment)
    source                : "alignment" | "geo" | "both"
    alignment_score       : float | None
    alignment_rank        : int | None
    geo_cascade_call      : "A->B" | "B->A" | "shared" | "none" | None
    geo_relay_T           : str | None
    geo_p_fwd_bonf_min    : float | None  (min Bonferroni p across cell types
                                           for the canonical A→B direction)

This file is consumed directly by `analyze_cell_type_ablation.py --pairs_file`.
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple


def _canonical(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _load_alignment(path: Path, top_k: Optional[int] = None) -> Dict[Tuple[str, str], dict]:
    """Return {(A, B canonical) -> {alignment_score, alignment_rank, relay_T}}.

    If `top_k` is given, only the top_k entries by rank ascending are kept.
    """
    with open(path) as f:
        rows = json.load(f)
    if top_k is not None:
        rows = sorted(rows, key=lambda r: r["rank"])[:top_k]
    out: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        key = _canonical(r["A"], r["B"])
        # Keep the best rank if duplicated.
        if key in out and out[key]["alignment_rank"] <= r["rank"]:
            continue
        out[key] = {
            "alignment_score": float(r["score"]),
            "alignment_rank":  int(r["rank"]),
            "alignment_relay_T": r.get("relay_cell_type"),
        }
    return out


def _load_geo(path: Path) -> Dict[Tuple[str, str], dict]:
    """Return {(A, B canonical) -> {cascade_call, relay_T, p_fwd_bonf_min}}.

    Excludes cascade_call == "none". For canonical (A, B), we report the
    cascade_call as seen for the ordered pair (A, B) (the relay_T and p come
    from that ordered pair). If the geo result only stored the reverse ordered
    pair (B, A), we flip the call label so the canonical entry reflects the
    canonical (A, B) direction.
    """
    with open(path, "rb") as f:
        geo = pickle.load(f)
    sig = geo["sig_result"]

    cascade_call: Dict[Tuple[str, str], str] = sig["cascade_call"]
    relay_T: Dict[Tuple[str, str], Optional[str]] = sig["relay_T"]
    p_fwd_bonf: Dict[Tuple[str, str, str], float] = sig["p_fwd_bonf"]

    # Minimum p_fwd_bonf across cell types per ordered pair.
    p_pair_fwd_min: Dict[Tuple[str, str], float] = defaultdict(lambda: 1.0)
    for (a, b, _ct), p in p_fwd_bonf.items():
        if p < p_pair_fwd_min[(a, b)]:
            p_pair_fwd_min[(a, b)] = float(p)

    out: Dict[Tuple[str, str], dict] = {}
    for (a, b), call in cascade_call.items():
        if call == "none":
            continue
        key = _canonical(a, b)
        if (a, b) == key:
            canonical_call = call
        else:
            # We stored the (B, A) entry; flip the call to reflect canonical.
            canonical_call = (
                "B->A" if call == "A->B"
                else "A->B" if call == "B->A"
                else call  # "shared" stays "shared"
            )
        record = {
            "geo_cascade_call":  canonical_call,
            "geo_relay_T":       relay_T.get((a, b)),
            "geo_p_fwd_bonf_min": p_pair_fwd_min.get((a, b)),
        }
        # If both ordered entries exist, prefer the record that actually carries
        # forward-Wilcoxon data (non-None p_fwd_bonf_min). Among informative
        # records, prefer the smaller p.
        if key in out:
            prev_p = out[key].get("geo_p_fwd_bonf_min")
            cur_p  = record["geo_p_fwd_bonf_min"]
            if prev_p is not None and cur_p is None:
                continue
            if (prev_p is not None and cur_p is not None
                    and prev_p <= cur_p):
                continue
        out[key] = record
    return out


def _merge(
    align: Dict[Tuple[str, str], dict],
    geo:   Dict[Tuple[str, str], dict],
) -> list:
    keys = set(align) | set(geo)
    rows = []
    for key in keys:
        a, b = key
        in_a = key in align
        in_g = key in geo
        source = "both" if (in_a and in_g) else ("alignment" if in_a else "geo")
        entry = {
            "A": a,
            "B": b,
            "source": source,
            "alignment_score":   align[key]["alignment_score"]  if in_a else None,
            "alignment_rank":    align[key]["alignment_rank"]   if in_a else None,
            "geo_cascade_call":  geo[key]["geo_cascade_call"]   if in_g else None,
            "geo_relay_T":       geo[key]["geo_relay_T"]        if in_g else None,
            "geo_p_fwd_bonf_min": geo[key]["geo_p_fwd_bonf_min"] if in_g else None,
        }
        # Preferred relay: geo when available (statistical evidence), else alignment.
        entry["relay_cell_type"] = (
            entry["geo_relay_T"]
            if entry["geo_relay_T"] is not None
            else (align[key]["alignment_relay_T"] if in_a else None)
        )
        rows.append(entry)
    # Sort: geo-significant first (lowest p), then alignment rank.
    def _sort_key(r):
        p = r["geo_p_fwd_bonf_min"] if r["geo_p_fwd_bonf_min"] is not None else 1.0
        rk = r["alignment_rank"] if r["alignment_rank"] is not None else 10**9
        return (p, rk)
    rows.sort(key=_sort_key)
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--alignment_pairs_file", type=str, default=None,
                   help="Path to top_pairs_<metric>_<dim>.json from alignment pipeline.")
    p.add_argument("--alignment_top_k", type=int, default=None,
                   help="Optional cap: keep only the top_k alignment entries by rank.")
    p.add_argument("--geo_results", type=str, default=None,
                   help="Path to latent_geo_results.pkl from run_geo_extract.py.")
    p.add_argument("--output", type=str, required=True,
                   help="Path to write top_pairs_union.json.")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.alignment_pairs_file and not args.geo_results:
        raise SystemExit("Provide at least one of --alignment_pairs_file or --geo_results.")

    align = (
        _load_alignment(Path(args.alignment_pairs_file), top_k=args.alignment_top_k)
        if args.alignment_pairs_file else {}
    )
    geo   = _load_geo(Path(args.geo_results)) if args.geo_results else {}

    rows = _merge(align, geo)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    n_total = len(rows)
    n_both  = sum(1 for r in rows if r["source"] == "both")
    n_align = sum(1 for r in rows if r["source"] == "alignment")
    n_geo   = sum(1 for r in rows if r["source"] == "geo")
    print(f"Alignment pairs : {len(align)}", flush=True)
    print(f"Geo pairs       : {len(geo)}", flush=True)
    print(f"Union pairs     : {n_total} "
          f"(both: {n_both}, alignment-only: {n_align}, geo-only: {n_geo})",
          flush=True)
    print(f"Saved to        : {out_path}", flush=True)


if __name__ == "__main__":
    main()
