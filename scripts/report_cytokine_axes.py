"""
Reframe cascade_pairs_pool_then_call.csv as direction-agnostic cytokine axes.

Headline change vs the cascade reporter:
- A "cascade" is a directed (A → B) call. Our literature review showed
  directional inference is at chance (49% correct on 39 documented pairs;
  see reports/cascade_pairs/literature_review.md §8).
- An "axis" is an unordered cytokine pair {A, B} with a documented coupling
  in published immunology — direction may be A→B, B→A, bidirectional, or
  context-dependent.

This script reads `cascade_pairs_pool_then_call.csv` (output of
`scripts/report_cascade_pairs.py`) and produces `cytokine_axes.csv` with:

  - axis_a, axis_b              : unordered pair (a < b lexicographically)
  - axis_strength               : max(pooled_relay_fwd, pooled_relay_rev)
  - n_seeds_supporting          : n_seeds_a_to_b + n_seeds_b_to_a (out of 8)
  - relay_T_candidates          : top-3 most-frequent relay cell types
  - dominant_direction_in_model : "a->b", "b->a", or "tied"
  - literature_status           : KNOWN_DIRECTIONAL / KNOWN_COREGULATED /
                                  PARTIAL / NOVEL / NAME_AMBIGUOUS / PRE_REGISTERED
                                  (joined from literature_review_aggregate.json)
  - literature_direction        : the documented A→B direction in literature,
                                  if known; "reverse" if reversed; "bidir" if both
  - literature_summary          : 1–2 sentence evidence from the lit review
  - citations                   : ; separated list of papers

Args
----
--cascade_pairs_csv : output of report_cascade_pairs.py (pool_then_call variant)
--lit_review_json   : reports/cascade_pairs/literature_review_aggregate.json
--output            : cytokine_axes.csv
--min_seeds         : minimum n_fwd + n_rev across both directions (default 5)
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional


def _parse_t_distribution(s: str) -> Counter:
    """Parse the T_distribution_a_to_b string into a Counter of cell types."""
    if not s:
        return Counter()
    out = Counter()
    for token in s.split(","):
        idx = token.rfind(":")
        if idx < 0:
            continue
        ct = token[:idx].strip()
        n_str = token[idx + 1:].strip()
        try:
            out[ct] = int(n_str)
        except ValueError:
            continue
    return out


def _top_T_candidates(t_counter: Counter, k: int = 3) -> str:
    """Return top-k cell types as 'CT1:n1, CT2:n2, CT3:n3'."""
    top = t_counter.most_common(k)
    return ", ".join(f"{ct}:{n}" for ct, n in top)


def _canonicalize(a: str, b: str) -> tuple[str, str]:
    """Return (smaller, larger) lexicographically — direction-agnostic key."""
    return (a, b) if a <= b else (b, a)


# Pre-registered KNOWN_CASCADES (CLAUDE.md §16, scripts/analyze_inner_product_results.py).
_KNOWN_CASCADES_DIRECTIONAL = [
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
_KNOWN_CASCADES_UNORDERED = {frozenset({a, b}): (a, b) for a, b in _KNOWN_CASCADES_DIRECTIONAL}


def _build_lit_lookup(lit_json_path: Path) -> dict:
    """Index the lit review aggregate by canonical unordered pair."""
    if not lit_json_path.exists():
        return {}
    with open(lit_json_path) as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        # Lit review used directional (A, B); index by unordered pair and record
        # whether the model's A→B call matched the lit's A→B direction.
        key = frozenset({r["A"], r["B"]})
        # If multiple entries map to the same unordered pair (shouldn't happen
        # in our data — pool_then_call CSV is already canonical), keep the
        # stronger-classified one.
        ORDER = {"KNOWN_DIRECTIONAL": 0, "KNOWN_COREGULATED": 1, "PARTIAL": 2,
                 "NOVEL": 3, "NAME_AMBIGUOUS": 4}
        if key in out:
            if ORDER.get(r["classification"], 99) >= ORDER.get(out[key]["classification"], 99):
                continue
        out[key] = r
    return out


def _is_reverse_direction(lit_row: dict) -> bool:
    """Determine if the literature documents B→A rather than A→B (model's call)."""
    s = lit_row.get("evidence_summary", "").lower()
    A, B = lit_row["A"], lit_row["B"]
    keywords = [
        "reverse direction", "direction is reverse", "reverse-documented",
        "documented in reverse", "documented direction is reverse",
        "opposite direction", "reversed in literature",
        "the other way around", "actually inhibits",
    ]
    cause_phrases = [
        f"{B} induces {A}",
        f"{B} upregulates {A}",
        f"{B} drives {A}",
        f"{B} promotes {A}",
        f"{B} stimulates {A}",
        f"{B} → {A}",
        f"{B} -> {A}",
        f"{B}→{A}",
    ]
    return any(k in s for k in keywords) or any(p.lower() in s for p in cause_phrases)


def _is_bidirectional(lit_row: dict) -> bool:
    """Determine if the literature documents both A→B and B→A."""
    s = lit_row.get("evidence_summary", "").lower()
    return any(k in s for k in [
        "bidirectional", "both directions", "co-induced",
        "shared family", "coregulated", "co-regulated", "share the same",
    ])


def _classify_axis_direction(lit_row: Optional[dict]) -> str:
    """Map literature evidence into a direction tag relative to the canonical (A,B) we report."""
    if lit_row is None:
        return "no_lit"
    if lit_row["classification"] == "KNOWN_DIRECTIONAL":
        return "a_to_b"  # lit's A→B matches our canonical A→B
    if lit_row["classification"] == "KNOWN_COREGULATED":
        if _is_reverse_direction(lit_row):
            return "b_to_a"  # lit says B→A
        if _is_bidirectional(lit_row):
            return "bidir"
        return "coregulated_other"
    if lit_row["classification"] == "PARTIAL":
        return "partial_lit"
    if lit_row["classification"] == "NOVEL":
        return "no_lit"
    if lit_row["classification"] == "NAME_AMBIGUOUS":
        return "ambiguous"
    return "no_lit"


def build_axes(cascade_pairs_csv: Path, lit_review_json: Path, min_seeds: int) -> list[dict]:
    """Aggregate cascade pair rows into direction-agnostic axes."""
    lit_lookup = _build_lit_lookup(lit_review_json)
    axes = []
    with open(cascade_pairs_csv) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            a, b = row["A"], row["B"]
            n_fwd = int(row["n_seeds_a_to_b"])
            n_rev = int(row["n_seeds_b_to_a"])
            n_total = n_fwd + n_rev
            if n_total < min_seeds:
                continue
            pooled_fwd = float(row["pooled_relay"])
            pooled_rev = float(row["pooled_reverse_relay"])
            axis_strength = max(pooled_fwd, pooled_rev)
            if pooled_fwd > pooled_rev:
                model_dir = "a_to_b"
            elif pooled_rev > pooled_fwd:
                model_dir = "b_to_a"
            else:
                model_dir = "tied"

            t_dist = _parse_t_distribution(row.get("T_distribution_a_to_b", ""))
            top_T = _top_T_candidates(t_dist, k=3)

            # Canonicalize as unordered axis
            ax_a, ax_b = _canonicalize(a, b)

            key = frozenset({a, b})
            is_pre_registered = key in _KNOWN_CASCADES_UNORDERED
            lit_row = lit_lookup.get(key)
            if is_pre_registered:
                lit_status = "PRE_REGISTERED"
                # Pre-registered direction (a, b) where a is the canonical source.
                pre_a, pre_b = _KNOWN_CASCADES_UNORDERED[key]
                lit_direction = "a_to_b" if (pre_a, pre_b) == (ax_a, ax_b) else "b_to_a"
                lit_summary = "Pre-registered KNOWN_CASCADE."
                citations = ""
            elif lit_row is not None:
                lit_status = lit_row["classification"]
                lit_direction = _classify_axis_direction(lit_row)
                lit_summary = lit_row.get("evidence_summary", "")
                citations = "; ".join(lit_row.get("citations", []))
            else:
                lit_status = "UNCLASSIFIED"
                lit_direction = "no_lit"
                lit_summary = ""
                citations = ""

            axes.append({
                "axis_a": ax_a,
                "axis_b": ax_b,
                "axis_strength": axis_strength,
                "pooled_a_to_b": pooled_fwd,
                "pooled_b_to_a": pooled_rev,
                "n_seeds_a_to_b": n_fwd,
                "n_seeds_b_to_a": n_rev,
                "n_seeds_supporting": n_total,
                "relay_T_candidates": top_T,
                "dominant_direction_in_model": model_dir,
                "literature_status": lit_status,
                "literature_direction": lit_direction,
                "literature_summary": lit_summary[:300],
                "citations": citations[:500],
            })
    # Sort by strongest axes first
    axes.sort(key=lambda r: -r["axis_strength"])
    return axes


def write_csv(axes: list[dict], output_path: Path) -> None:
    if not axes:
        print(f"No axes met the threshold; not writing {output_path}")
        return
    fieldnames = list(axes[0].keys())
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in axes:
            w.writerow(r)


def summarize(axes: list[dict]) -> None:
    """Print a count summary of axes by literature status."""
    by_status = Counter(r["literature_status"] for r in axes)
    by_direction = Counter(r["literature_direction"] for r in axes)
    print(f"Total axes called: {len(axes)}")
    print("\nLiterature status counts:")
    for k in ["PRE_REGISTERED", "KNOWN_DIRECTIONAL", "KNOWN_COREGULATED",
              "PARTIAL", "NOVEL", "NAME_AMBIGUOUS", "UNCLASSIFIED"]:
        if by_status.get(k, 0):
            print(f"  {k:22s} : {by_status[k]:3d}")
    print("\nLiterature-direction tags (for axes with lit support):")
    for k in ["a_to_b", "b_to_a", "bidir", "coregulated_other",
              "partial_lit", "ambiguous", "no_lit"]:
        if by_direction.get(k, 0):
            print(f"  {k:22s} : {by_direction[k]:3d}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cascade_pairs_csv", required=True, type=Path,
                   help="Output CSV of report_cascade_pairs.py (pool_then_call variant).")
    p.add_argument("--lit_review_json", type=Path,
                   default=Path("reports/cascade_pairs/literature_review_aggregate.json"),
                   help="JSON aggregate from the literature review.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output cytokine_axes.csv path.")
    p.add_argument("--min_seeds", type=int, default=5,
                   help="Minimum sum n_fwd+n_rev (out of 8) for an axis to be reported.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    axes = build_axes(args.cascade_pairs_csv, args.lit_review_json, args.min_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(axes, args.output)
    summarize(axes)
    print(f"\nWrote {len(axes)} axes to {args.output}")


if __name__ == "__main__":
    main()
