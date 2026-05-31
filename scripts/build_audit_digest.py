"""
Build the per-axis evidence digest for the strict literature audit.

Steps:
1. Read `reports/cascade_pairs/literature_review_aggregate.json`.
2. Restrict to the 53 evaluable axes (derived from the full-19 pipeline output).
3. For each canonical pair {x, y} (x < y alphabetically), collect evidence
   separately for the x→y direction and the y→x direction. Multiple T-entries
   per direction are concatenated with "// T=<T>: ..." separators; citation
   lists are unioned.
4. Classify each citation as `primary` (PMC/PMID/biorxiv/major journal DOI)
   vs `secondary` (Wikipedia / R&D Systems / aggregator reviews). When in
   doubt classify as secondary — the POSITIVE_STRONG bar requires ≥1
   primary citation so a conservative classifier raises the audit bar.
5. Emit `reports/cascade_pairs/audit_digest.csv`.

Imports allowed: argparse, csv, json, sys, pathlib, collections, re, urllib.
NO pandas / yaml dependencies here — keeps the digest reproducible without
extra packages. Downstream scripts use pandas / pyyaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = REPO_ROOT / "reports/cascade_pairs/literature_review_aggregate.json"
DEFAULT_CSV  = REPO_ROOT / "reports/cascade_pairs/cytokine_axes.csv"
DEFAULT_PIPELINE = REPO_ROOT / "results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv"
DEFAULT_OUT  = REPO_ROOT / "reports/cascade_pairs/audit_digest.csv"


# --------------------------------------------------------------------------- #
# Citation typing
# --------------------------------------------------------------------------- #

# Host substrings that indicate primary research.  Conservative — when in
# doubt, classify as secondary (raises the POSITIVE_STRONG bar).
PRIMARY_HOSTS = (
    "pmc.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov/pmc",
    "journals.asm.org",
    "journals.plos.org",
    "biorxiv.org",
    "nature.com",
    "sciencedirect.com",
    "cell.com",
    "embopress.org",
    "elifesciences.org",
    "rupress.org",
    "academic.oup.com",
    "onlinelibrary.wiley.com",
    "jimmunol.org",
    "ashpublications.org",
    "rdcu.be",
)

SECONDARY_HOSTS = (
    "rndsystems.com",
    "wikipedia.org",
    "frontiersin.org",   # Frontiers is review-heavy in immunology; conservative
    "abcam.com",
    "biolegend.com",
    "thermofisher.com",
    "mdpi.com",          # MDPI mixed; conservative — let user upgrade individual cases
)


def classify_citation(url: str) -> str:
    """Return 'primary' or 'secondary' based on URL host."""
    url = url.strip().lower()
    if not url:
        return "secondary"
    # Try to extract the host
    try:
        # The JSON has citations like "Title — URL" so split on em-dash
        if " — " in url:
            url = url.split(" — ", 1)[1]
        host = urlparse(url).netloc.lower()
        if not host:
            # Fallback: search for substring
            host = url
    except Exception:
        host = url
    for h in PRIMARY_HOSTS:
        if h in host:
            return "primary"
    for h in SECONDARY_HOSTS:
        if h in host:
            return "secondary"
    return "secondary"


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_lit_aggregate(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def load_evaluable_axes(pipeline_csv: Path) -> set[tuple[str, str]]:
    """Return the set of canonical (axis_a, axis_b) pairs in the pipeline output."""
    axes = set()
    with pipeline_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            a, b = row["axis_a"], row["axis_b"]
            # Canonical: alphabetical ordering (the pipeline already uses this)
            ca, cb = sorted([a, b])
            axes.add((ca, cb))
    return axes


def load_original_axes(axes_csv: Path) -> dict[tuple[str, str], dict]:
    """Map canonical (axis_a, axis_b) → row dict from cytokine_axes.csv."""
    out: dict[tuple[str, str], dict] = {}
    with axes_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ca, cb = sorted([row["axis_a"], row["axis_b"]])
            out[(ca, cb)] = row
    return out


# --------------------------------------------------------------------------- #
# Build digest
# --------------------------------------------------------------------------- #

def build_digest(
    lit: list[dict],
    evaluable: set[tuple[str, str]],
    originals: dict[tuple[str, str], dict],
) -> list[dict]:
    """
    For each canonical pair in `evaluable`, collect evidence for both
    directions (A→B and B→A) from the literature JSON.

    Each lit entry has fields: A, B, T, classification, evidence_summary,
    citations (list), relay, reverse_relay, n_fwd, n_rev.

    The (A, B) in each entry is *directional* — A is upstream, B downstream.
    Multiple T entries for the same (A, B) are concatenated.
    """
    # bucket lit entries by directional (A, B) tuple
    by_dir: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in lit:
        by_dir[(e["A"], e["B"])].append(e)

    rows = []
    missing_in_lit = []

    for ca, cb in sorted(evaluable):
        a_to_b_entries = by_dir.get((ca, cb), [])
        b_to_a_entries = by_dir.get((cb, ca), [])

        # If we have zero entries in either direction, the pair isn't in the
        # JSON at all. Record it but mark explicitly.
        if not a_to_b_entries and not b_to_a_entries:
            missing_in_lit.append((ca, cb))

        a_to_b_summary, a_to_b_cits_primary, a_to_b_cits_secondary, a_to_b_T = \
            _collapse_direction(a_to_b_entries)
        b_to_a_summary, b_to_a_cits_primary, b_to_a_cits_secondary, b_to_a_T = \
            _collapse_direction(b_to_a_entries)

        # Pull the original-tag info from cytokine_axes.csv (one row per
        # canonical pair; the CSV already canonicalised so should match)
        orig = originals.get((ca, cb), {})
        rows.append({
            "axis_a": ca,
            "axis_b": cb,
            "original_classification": orig.get("literature_status", ""),
            "original_direction": orig.get("literature_direction", ""),
            "original_summary": orig.get("literature_summary", ""),
            "relay_T_candidates": orig.get("relay_T_candidates", ""),
            "a_to_b_n_lit_entries": len(a_to_b_entries),
            "a_to_b_T_list": a_to_b_T,
            "a_to_b_summary": a_to_b_summary,
            "a_to_b_citations_primary": " ||| ".join(a_to_b_cits_primary),
            "a_to_b_citations_secondary": " ||| ".join(a_to_b_cits_secondary),
            "a_to_b_n_primary": len(a_to_b_cits_primary),
            "a_to_b_n_secondary": len(a_to_b_cits_secondary),
            "b_to_a_n_lit_entries": len(b_to_a_entries),
            "b_to_a_T_list": b_to_a_T,
            "b_to_a_summary": b_to_a_summary,
            "b_to_a_citations_primary": " ||| ".join(b_to_a_cits_primary),
            "b_to_a_citations_secondary": " ||| ".join(b_to_a_cits_secondary),
            "b_to_a_n_primary": len(b_to_a_cits_primary),
            "b_to_a_n_secondary": len(b_to_a_cits_secondary),
        })

    if missing_in_lit:
        print(
            f"WARNING: {len(missing_in_lit)} evaluable axes have zero "
            f"lit-JSON entries in either direction:",
            file=sys.stderr,
        )
        for ca, cb in missing_in_lit:
            print(f"  {ca} / {cb}", file=sys.stderr)

    return rows


def _collapse_direction(entries: list[dict]) -> tuple[str, list[str], list[str], str]:
    """
    Collapse multiple T-entries for the same directional (A, B) into:
      - one summary string with "// T=<T>: ..." separators
      - union of primary citations (with title preserved)
      - union of secondary citations
      - comma-separated list of T values

    If `entries` is empty, returns ("", [], [], "").
    """
    if not entries:
        return ("", [], [], "")

    summary_parts = []
    primary_cits = []
    secondary_cits = []
    T_list = []
    seen_primary = set()
    seen_secondary = set()

    for e in entries:
        T = e.get("T", "?")
        summary = e.get("evidence_summary", "").strip()
        if summary:
            summary_parts.append(f"// T={T}: {summary}")
        T_list.append(T)
        for cit in e.get("citations", []):
            kind = classify_citation(cit)
            if kind == "primary":
                if cit not in seen_primary:
                    primary_cits.append(cit)
                    seen_primary.add(cit)
            else:
                if cit not in seen_secondary:
                    secondary_cits.append(cit)
                    seen_secondary.add(cit)

    return ("\n".join(summary_parts), primary_cits, secondary_cits, ", ".join(T_list))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lit_json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--axes_csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--pipeline_csv", type=Path, default=DEFAULT_PIPELINE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    for path in (args.lit_json, args.axes_csv, args.pipeline_csv):
        if not path.exists():
            print(f"FATAL: missing input: {path}", file=sys.stderr)
            sys.exit(2)

    lit = load_lit_aggregate(args.lit_json)
    evaluable = load_evaluable_axes(args.pipeline_csv)
    originals = load_original_axes(args.axes_csv)

    print(f"Loaded {len(lit)} lit-JSON entries", flush=True)
    print(f"Loaded {len(evaluable)} evaluable canonical pairs", flush=True)
    print(f"Loaded {len(originals)} rows from cytokine_axes.csv", flush=True)

    rows = build_digest(lit, evaluable, originals)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "axis_a", "axis_b",
        "original_classification", "original_direction",
        "original_summary", "relay_T_candidates",
        "a_to_b_n_lit_entries", "a_to_b_T_list", "a_to_b_summary",
        "a_to_b_n_primary", "a_to_b_n_secondary",
        "a_to_b_citations_primary", "a_to_b_citations_secondary",
        "b_to_a_n_lit_entries", "b_to_a_T_list", "b_to_a_summary",
        "b_to_a_n_primary", "b_to_a_n_secondary",
        "b_to_a_citations_primary", "b_to_a_citations_secondary",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out} ({len(rows)} rows)", flush=True)

    # Summary
    n_no_lit_either = sum(
        1 for r in rows if r["a_to_b_n_lit_entries"] == 0 and r["b_to_a_n_lit_entries"] == 0
    )
    n_lit_one_dir_only = sum(
        1 for r in rows
        if (r["a_to_b_n_lit_entries"] > 0) != (r["b_to_a_n_lit_entries"] > 0)
    )
    n_lit_both_dirs = sum(
        1 for r in rows if r["a_to_b_n_lit_entries"] > 0 and r["b_to_a_n_lit_entries"] > 0
    )
    print(f"  no lit entries either direction:   {n_no_lit_either}", flush=True)
    print(f"  lit entries one direction only:   {n_lit_one_dir_only}", flush=True)
    print(f"  lit entries both directions:      {n_lit_both_dirs}", flush=True)


if __name__ == "__main__":
    main()
