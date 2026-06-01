"""
Author + finalize the Immune Dictionary (Cui Nature 2024) cascade-direction labels
for the §26 cross_asym primary metric.

This is the §25.1 pre-registration artifact — committed BEFORE any audit run.

Stimulus names follow what the ID adapter writes into the manifest (see
scripts/build_pseudotubes_immune_dictionary.py).  The user-locked benchmark
cytokine set is: IL-1-beta, IL-6, TNF, IFN-beta, IFN-gamma, IL-12, IL-2,
IL-15, IL-18, IL-4, IL-13, IL-10.  If the adapter's actual cytokine column
uses different spelling (e.g., "IL-1b" or "IL1beta"), the binary trainer will
fail loudly on a missing target — which is the intended defensive behaviour.

Convention (mirrors finalize_sheu_labels.py / pipeline _expected_sign):
    expected_sign = +1  -> cascade is axis_a -> axis_b (a_to_b);
                          cross_asym(axis_a, axis_b) should be POSITIVE.
    expected_sign = -1  -> cascade is axis_b -> axis_a (b_to_a); negative.
    expected_sign =  0  -> NEGATIVE (no cascade, antagonistic); |cross_asym| ~ 0
                          DESCRIPTIVE per CLAUDE.md §26.4 ("direction not existence").
    expected_sign = None -> UNKNOWN / BIDIRECTIONAL; excluded from accuracy.

cross_asym(a,b) = s(a, S_b)_norm - s(b, S_a)_norm   (antisymmetric; §26.1).
A SIGN-ONLY metric; magnitude is NOT a coupling gate.  UNKNOWN/NEGATIVE pairs
are REPORTED descriptively in CASCADE_SWEEP_RESULTS.md, not scored.

This script emits:
    reports/immune_dictionary/id_axes_labeled.csv     (machine-readable; consumed
        by run_pipeline_a_bridge_b.py AND retally_pipeline_against_audit.py)
    reports/immune_dictionary/id_cascade_labels.yaml  (human-readable record)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "immune_dictionary"
YAML_PATH = OUT_DIR / "id_cascade_labels.yaml"
CSV_PATH = OUT_DIR / "id_axes_labeled.csv"

# Benchmark cytokines (12).  These are the ones binary models train on.
BENCHMARK_CYTOKINES = [
    "IFN-beta", "IFN-gamma", "IL-1-beta", "IL-10", "IL-12", "IL-13",
    "IL-15", "IL-18", "IL-2", "IL-4", "IL-6", "TNF",
]


# ----------------------------------------------------------------------- #
# Hand-curated labels. Key = canonical (axis_a, axis_b) with axis_a < axis_b.
# Lex sort:  "IFN-beta" < "IFN-gamma" < "IL-1-beta" < "IL-10" < "IL-12" <
#            "IL-13" < "IL-15" < "IL-18" < "IL-2" < "IL-4" < "IL-6" < "TNF"
# (ASCII: '-' (45) < '0' (48); so "IL-1-beta" < "IL-10".)
# ----------------------------------------------------------------------- #
LABELS = {
    # ---- 4 NK-cell-cluster IFN-gamma cascades: IL-{2,12,15,18} -> IFN-gamma
    # (canonical §2.7 paper-validated positive control).  In each, axis_a =
    # "IFN-gamma" < axis_b = the upstream cytokine, so cascade is b -> a,
    # expected_sign = -1.
    # ----
    ("IFN-gamma", "IL-12"): dict(
        pair_status="BIDIRECTIONAL", expected_sign=None,
        benchmark_class="BIDIRECTIONAL", counts_in_benchmark=False,
        upstream="",
        reason="Bidirectional positive feedback: IL-12 -> STAT4 -> IFN-gamma "
               "in NK/T, and IFN-gamma -> STAT1 -> Il12b in DCs/macs.  Direction "
               "is cell-type-dependent in vivo; excluded from signed accuracy "
               "per the user pre-reg.",
    ),
    ("IFN-gamma", "IL-15"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="NK_IFNG", counts_in_benchmark=True, upstream="IL-15",
        reason="IL-15 (and IL-2/IL-12/IL-18) drives NK-cell IFN-gamma production "
               "in vivo (§2.7 paper-validated NK cascade).  IL-15 upstream -> "
               "expected_sign = -1.",
    ),
    ("IFN-gamma", "IL-18"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="NK_IFNG", counts_in_benchmark=True, upstream="IL-18",
        reason="IL-18 synergizes with IL-12 to drive NK-cell IFN-gamma "
               "(§2.7 paper-validated cascade).  IL-18 upstream -> "
               "expected_sign = -1.",
    ),
    ("IFN-gamma", "IL-2"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="NK_IFNG", counts_in_benchmark=True, upstream="IL-2",
        reason="IL-2 supports NK-cell IFN-gamma production (§2.7 cascade).  "
               "IL-2 upstream -> expected_sign = -1.",
    ),
    # ---- NEGATIVE: Th2 antagonism (IL-4 inhibits Th1 IFN-gamma).  Descriptive only.
    ("IFN-gamma", "IL-4"): dict(
        pair_status="NEGATIVE_NO_CASCADE", expected_sign=0,
        benchmark_class="ANTAGONISM", counts_in_benchmark=False, upstream="",
        reason="Th2 cytokine IL-4 actively suppresses Th1 IFN-gamma program "
               "(antagonism, not cascade).  No directional cross-induction; "
               "|cross_asym| may still be large (signature distinctness), so "
               "this pair is DESCRIPTIVE only per §26.4.",
    ),
    # ---- Type-I IFN priming of NK cells for IFN-gamma production (a -> b)
    ("IFN-beta", "IFN-gamma"): dict(
        pair_status="DIRECTIONAL_a_to_b", expected_sign=+1,
        benchmark_class="TYPE1_IFN_PRIME", counts_in_benchmark=True,
        upstream="IFN-beta",
        reason="Type-I IFN primes NK cells for IFN-gamma production "
               "(Nguyen 2002 J Immunol, Mack 2011 J Exp Med).  IFN-beta upstream "
               "-> expected_sign = +1.  Distinct pathways: IFNAR/STAT1+STAT2 vs "
               "IFNGR/STAT1-only (GAS-driven); precondition holds.",
    ),
    # ---- NF-kB -> STAT3 cascades.  IL-1-beta -> IL-6 induction in myeloid cells.
    ("IL-1-beta", "IL-6"): dict(
        pair_status="DIRECTIONAL_a_to_b", expected_sign=+1,
        benchmark_class="NFKB_STAT3", counts_in_benchmark=True,
        upstream="IL-1-beta",
        reason="IL-1R -> MyD88 -> NF-kB induces IL-6 transcription in monos/macs "
               "(NF-kB targets Il6 promoter).  IL-6 then signals STAT3 -> distinct "
               "downstream program (gp130/JAK1).  IL-1-beta upstream -> +1.  "
               "Pathways distinct: NF-kB vs STAT3 -> precondition holds.",
    ),
    # ---- UNKNOWN: NF-kB <-> TNF overlap (known §24 failure mode, replicated here)
    ("IL-1-beta", "TNF"): dict(
        pair_status="UNKNOWN", expected_sign=None,
        benchmark_class="OVERLAP_NFKB", counts_in_benchmark=False, upstream="",
        reason="Both IL-1-beta and TNF signal through NF-kB; programs overlap "
               "heavily (§24 known failure mode replicated here).  cross_asym "
               "sign is unreliable when S_X collapse onto a shared activation "
               "axis.  DESCRIPTIVE only.",
    ),
    # ---- UNKNOWN: IL-10 -> IL-12 antagonism (actually negative cascade)
    ("IL-10", "IL-12"): dict(
        pair_status="NEGATIVE_NO_CASCADE", expected_sign=0,
        benchmark_class="ANTAGONISM", counts_in_benchmark=False, upstream="",
        reason="IL-10 -> STAT3 actively SUPPRESSES Il12b transcription "
               "(Moore 2001 Annu Rev Immunol).  Negative cascade -> any sign of "
               "cross_asym would still be biologically interpretable (suppression "
               "engages STAT3 targets in macs); DESCRIPTIVE per §26.4.",
    ),
    # ---- UNKNOWN: IL-13 / IL-4 overlap (both STAT6 via IL-4Ralpha)
    ("IL-13", "IL-4"): dict(
        pair_status="UNKNOWN", expected_sign=None,
        benchmark_class="OVERLAP_STAT6", counts_in_benchmark=False, upstream="",
        reason="IL-4 and IL-13 both signal via STAT6 through type-II IL-4R "
               "(IL-4Ralpha + IL-13Ralpha1).  Programs overlap heavily; sign of "
               "cross_asym is unreliable for the same reason as IL-1b/TNF.  "
               "DESCRIPTIVE.",
    ),
    # ---- UNKNOWN: IL-2 / IL-15 overlap (both common gamma-chain -> STAT5)
    ("IL-15", "IL-2"): dict(
        pair_status="UNKNOWN", expected_sign=None,
        benchmark_class="OVERLAP_STAT5", counts_in_benchmark=False, upstream="",
        reason="IL-2 and IL-15 both signal through common gamma-chain (CD132) "
               "and STAT5; gene programs overlap heavily.  cross_asym sign "
               "unreliable here.  DESCRIPTIVE.",
    ),
    # ---- TNF -> IL-6 (NF-kB -> STAT3, b -> a since alphabetically IL-6 < TNF)
    ("IL-6", "TNF"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="NFKB_STAT3", counts_in_benchmark=True, upstream="TNF",
        reason="TNF -> TNFR1 -> NF-kB induces IL-6 transcription in myeloid "
               "cells (NF-kB targets Il6 promoter).  IL-6 signals STAT3.  "
               "TNF upstream -> expected_sign = -1.  NF-kB vs STAT3 distinct -> "
               "precondition holds (in contrast to IL-1b/TNF which share NF-kB).",
    ),
}


def _expected_to_lit_dir(pair_status: str) -> str:
    if pair_status == "DIRECTIONAL_a_to_b":
        return "a_to_b"
    if pair_status == "DIRECTIONAL_b_to_a":
        return "b_to_a"
    if pair_status == "BIDIRECTIONAL":
        return "bidir"
    return "no_lit"


def validate() -> None:
    # 1. all axis_a < axis_b lexicographically
    # 2. expected_sign consistent with pair_status
    # 3. counts_in_benchmark consistent with pair_status
    # 4. all cytokines in the keys belong to BENCHMARK_CYTOKINES (so binary
    #    models will exist)
    bench_set = set(BENCHMARK_CYTOKINES)
    for (a, b), d in LABELS.items():
        if a >= b:
            print(f"FATAL: non-canonical order: {(a, b)}", file=sys.stderr)
            sys.exit(2)
        if a not in bench_set:
            print(f"FATAL: {a!r} not in BENCHMARK_CYTOKINES", file=sys.stderr)
            sys.exit(2)
        if b not in bench_set:
            print(f"FATAL: {b!r} not in BENCHMARK_CYTOKINES", file=sys.stderr)
            sys.exit(2)
        ps, sign = d["pair_status"], d["expected_sign"]
        if ps == "DIRECTIONAL_a_to_b" and sign != +1:
            print(f"FATAL: {a}/{b} a_to_b but sign {sign}", file=sys.stderr)
            sys.exit(2)
        if ps == "DIRECTIONAL_b_to_a" and sign != -1:
            print(f"FATAL: {a}/{b} b_to_a but sign {sign}", file=sys.stderr)
            sys.exit(2)
        if ps.startswith("DIRECTIONAL") and not d["counts_in_benchmark"]:
            print(f"FATAL: {a}/{b} directional but not in benchmark", file=sys.stderr)
            sys.exit(2)
        if ps == "NEGATIVE_NO_CASCADE" and (sign != 0 or d["counts_in_benchmark"]):
            print(f"FATAL: {a}/{b} negative misconfigured", file=sys.stderr)
            sys.exit(2)
        if ps in ("UNKNOWN", "BIDIRECTIONAL") and (
            sign is not None or d["counts_in_benchmark"]
        ):
            print(f"FATAL: {a}/{b} {ps} misconfigured", file=sys.stderr)
            sys.exit(2)


def emit_csv() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "axis_a", "axis_b", "literature_status", "literature_direction",
        "expected_sign", "pair_status", "benchmark_class",
        "counts_in_benchmark", "upstream", "original_direction",
        "tag_changed", "reason",
    ]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (a, b), d in sorted(LABELS.items()):
            lit_dir = _expected_to_lit_dir(d["pair_status"])
            w.writerow({
                "axis_a": a, "axis_b": b,
                "literature_status": d["benchmark_class"],
                "literature_direction": lit_dir,
                "expected_sign": "" if d["expected_sign"] is None else d["expected_sign"],
                "pair_status": d["pair_status"],
                "benchmark_class": d["benchmark_class"],
                "counts_in_benchmark": d["counts_in_benchmark"],
                "upstream": d["upstream"],
                "original_direction": lit_dir,
                "tag_changed": False,
                "reason": d["reason"],
            })
    print(f"Wrote {CSV_PATH} ({len(LABELS)} labeled pairs)")


def emit_yaml() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Immune Dictionary cascade-direction labels (§26 cross_asym primary metric).",
        "# Benchmark cytokines (binary models train on these only):",
        "#   " + ", ".join(BENCHMARK_CYTOKINES),
        "# expected_sign: +1 a_to_b, -1 b_to_a, 0 NEGATIVE (no cascade), null UNKNOWN/BIDIR.",
        "labels:",
    ]
    for (a, b), d in sorted(LABELS.items()):
        sign = "null" if d["expected_sign"] is None else d["expected_sign"]
        lines.append(f"  - axis: [{a}, {b}]")
        lines.append(f"    pair_status: {d['pair_status']}")
        lines.append(f"    expected_sign: {sign}")
        lines.append(f"    benchmark_class: {d['benchmark_class']}")
        lines.append(f"    counts_in_benchmark: {str(d['counts_in_benchmark']).lower()}")
        lines.append(f"    upstream: {d['upstream'] or 'null'}")
        lines.append(f"    reason: \"{d['reason']}\"")
    YAML_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {YAML_PATH}")


def main() -> None:
    validate()
    emit_yaml()
    emit_csv()
    # Tally
    n_bench = sum(1 for d in LABELS.values() if d["counts_in_benchmark"])
    by_class = {}
    for d in LABELS.values():
        by_class[d["benchmark_class"]] = by_class.get(d["benchmark_class"], 0) + 1
    print(f"\nbenchmark (directional, counts_in_benchmark=True): {n_bench}")
    print("by benchmark_class:")
    for c in sorted(by_class):
        print(f"  {c:18s} {by_class[c]}")


if __name__ == "__main__":
    main()
