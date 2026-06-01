"""
Author + finalize the Immune Dictionary (Cui Nature 2024) cascade-direction labels
for the cross_asym primary metric (CLAUDE.md section 26).

This is the section-25.1 pre-registration artifact -- committed BEFORE any audit run.

Stimulus names are the SCP `cyt` machine names (verified present in SCP2554 via
the public API; see data/immune_dictionary_scp_metadata.parquet):
    IFNb, IFNg, IL1b, IL10, IL12, IL13, IL15, IL18, IL2, IL4, IL6, TNFa
The pseudo-tube `cytokine` column uses exactly these strings, so the label CSV,
the binary trainer target list, and the manifest are all consistent.

Convention (mirrors finalize_sheu_labels.py / pipeline _expected_sign):
    expected_sign = +1  -> cascade is axis_a -> axis_b (a_to_b); cross_asym positive.
    expected_sign = -1  -> cascade is axis_b -> axis_a (b_to_a); negative.
    expected_sign =  0  -> NEGATIVE (no cascade / antagonistic); DESCRIPTIVE (sec 26.4).
    expected_sign = None -> UNKNOWN / BIDIRECTIONAL; excluded from accuracy.

cross_asym(a,b) = s(a, S_b)_norm - s(b, S_a)_norm   (antisymmetric; sec 26.1).
A SIGN-ONLY metric; magnitude is NOT a coupling gate.  UNKNOWN/NEGATIVE pairs are
REPORTED descriptively in CASCADE_SWEEP_RESULTS.md, not scored.

Emits:
    reports/immune_dictionary/id_axes_labeled.csv     (machine-readable; consumed by
        run_pipeline_a_bridge_b.py AND retally_pipeline_against_audit.py)
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

# Benchmark cytokines (12).  SCP `cyt` machine names.
BENCHMARK_CYTOKINES = [
    "IFNb", "IFNg", "IL1b", "IL10", "IL12", "IL13",
    "IL15", "IL18", "IL2", "IL4", "IL6", "TNFa",
]


# ----------------------------------------------------------------------- #
# Hand-curated labels. Key = canonical (axis_a, axis_b) with axis_a < axis_b.
# SCP-name lex sort:
#   "IFNb" < "IFNg" < "IL10" < "IL12" < "IL13" < "IL15" < "IL18" < "IL1b"
#   < "IL2" < "IL4" < "IL6" < "TNFa"
# (ASCII: '0'(48) < 'b'(98) so "IL10" < "IL1b"; '1' < '2' so "IL1b" < "IL2".)
# Biology / upstream assignment is identical to the human-name version; only the
# string representation and the resulting canonical pair order change.
# ----------------------------------------------------------------------- #
LABELS = {
    # ---- NK-cluster IFNg cascades: IL-{2,15,18} -> IFNg (sec 2.7 positive control)
    ("IFNg", "IL12"): dict(
        pair_status="BIDIRECTIONAL", expected_sign=None,
        benchmark_class="BIDIRECTIONAL", counts_in_benchmark=False, upstream="",
        reason="Bidirectional feedback: IL-12 -> STAT4 -> IFNg in NK/T, and "
               "IFNg -> STAT1 -> Il12b in DCs/macs.  Direction cell-type-dependent "
               "in vivo; excluded from signed accuracy.",
    ),
    ("IFNg", "IL15"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="NK_IFNG", counts_in_benchmark=True, upstream="IL15",
        reason="IL-15 drives NK-cell IFNg in vivo (sec 2.7 NK cascade).  "
               "IL15 upstream -> -1.",
    ),
    ("IFNg", "IL18"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="NK_IFNG", counts_in_benchmark=True, upstream="IL18",
        reason="IL-18 synergizes with IL-12 to drive NK-cell IFNg (sec 2.7).  "
               "IL18 upstream -> -1.",
    ),
    ("IFNg", "IL2"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="NK_IFNG", counts_in_benchmark=True, upstream="IL2",
        reason="IL-2 supports NK-cell IFNg production (sec 2.7 cascade).  "
               "IL2 upstream -> -1.",
    ),
    # ---- NEGATIVE: Th2 antagonism (IL-4 inhibits Th1 IFNg).  Descriptive only.
    ("IFNg", "IL4"): dict(
        pair_status="NEGATIVE_NO_CASCADE", expected_sign=0,
        benchmark_class="ANTAGONISM", counts_in_benchmark=False, upstream="",
        reason="Th2 IL-4 suppresses Th1 IFNg (antagonism, not cascade).  "
               "|cross_asym| may still be large; DESCRIPTIVE per sec 26.4.",
    ),
    # ---- Type-I IFN priming of NK for IFNg production (a -> b)
    ("IFNb", "IFNg"): dict(
        pair_status="DIRECTIONAL_a_to_b", expected_sign=+1,
        benchmark_class="TYPE1_IFN_PRIME", counts_in_benchmark=True, upstream="IFNb",
        reason="Type-I IFN primes NK for IFNg (Nguyen 2002, Mack 2011).  IFNb "
               "upstream -> +1.  Distinct pathways IFNAR(STAT1/2) vs IFNGR(STAT1-GAS).",
    ),
    # ---- NF-kB -> STAT3 cascade.  IL1b -> IL6 induction in myeloid cells.
    ("IL1b", "IL6"): dict(
        pair_status="DIRECTIONAL_a_to_b", expected_sign=+1,
        benchmark_class="NFKB_STAT3", counts_in_benchmark=True, upstream="IL1b",
        reason="IL-1R -> MyD88 -> NF-kB induces IL6; IL6 -> STAT3 (distinct).  "
               "IL1b upstream -> +1.  NF-kB vs STAT3 distinct -> precondition holds.",
    ),
    # ---- TNF -> IL6 (NF-kB -> STAT3; b -> a since "IL6" < "TNFa")
    ("IL6", "TNFa"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="NFKB_STAT3", counts_in_benchmark=True, upstream="TNFa",
        reason="TNF -> TNFR1 -> NF-kB induces IL6; IL6 -> STAT3.  TNFa upstream "
               "-> -1.  NF-kB vs STAT3 distinct (unlike IL1b/TNFa which share NF-kB).",
    ),
    # ---- NEGATIVE: IL-10 -> IL-12 antagonism (negative cascade). Descriptive.
    ("IL10", "IL12"): dict(
        pair_status="NEGATIVE_NO_CASCADE", expected_sign=0,
        benchmark_class="ANTAGONISM", counts_in_benchmark=False, upstream="",
        reason="IL-10 -> STAT3 SUPPRESSES Il12b (Moore 2001).  Negative cascade "
               "-> DESCRIPTIVE per sec 26.4.",
    ),
    # ---- UNKNOWN: IL-13 / IL-4 overlap (both STAT6 via IL-4Ralpha)
    ("IL13", "IL4"): dict(
        pair_status="UNKNOWN", expected_sign=None,
        benchmark_class="OVERLAP_STAT6", counts_in_benchmark=False, upstream="",
        reason="IL-4 and IL-13 both signal STAT6 via type-II IL-4R; programs "
               "overlap heavily; cross_asym sign unreliable.  DESCRIPTIVE.",
    ),
    # ---- UNKNOWN: IL-2 / IL-15 overlap (both common gamma-chain -> STAT5)
    ("IL15", "IL2"): dict(
        pair_status="UNKNOWN", expected_sign=None,
        benchmark_class="OVERLAP_STAT5", counts_in_benchmark=False, upstream="",
        reason="IL-2 and IL-15 both signal common gamma-chain -> STAT5; programs "
               "overlap heavily; cross_asym sign unreliable.  DESCRIPTIVE.",
    ),
    # ---- UNKNOWN: NF-kB <-> TNF overlap (known sec 24 failure mode, replicated)
    ("IL1b", "TNFa"): dict(
        pair_status="UNKNOWN", expected_sign=None,
        benchmark_class="OVERLAP_NFKB", counts_in_benchmark=False, upstream="",
        reason="Both IL1b and TNF signal NF-kB; programs overlap (sec 24 known "
               "failure).  cross_asym sign unreliable on shared activation axis.  "
               "DESCRIPTIVE.",
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
    bench_set = set(BENCHMARK_CYTOKINES)
    for (a, b), d in LABELS.items():
        if a >= b:
            print(f"FATAL: non-canonical order: {(a, b)}", file=sys.stderr)
            sys.exit(2)
        if a not in bench_set or b not in bench_set:
            print(f"FATAL: {(a, b)} not in BENCHMARK_CYTOKINES", file=sys.stderr)
            sys.exit(2)
        ps, sign = d["pair_status"], d["expected_sign"]
        if ps == "DIRECTIONAL_a_to_b" and sign != +1:
            print(f"FATAL: {a}/{b} a_to_b but sign {sign}", file=sys.stderr); sys.exit(2)
        if ps == "DIRECTIONAL_b_to_a" and sign != -1:
            print(f"FATAL: {a}/{b} b_to_a but sign {sign}", file=sys.stderr); sys.exit(2)
        if ps.startswith("DIRECTIONAL") and not d["counts_in_benchmark"]:
            print(f"FATAL: {a}/{b} directional but not in benchmark", file=sys.stderr); sys.exit(2)
        if ps == "NEGATIVE_NO_CASCADE" and (sign != 0 or d["counts_in_benchmark"]):
            print(f"FATAL: {a}/{b} negative misconfigured", file=sys.stderr); sys.exit(2)
        if ps in ("UNKNOWN", "BIDIRECTIONAL") and (sign is not None or d["counts_in_benchmark"]):
            print(f"FATAL: {a}/{b} {ps} misconfigured", file=sys.stderr); sys.exit(2)


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
        "# Immune Dictionary cascade-direction labels (cross_asym, CLAUDE.md sec 26).",
        "# Benchmark cytokines (SCP cyt names; binary models train on these only):",
        "#   " + ", ".join(BENCHMARK_CYTOKINES),
        "# expected_sign: +1 a_to_b, -1 b_to_a, 0 NEGATIVE, null UNKNOWN/BIDIR.",
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
