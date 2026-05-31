"""
Author + finalize the crystal-clear Sheu cascade-direction labels.

Unlike the Oesinghaus labels (web-search-derived, noisy), the Sheu labels come
directly from textbook TLR receptor/adaptor biology:
    TLR3/TLR4 → TRIF → IRF3 → IFN-β
    TLR4/TLR2/TLR9 → MyD88 → NF-κB → TNF
    TNF → TNFR → NF-κB
    IFN-β → IFNAR → ISGs

Stimulus names follow the Sheu pseudotube manifest convention (verified against
scripts/build_pseudotubes_sheu2024.py ACTIVE_STIMULI_3HR and
cytokine_mil/analysis/eda_pair_benchmark.py): LPS, LPSlo, P3CSK, PIC, TNF,
CpG, IFNb  (NOT Pam3CSK4 / polyIC — those are config-doc aliases only).

This script embeds the hand-curated decisions, validates them (all C(7,2)=21
pairs present, canonical alphabetical ordering, expected_sign consistent with
pair_status), and emits:
    reports/sheu_cascade/sheu_cascade_labels.yaml   (human-readable record)
    reports/sheu_cascade/sheu_axes_labeled.csv      (machine-readable; consumed
        by run_pipeline_a_bridge_b_sheu.py AND retally_pipeline_against_audit.py)

Convention (matches the pipeline driver `_expected_sign`):
    expected_sign = +1  ⇔ cascade is axis_a → axis_b (a_to_b);
                          cross_asym(axis_a, axis_b) should be POSITIVE.
    expected_sign = −1  ⇔ cascade is axis_b → axis_a (b_to_a); negative.
    expected_sign =  0  ⇔ NEGATIVE control (no cascade) — |cross_asym| ≈ 0.
    expected_sign = None⇔ UNKNOWN — excluded from all accuracy.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "sheu_cascade"
YAML_PATH = OUT_DIR / "sheu_cascade_labels.yaml"
CSV_PATH = OUT_DIR / "sheu_axes_labeled.csv"

STIMULI = ["CpG", "IFNb", "LPS", "LPSlo", "P3CSK", "PIC", "TNF"]


# ----------------------------------------------------------------------- #
# Hand-curated labels. Key = canonical (axis_a, axis_b) with axis_a < axis_b.
# ----------------------------------------------------------------------- #
# fields: pair_status, expected_sign, benchmark_class, counts_in_benchmark,
#         upstream (the stimulus that is upstream, or "" for negative/unknown),
#         reason
LABELS = {
    # ---- IFN cascades: distinct pathways, §24 precondition holds (MUST) ----
    ("IFNb", "PIC"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="IFN_MUST", counts_in_benchmark=True, upstream="PIC",
        reason="polyIC engages TLR3->TRIF->IRF3 directly; IFN-beta is the "
               "autocrine product acting via IFNAR. S_PIC (IRF3-direct + ISGs) "
               "strictly contains S_IFNb (ISGs) -> cleanest cascade; precondition holds.",
    ),
    ("IFNb", "LPS"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="IFN_MUST", counts_in_benchmark=True, upstream="LPS",
        reason="LPS engages the TLR4->TRIF arm -> IRF3 -> IFN-beta. LPS-tubes "
               "carry the IFN-beta response (autocrine); IFNb-tubes do not carry "
               "the IRF3-direct genes. Upstream = LPS.",
    ),
    # ---- IFN cascade, low dose: direction unambiguous, magnitude weaker (SHOULD)
    ("IFNb", "LPSlo"): dict(
        pair_status="DIRECTIONAL_b_to_a", expected_sign=-1,
        benchmark_class="IFN_SHOULD", counts_in_benchmark=True, upstream="LPSlo",
        reason="Low-dose LPS still triggers TLR4->TRIF->IRF3->IFN-beta; the "
               "direction (LPSlo upstream) is unambiguous though IFN-beta "
               "induction is weaker than full LPS.",
    ),
    # ---- NF-kB cascades: P_A ~ P_B overlap, §24 precondition RISK (SHOULD) ----
    ("CpG", "TNF"): dict(
        pair_status="DIRECTIONAL_a_to_b", expected_sign=+1,
        benchmark_class="NFKB_SHOULD", counts_in_benchmark=True, upstream="CpG",
        reason="TLR9->MyD88->NF-kB->autocrine TNF. CpG upstream. NF-kB and TNFR "
               "programs overlap -> precondition risk; expected weaker than IFN cascades.",
    ),
    ("LPS", "TNF"): dict(
        pair_status="DIRECTIONAL_a_to_b", expected_sign=+1,
        benchmark_class="NFKB_SHOULD", counts_in_benchmark=True, upstream="LPS",
        reason="TLR4->MyD88->NF-kB->autocrine TNF loop. LPS upstream. NF-kB/TNFR "
               "overlap -> precondition risk (this cascade failed §24 with curated sets).",
    ),
    ("LPSlo", "TNF"): dict(
        pair_status="DIRECTIONAL_a_to_b", expected_sign=+1,
        benchmark_class="NFKB_SHOULD", counts_in_benchmark=True, upstream="LPSlo",
        reason="Low-dose TLR4->NF-kB->TNF. LPSlo upstream. NF-kB/TNFR overlap -> risk.",
    ),
    ("P3CSK", "TNF"): dict(
        pair_status="DIRECTIONAL_a_to_b", expected_sign=+1,
        benchmark_class="NFKB_SHOULD", counts_in_benchmark=True, upstream="P3CSK",
        reason="TLR2->MyD88->NF-kB->TNF. P3CSK upstream. NF-kB/TNFR overlap -> risk.",
    ),
    # ---- NEGATIVES: no cascade. expect |cross_asym| ~ 0 (specificity) ----
    ("CpG", "IFNb"): dict(
        pair_status="NEGATIVE_NO_CASCADE", expected_sign=0,
        benchmark_class="NEGATIVE", counts_in_benchmark=False, upstream="",
        reason="TLR9 type-I IFN is plasmacytoid-DC restricted; BMDM produce "
               "minimal IFN-beta via CpG. No cascade in either direction.",
    ),
    ("IFNb", "P3CSK"): dict(
        pair_status="NEGATIVE_NO_CASCADE", expected_sign=0,
        benchmark_class="NEGATIVE", counts_in_benchmark=False, upstream="",
        reason="TLR2 has no TRIF arm -> no IRF3 -> no IFN-beta induction. "
               "P3CSK does not drive IFN-beta; IFN-beta does not drive P3CSK biology.",
    ),
    ("IFNb", "TNF"): dict(
        pair_status="NEGATIVE_NO_CASCADE", expected_sign=0,
        benchmark_class="NEGATIVE", counts_in_benchmark=False, upstream="",
        reason="No TNF->IFN-beta cross-induction in macrophages; IFNAR and TNFR "
               "programs are distinct with no cascade link in BMDM.",
    ),
    ("LPS", "LPSlo"): dict(
        pair_status="NEGATIVE_NO_CASCADE", expected_sign=0,
        benchmark_class="NEGATIVE", counts_in_benchmark=False, upstream="",
        reason="Same TLR4 receptor at different doses — a dose-response pair, "
               "not a cascade. Neither induces the other.",
    ),
    # ---- UNKNOWN: no clear directional biology -> excluded from accuracy ----
    ("CpG", "LPS"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="Both signal via MyD88->NF-kB (LPS also TRIF); overlapping programs, "
               "no clean cascade direction.",
    ),
    ("CpG", "LPSlo"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="Both MyD88-biased; overlapping NF-kB, no clear directional cascade.",
    ),
    ("CpG", "P3CSK"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="Both MyD88-only (TLR9, TLR2); same adaptor, parallel — no cascade.",
    ),
    ("CpG", "PIC"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="CpG=TLR9/MyD88, PIC=TLR3/TRIF — orthogonal adaptors, no cascade link.",
    ),
    ("LPS", "P3CSK"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="LPS dual-arm (TRIF+MyD88), P3CSK MyD88-only; LPS engages a broader "
               "program but neither induces the other -> not a directional cascade.",
    ),
    ("LPS", "PIC"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="Different receptors (TLR4 vs TLR3); both reach TRIF but no induction "
               "between them in BMDM.",
    ),
    ("LPSlo", "P3CSK"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="Both MyD88-biased; coupled by similarity (§22 SHOULD) but no "
               "directional cascade between them.",
    ),
    ("LPSlo", "PIC"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="TLR4 vs TLR3 — separate adaptor axes, no cascade direction.",
    ),
    ("P3CSK", "PIC"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="TLR2/MyD88 vs TLR3/TRIF — orthogonal pathways, no cascade.",
    ),
    ("PIC", "TNF"): dict(
        pair_status="UNKNOWN", expected_sign=None, benchmark_class="UNKNOWN",
        counts_in_benchmark=False, upstream="",
        reason="polyIC can induce TNF via TRIF->NF-kB, but polyIC's dominant output "
               "is type-I IFN not TNF; direction ambiguous -> excluded to stay crystal-clear.",
    ),
}


def _expected_to_lit_dir(pair_status: str) -> str:
    if pair_status == "DIRECTIONAL_a_to_b":
        return "a_to_b"
    if pair_status == "DIRECTIONAL_b_to_a":
        return "b_to_a"
    return "no_lit"


def validate() -> None:
    # all 21 canonical pairs present, canonical ordering, sign consistency
    all_pairs = set()
    for i in range(len(STIMULI)):
        for j in range(i + 1, len(STIMULI)):
            all_pairs.add((STIMULI[i], STIMULI[j]))
    keys = set(LABELS.keys())
    missing = all_pairs - keys
    extra = keys - all_pairs
    if missing:
        print(f"FATAL: missing pairs: {sorted(missing)}", file=sys.stderr)
        sys.exit(2)
    if extra:
        print(f"FATAL: extra/non-canonical pairs: {sorted(extra)}", file=sys.stderr)
        sys.exit(2)
    for (a, b), d in LABELS.items():
        if a >= b:
            print(f"FATAL: non-canonical order: {(a, b)}", file=sys.stderr)
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
        if ps == "UNKNOWN" and (sign is not None or d["counts_in_benchmark"]):
            print(f"FATAL: {a}/{b} unknown misconfigured", file=sys.stderr); sys.exit(2)


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
                "original_direction": lit_dir,   # hand-curated == final
                "tag_changed": False,
                "reason": d["reason"],
            })
    print(f"Wrote {CSV_PATH} ({len(LABELS)} pairs)")


def emit_yaml() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Crystal-clear Sheu cascade-direction labels (TLR receptor biology).",
        "# Stimulus names = Sheu manifest convention (P3CSK, PIC — not Pam3CSK4/polyIC).",
        "# expected_sign: +1 a_to_b, -1 b_to_a, 0 negative (no cascade), null unknown.",
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
    # quick tally
    by_class = {}
    for d in LABELS.values():
        by_class[d["benchmark_class"]] = by_class.get(d["benchmark_class"], 0) + 1
    n_bench = sum(1 for d in LABELS.values() if d["counts_in_benchmark"])
    print(f"\nbenchmark (directional, counts_in_benchmark=True): {n_bench}")
    for c in sorted(by_class):
        print(f"  {c:14s} {by_class[c]}")


if __name__ == "__main__":
    main()
