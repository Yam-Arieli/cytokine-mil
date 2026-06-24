#!/usr/bin/env python
"""§32 confirmatory — decompose the STATE-axis cross_asym INVERSION into
(a) control-composition vs (b) retention-biology, by re-scoring direction under
several control definitions while HOLDING THE SIGNATURES FIXED.

  cross_asym(a,b) = M[a,b] − M[b,a],  M[a,b] = median_T[ s(a, S_b) − ctrl_{S_b} ]

The discovered signatures S_X (state-vs-Resting binary IG) are FIXED; only the
control term varies. That isolates the two mechanisms cleanly:
  * the RAW cross-engagement  s(a,S_b) − s(b,S_a)  (control = ZERO)  = effect (b),
    the differentiation retention-biology (memory retains the naive program);
  * the control term  ctrl_{S_b} − ctrl_{S_a}                       = effect (a),
    the composition of the baseline (day-0 'Resting' carries pre-existing memory).

Controls tested (no GPU; reads the saved fit_state artifacts):
  Resting   original day-0 control (naive-dominated + baseline memory)
  ZERO      no subtraction -> raw cross-engagement = effect (b) ALONE
  Balanced  per donor x cell_type mean of {Naive,Effector,Memory} -> removes the
            naive/memory imbalance of the control = tests effect (a)
  Naive / Memory / Effector   each state as the control (diagnostics)

True order: Naive < Effector < Memory (Naive upstream). For the alphabetical pairs,
CORRECT means cross_asym(Effector,Naive)<0 and cross_asym(Memory,Naive)<0 (Naive=b
upstream); the observed run had them POSITIVE (inverted). Kendall tau vs truth: +1
correct, -1 fully inverted.

Usage (cluster, CPU):
  python scripts/analyze_vaccine_state_control_decomp.py \
      --fit_dir results/vaccine_progression/fit_state \
      --out reports/vaccine_progression/STATE_CONTROL_DECOMP.md
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from cascadir.progression import (
    CELLTYPE_COL, CONDITION_COL, DONOR_COL, NCELLS_COL, SIG_PREFIX,
    kendall_tau, pooled_cross_asym, recover_order,
)

STATES = ["Naive", "Effector", "Memory"]   # true order, most-upstream first
CONTROL_ORIG = "Resting"


def _sig_cols():
    return [f"{SIG_PREFIX}{s}" for s in STATES]


def _add_zero(cache: pd.DataFrame) -> pd.DataFrame:
    """A control that scores 0 on every signature -> cross_asym = raw engagement."""
    base = cache[[DONOR_COL, CELLTYPE_COL]].drop_duplicates().copy()
    base[CONDITION_COL] = "ZERO"
    base[NCELLS_COL] = 1
    for c in _sig_cols():
        base[c] = 0.0
    return pd.concat([cache, base], ignore_index=True)


def _add_balanced(cache: pd.DataFrame) -> pd.DataFrame:
    """A control = per (donor, cell_type) mean over the 3 STATE rows (no naive/memory
    imbalance). If swapping to this un-inverts direction, effect (a) was decisive."""
    sub = cache[cache[CONDITION_COL].isin(STATES)]
    agg = (sub.groupby([DONOR_COL, CELLTYPE_COL])
              .agg({**{c: "mean" for c in _sig_cols()}, NCELLS_COL: "mean"})
              .reset_index())
    agg[CONDITION_COL] = "Balanced"
    return pd.concat([cache, agg], ignore_index=True)


def _score(cache: pd.DataFrame, control: str) -> dict:
    ca = pooled_cross_asym(cache, STATES, control)
    order = recover_order(ca, STATES)
    tau = kendall_tau(order, STATES)
    return {"cross_asym": ca, "order": order, "tau": tau}


def _sig_overlap(sigs: dict) -> str:
    lines = [f"- |S_{s}| = {len(sigs.get(s, []))}" for s in STATES]
    for a, b in combinations(STATES, 2):
        A, B = set(sigs.get(a, [])), set(sigs.get(b, []))
        j = len(A & B) / len(A | B) if (A | B) else float("nan")
        lines.append(f"- Jaccard(S_{a}, S_{b}) = {j:.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_dir", default="results/vaccine_progression/fit_state")
    ap.add_argument("--out", default="reports/vaccine_progression/STATE_CONTROL_DECOMP.md")
    args = ap.parse_args()
    fit = Path(args.fit_dir)

    cache = pd.read_parquet(fit / "donor_signature_scores.parquet")
    sigs = json.loads((fit / "signatures.json").read_text())
    present = sorted(set(cache[CONDITION_COL]))
    print(f"[load] cache rows={len(cache)}; conditions={present}; "
          f"cell_types={sorted(set(cache[CELLTYPE_COL]))}")

    cache_z = _add_zero(cache)
    cache_b = _add_balanced(cache)

    controls = [
        ("Resting (orig)", cache, CONTROL_ORIG),
        ("ZERO  = raw engagement (effect b)", cache_z, "ZERO"),
        ("Balanced (removes naive/mem imbalance; tests effect a)", cache_b, "Balanced"),
        ("Naive (as control)", cache, "Naive"),
        ("Memory (as control)", cache, "Memory"),
        ("Effector (as control)", cache, "Effector"),
    ]

    results = []
    for name, cdf, lab in controls:
        if lab not in set(cdf[CONDITION_COL]):
            print(f"[skip] control {lab} absent"); continue
        r = _score(cdf, lab)
        results.append((name, lab, r))

    # headline pairs (alphabetical a<b): correct sign is NEGATIVE (Naive=b upstream)
    pairs = [("Effector", "Naive"), ("Memory", "Naive"), ("Effector", "Memory")]

    print("\n=== cross_asym per control (correct sign for X–Naive is NEGATIVE) ===")
    hdr = f"{'control':<46} " + " ".join(f"{a[:3]}-{b[:3]:<8}" for a, b in pairs) + "  order  tau"
    print(hdr)
    rows_md = []
    for name, lab, r in results:
        ca = r["cross_asym"]
        cells = " ".join(f"{ca.get((a, b), float('nan')):+8.3f}   " for a, b in pairs)
        order_s = ">".join(x[:3] for x in r["order"])
        print(f"{name:<46} {cells}  {order_s}  tau={r['tau']:+.2f}")
        rows_md.append((name, lab, r))

    # effect decomposition for the two inverted pairs
    ca_zero = dict(next(r for n, l, r in results if l == "ZERO")["cross_asym"])
    ca_rest = dict(next(r for n, l, r in results if l == CONTROL_ORIG)["cross_asym"])
    decomp = []
    for a, b in [("Effector", "Naive"), ("Memory", "Naive")]:
        raw = ca_zero[(a, b)]                  # effect (b)
        full = ca_rest[(a, b)]                 # observed
        ctrl_term = raw - full                 # effect (a): how much the control shifted it
        decomp.append((a, b, raw, full, ctrl_term))

    _write_report(args.out, sigs, results, pairs, decomp)
    print(f"\n[write] {args.out}")
    return 0


def _verdict(decomp, results):
    tau = {l: r["tau"] for _, l, r in results}
    z, rest = tau.get("ZERO"), tau.get(CONTROL_ORIG)
    bal = tau.get("Balanced")
    if z is not None and z <= -0.5:
        return ("Effect (b) RETENTION-BIOLOGY is sufficient: even with NO control "
                "(raw cross-engagement) the direction stays inverted "
                f"(tau_ZERO={z:+.2f}). The naive-dominated control (effect a) "
                f"reinforces it (tau_Resting={rest:+.2f}) but is not the sole cause. "
                f"Balanced control tau={bal:+.2f}.")
    if z is not None and z > -0.5 and (rest is not None and rest <= -0.5):
        return ("Effect (a) CONTROL-COMPOSITION dominates: removing the control "
                f"un-inverts the raw engagement (tau_ZERO={z:+.2f}) while the "
                f"day-0 control inverts it (tau_Resting={rest:+.2f}). "
                f"Balanced control tau={bal:+.2f} corroborates.")
    return (f"Mixed: tau_ZERO={z}, tau_Resting={rest}, tau_Balanced={bal}. "
            "Inspect the per-pair decomposition.")


def _write_report(out, sigs, results, pairs, decomp):
    v = _verdict(decomp, results)
    lines = [
        "# §32 confirmatory — STATE-axis inversion: control-composition (a) vs retention-biology (b)",
        "",
        "Holds the discovered signatures FIXED (state-vs-Resting binary IG) and varies ONLY the "
        "`cross_asym` control, so `cross_asym = raw_engagement − control_term` is decomposed: "
        "ZERO control = raw engagement = **effect (b)**; the control term = **effect (a)**. "
        "True order Naive<Effector<Memory ⇒ for the X–Naive pairs the CORRECT sign is **negative** "
        "(Naive upstream); the observed run had them positive (inverted), τ=−1.",
        "",
        "## Verdict",
        "",
        v,
        "",
        "## cross_asym per control (correct sign for X–Naive is NEGATIVE; τ: +1 correct, −1 inverted)",
        "",
        "| control | " + " | ".join(f"{a}–{b}" for a, b in pairs) + " | recovered order | τ |",
        "|---|" + "---|" * (len(pairs) + 2),
    ]
    for name, lab, r in results:
        ca = r["cross_asym"]
        cells = " | ".join(f"{ca.get((a, b), float('nan')):+.3f}" for a, b in pairs)
        order_s = " > ".join(r["order"])
        lines.append(f"| {name} | {cells} | {order_s} | {r['tau']:+.2f} |")
    lines += [
        "",
        "## Effect decomposition (inverted X–Naive pairs)",
        "",
        "`cross_asym_Resting = raw_engagement(ZERO) − control_term`",
        "",
        "| pair | raw engagement = effect (b) | control term = effect (a) | observed (Resting) |",
        "|---|---:|---:|---:|",
    ]
    for a, b, raw, full, ctrl in decomp:
        lines.append(f"| {a}–{b} | {raw:+.3f} | {ctrl:+.3f} | {full:+.3f} |")
    lines += [
        "",
        "## Signature sizes / overlap (is S_Naive degenerate vs the Resting control?)",
        "",
        _sig_overlap(sigs),
        "",
        "_Generated by `scripts/analyze_vaccine_state_control_decomp.py` from the saved "
        "`fit_state` artifacts (no GPU re-fit). Method basis: CLAUDE.md §26/§32._",
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
