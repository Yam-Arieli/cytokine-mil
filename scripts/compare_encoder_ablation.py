"""
Compare the two encoder-ablation arms (two_stage vs single_stage) on Oesinghaus:
does the Stage-1 cell-type PRE-TRAINING matter, or does a single end-to-end train match it?

Three readouts per arm:
  - binary val P(correct): did the model learn stimulus-vs-PBS? (from dynamics_*.pkl)
  - direction accuracy: sign(cross_asym) vs expected_sign on the 17 audited pairs
    (from coupling_cell/cell_degree_IG_vsPBS.csv).
  - donor-level coupling: benchmark recall + over-call (from donor_coupling_summary.csv).

Writes a markdown verdict comparing the arms. numpy/pandas only.
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["two_stage", "single_stage"]


def _val_pcorrect(bin_dir: Path):
    pkls = sorted(glob.glob(str(bin_dir / "dynamics_*.pkl")))
    vals = []
    for f in pkls:
        try:
            with open(f, "rb") as fh:
                p = pickle.load(fh)
        except Exception:
            continue
        for r in p.get("val_records", []):
            t = r.get("p_correct_trajectory")
            if t is not None and len(np.asarray(t)):
                vals.append(float(np.asarray(t)[-1]))
    return (float(np.mean(vals)) if vals else float("nan")), len(pkls)


def _expected_signs(audit_csv: str):
    """Canonical-pair -> expected_sign (+1 a_upstream / -1 b_upstream), reordered to
    the sorted (a<=b) convention used by cross_asym."""
    a = pd.read_csv(audit_csv)
    out = {}
    for r in a.itertuples():
        es = str(getattr(r, "expected_sign", ""))
        if es not in ("1", "-1", "1.0", "-1.0"):
            continue
        aa, bb = str(r.axis_a), str(r.axis_b)
        sign = 1.0 if float(es) > 0 else -1.0
        k = tuple(sorted((aa, bb)))
        if (aa, bb) != k:        # reordered -> flip
            sign = -sign
        out[k] = sign
    return out


def _direction_acc(cell_csv: Path, exp: dict):
    if not cell_csv.exists():
        return 0, 0
    df = pd.read_csv(cell_csv)
    ok = n = 0
    for r in df.itertuples():
        k = tuple(sorted((str(r.axis_a), str(r.axis_b))))
        ca = float(getattr(r, "cross_asym", float("nan")))
        if k in exp and np.isfinite(ca) and ca != 0.0:
            n += 1
            ok += int(np.sign(ca) == np.sign(exp[k]))
    return ok, n


def _coupling(summary_csv: Path):
    if not summary_csv.exists():
        return None
    s = pd.read_csv(summary_csv)
    row = s[(s["mode"] == "hub") & (s["variant"] == "IG_vsPBS")]
    if not len(row):
        return None
    r = row.iloc[0]
    return (str(r.get("benchmark_recall_q10", "?")),
            r.get("donor_coupled_q10_frac", float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/encoder_ablation")
    ap.add_argument("--audit_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(args.base)
    exp = _expected_signs(args.audit_csv)

    rows = []
    for arm in ARMS:
        vacc, nmodels = _val_pcorrect(base / arm / "binary")
        dok, dn = _direction_acc(base / arm / "coupling_cell" / "cell_degree_IG_vsPBS.csv", exp)
        coup = _coupling(base / arm / "donor_coupling" / "donor_coupling_summary.csv")
        rows.append({
            "arm": arm, "n_models": nmodels,
            "val_pcorrect": vacc,
            "direction": f"{dok}/{dn}" + (f" ({100*dok/dn:.0f}%)" if dn else ""),
            "direction_frac": (dok / dn) if dn else float("nan"),
            "coupling_recall": coup[0] if coup else "n/a",
            "coupling_overcall": (f"{coup[1]:.2f}" if coup and pd.notna(coup[1]) else "n/a"),
        })
    df = pd.DataFrame(rows)

    two = df[df.arm == "two_stage"].iloc[0]
    one = df[df.arm == "single_stage"].iloc[0]
    dv = one.val_pcorrect - two.val_pcorrect
    dd = one.direction_frac - two.direction_frac
    if (dv > -0.03) and (dd > -0.06):
        verdict = ("**Single-stage MATCHES two-stage** (val and direction within noise) "
                   "-> the cell-type pre-training stage is REMOVABLE; a single end-to-end "
                   "train suffices. The encoder architecture still matters (§2a), but the "
                   "separate Stage-1 step does not.")
    elif (dv < -0.05) or (dd < -0.10):
        verdict = ("**Two-stage WINS** (single-stage drops on val and/or direction) -> the "
                   "cell-type pre-training stage is LOAD-BEARING; keep it.")
    else:
        verdict = ("**Mixed/borderline** -- single-stage is close but not clearly equal; "
                   "report both, consider an LR sweep before concluding.")

    L = ["## (2b) Two-stage vs single-stage encoder training -- Oesinghaus", ""]
    L.append(f"**Verdict:** {verdict}")
    L.append("")
    L.append("| arm | binary val P(correct) | direction acc | donor coupling recall | over-call |")
    L.append("| --- | ---: | :--: | :--: | :--: |")
    for r in rows:
        L.append(f"| {r['arm']} | {r['val_pcorrect']:.3f} | {r['direction']} | "
                 f"{r['coupling_recall']} | {r['coupling_overcall']} |")
    L.append("")
    L.append("- two_stage = Stage-1 cell-type pre-train -> frozen Stage-2 MIL (current).")
    L.append("- single_stage = random-init encoder, end-to-end, encoder unfrozen "
             "(lr 1e-3), no cell-type pre-training. Same 24 cytokines / data / seed.")
    L.append("- direction = sign(cross_asym) vs expected_sign on the 17 audited pairs.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
