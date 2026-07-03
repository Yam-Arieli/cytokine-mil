"""
Score cascade DIRECTION from confusion-dynamics asymmetry (training-dynamics route).

Mechanism: a multiclass model's tubes for an upstream cytokine A carry a weak
downstream signal S_B (cascade A->B), so the classifier asymmetrically confuses
A as B. The late-epoch confusion asymmetry
    Asym[A,B] = mean_late(C[A,B]) - mean_late(C[B,A]),   C[A,B]=A-tubes' softmax mass on B
gives direction: Asym[A,B] > 0  =>  A upstream (A->B)  (same sign convention as cross_asym).

Runs on ANY multiclass run's dynamics.pkl (records must have 'softmax_trajectory'):
  - the existing FULL-gene runs (the free control), and
  - the signature-restricted run (the treatment).

Scores sign(Asym[axis_a,axis_b]) vs `expected_sign` on the 17 audited benchmark pairs
(same denominator as the cross_asym 88%), plus a temporal early/late summary.

Reuses cytokine_mil/analysis/confusion_dynamics.py entirely.

Usage:
    python scripts/score_confusion_direction.py --base_dir results/attention_dynamics --seeds 42 123 7 \
        --label full_gene --out reports/confusion_direction/CONTROL_full_gene.md
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from cytokine_mil.analysis.confusion_dynamics import (
    compute_asymmetry_score, compute_confusion_trajectory, compute_temporal_profile,
)

AUDITED = REPO / "reports/cascade_pairs/cytokine_axes_audited.csv"
AXES = REPO / "reports/cascade_pairs/cytokine_axes.csv"


class _LE:
    """Minimal label encoder shim (compute_confusion_trajectory needs .cytokines/.encode)."""
    def __init__(self, cytokines):
        self._c = list(cytokines)
        self._i = {c: i for i, c in enumerate(self._c)}

    @property
    def cytokines(self):
        return self._c

    def encode(self, name):
        return self._i[name]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default=None, help="dir with seed_<s>/dynamics.pkl")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--dynamics", nargs="+", default=None, help="explicit dynamics.pkl paths")
    p.add_argument("--label", default="run")
    p.add_argument("--late_epoch_fraction", type=float, default=0.3)
    p.add_argument("--audited_csv", default=str(AUDITED))
    p.add_argument("--axes_csv", default=str(AXES))
    p.add_argument("--out", default=None)
    return p.parse_args()


def _safe_int(x):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return 0


def _dyn_paths(args):
    if args.dynamics:
        return [Path(p) for p in args.dynamics]
    if args.base_dir:
        return [Path(args.base_dir) / f"seed_{s}" / "dynamics.pkl" for s in args.seeds]
    sys.exit("Provide --base_dir or --dynamics")


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _benchmark_pairs(audited_csv):
    rows = _read_rows(audited_csv)
    out = []
    for r in rows:
        if str(r.get("counts_in_benchmark", "")).lower() == "true":
            out.append((r["axis_a"], r["axis_b"], _safe_int(r.get("expected_sign"))))
    return out


def _coregulated_pairs(axes_csv):
    """Shared-pathway pairs (KNOWN_COREGULATED) — the 'early/symmetric' contrast set."""
    out = []
    for r in _read_rows(axes_csv):
        if r.get("literature_status", "") == "KNOWN_COREGULATED":
            out.append((r["axis_a"], r["axis_b"]))
    return out


def _seed_confusion(dp):
    """Return (asymmetry KxK, confusion KxKxT, cytokine_names) for one seed, or None."""
    with open(dp, "rb") as f:
        d = pickle.load(f)
    recs = d.get("records") or []
    cyts = d.get("label_encoder_cytokines")
    if not recs or not cyts:
        return None
    if recs[0].get("softmax_trajectory") is None:
        return None
    le = _LE(cyts)
    C, names = compute_confusion_trajectory(recs, le)   # (K,K,T)
    return C, names


def _accuracy(obs_signs, expected):
    obs = np.asarray(obs_signs, dtype=float)
    exp = np.asarray(expected, dtype=float)
    called = obs != 0
    n_called = int(called.sum())
    correct_all = int((obs == exp).sum())
    correct_called = int(((obs == exp) & called).sum())
    return n_called, correct_called, correct_all, len(obs)


def main():
    args = _parse_args()
    per_seed_asym, C_seeds, names = [], [], None
    for dp in _dyn_paths(args):
        if not dp.exists():
            print(f"skip (missing): {dp}"); continue
        res = _seed_confusion(dp)
        if res is None:
            print(f"skip (no softmax_trajectory): {dp}"); continue
        C, nm = res
        names = nm
        per_seed_asym.append(compute_asymmetry_score(C, args.late_epoch_fraction))
        C_seeds.append(C)
        print(f"loaded {dp}: K={C.shape[0]} T={C.shape[2]}")
    if not per_seed_asym:
        sys.exit("No usable dynamics.pkl (need records with softmax_trajectory).")

    idx = {c: i for i, c in enumerate(names)}
    asym_mean = np.mean(np.stack(per_seed_asym), axis=0)          # (K,K)
    C_mean = np.mean(np.stack(C_seeds), axis=0)                    # (K,K,T)

    # ---- direction accuracy on the audited benchmark ----
    bench = _benchmark_pairs(args.audited_csv)
    rows, obs, exp = [], [], []
    for a, b, es in bench:
        if a not in idx or b not in idx or es == 0:
            continue
        s = float(np.sign(asym_mean[idx[a], idx[b]]))
        rows.append((a, b, es, s)); obs.append(s); exp.append(es)
    nc, cc, ca, nt = _accuracy(obs, exp)

    # per-seed accuracy spread
    per_seed_acc = []
    for A in per_seed_asym:
        o = [float(np.sign(A[idx[a], idx[b]])) for a, b, es in bench
             if a in idx and b in idx and es != 0]
        e = [es for a, b, es in bench if a in idx and b in idx and es != 0]
        per_seed_acc.append(_accuracy(o, e))

    # ---- temporal: peak fraction of cascade (benchmark) vs coregulated pairs ----
    def peak_fracs(pairs, directional):
        fr = []
        for a, b, *rest in pairs:
            if a not in idx or b not in idx:
                continue
            # for benchmark: score the CORRECT upstream->downstream direction
            if directional:
                es = rest[0]
                src, dst = (a, b) if es > 0 else (b, a)
            else:
                src, dst = a, b   # coregulated: either direction, take a->b
            tp = compute_temporal_profile(C_mean, idx[src], idx[dst])
            fr.append(tp["peak_fraction"])
        return np.array(fr)

    casc_fr = peak_fracs(bench, directional=True)
    coreg = _coregulated_pairs(args.axes_csv)
    coreg_fr = peak_fracs([(a, b) for a, b in coreg], directional=False)
    # one-sided permutation: cascade peaks later than coregulated
    temporal_p = float("nan"); temporal_obs = float("nan")
    if casc_fr.size >= 2 and coreg_fr.size >= 2:
        temporal_obs = float(casc_fr.mean() - coreg_fr.mean())
        pooled = np.concatenate([casc_fr, coreg_fr]); na = casc_fr.size
        rng = np.random.default_rng(0); ge = 0
        for _ in range(10000):
            rng.shuffle(pooled)
            if (pooled[:na].mean() - pooled[na:].mean()) >= temporal_obs:
                ge += 1
        temporal_p = (ge + 1) / 10001

    # ---- report ----
    L = [f"# Confusion-direction — `{args.label}`\n\n",
         f"Seeds used: {len(per_seed_asym)} · K={len(names)} · "
         f"late_epoch_fraction={args.late_epoch_fraction}\n\n",
         "Direction: `Asym[a,b] = late(C[a,b]−C[b,a])`, sign vs `expected_sign` on "
         f"{len(rows)} audited benchmark pairs (same denominator as cross_asym 88%).\n\n",
         "## Direction accuracy\n",
         "| method | accuracy (correct / benchmark) | of called |\n|---|---|---|\n",
         "| IG cross_asym (reference) | 15/17 = 88% | — |\n",
         f"| confusion-asym `{args.label}` (seed-mean asym) | **{ca}/{nt} = {ca/max(nt,1):.0%}** | {cc}/{nc} |\n"]
    for s, acc in zip(args.seeds if not args.dynamics else range(len(per_seed_acc)), per_seed_acc):
        L.append(f"| &nbsp;&nbsp;seed {s} | {acc[2]}/{acc[3]} | {acc[1]}/{acc[0]} |\n")
    L.append(f"\n## Temporal (cascade should peak LATER than coregulated)\n"
             f"- cascade pairs mean peak-fraction: {casc_fr.mean() if casc_fr.size else float('nan'):.3f} (n={casc_fr.size})\n"
             f"- coregulated pairs mean peak-fraction: {coreg_fr.mean() if coreg_fr.size else float('nan'):.3f} (n={coreg_fr.size})\n"
             f"- one-sided perm p (cascade later): **{temporal_p:.4f}** (Δ={temporal_obs:.3f})\n\n")
    L.append("## Per-pair benchmark calls\n| pair | expected | observed | correct |\n|---|---|---|---|\n")
    for a, b, es, s in sorted(rows, key=lambda r: (r[3] != r[2], r[0])):
        L.append(f"| {a}→{b} | {'+' if es>0 else '−'} | "
                 f"{'+' if s>0 else '−' if s<0 else '0'} | {'✓' if s==es else '✗'} |\n")
    L.append("\n> Direction≠existence≠causation. Additive to cross_asym (§26). "
             "Sign-based (tube-level confusion); donor-level FDR graph is the refinement.\n")

    out = Path(args.out) if args.out else REPO / f"reports/confusion_direction/CONFUSION_{args.label}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(L))
    print(f"\nSaved: {out}")
    print(f"  {args.label}: direction {ca}/{nt} = {ca/max(nt,1):.0%} | "
          f"temporal p={temporal_p:.4f} (casc {casc_fr.mean() if casc_fr.size else float('nan'):.2f} "
          f"vs coreg {coreg_fr.mean() if coreg_fr.size else float('nan'):.2f})")


if __name__ == "__main__":
    main()
