"""
Compare the §33 attention-collapse interventions vs the baseline.

Tabulates, per experiment (mean over seeds): P1@ep10, P1@final, the collapse
magnitude (top-1 cell-type attention share at the final epoch), and the final
train stimulus discriminability (mean p_correct) — the regularization tradeoff.

Reads each run's attention_trajectory.pkl + dynamics.pkl. No retraining.

Usage:
    python scripts/compare_attn_experiments.py \
        --baseline_dir results/attention_dynamics \
        --exp_base results/attn_reg --exps A B C --seeds 42 123 7
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.attention_dynamics import EXPECTED_DOMINANT  # noqa: E402

KEY = list(EXPECTED_DOMINANT.keys())


def _frac_match(traj, epoch_idx, top_k=3):
    matched, n = 0, 0
    for cyt in KEY:
        by_ct = traj.get(cyt)
        if not by_ct:
            continue
        n += 1
        ranked = sorted(by_ct.items(),
                        key=lambda kv: float(np.asarray(kv[1])[epoch_idx]), reverse=True)
        if any(ct in set(EXPECTED_DOMINANT[cyt]) for ct, _ in ranked[:top_k]):
            matched += 1
    return matched / n if n else float("nan")


def _top1_share(traj, epoch_idx):
    """Mean over cytokines of (max cell-type attention / total) at epoch — collapse proxy."""
    shares = []
    for by_ct in traj.values():
        vals = np.array([float(np.asarray(a)[epoch_idx]) for a in by_ct.values()])
        s = vals.sum()
        if s > 0:
            shares.append(vals.max() / s)
    return float(np.mean(shares)) if shares else float("nan")


def _final_p_correct(dyn_path):
    if not dyn_path.exists():
        return float("nan")
    with open(dyn_path, "rb") as f:
        recs = pickle.load(f).get("records") or []
    finals = [r["p_correct_trajectory"][-1] for r in recs if r.get("p_correct_trajectory")]
    return float(np.mean(finals)) if finals else float("nan")


def _seed_metrics(run_dir: Path):
    pkl = run_dir / "attention_trajectory.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        at = pickle.load(f)
    traj, epochs = at["trajectory"], at["epochs"]
    e10 = min(range(len(epochs)), key=lambda i: abs(epochs[i] - 10))
    return {
        "p1_ep10": _frac_match(traj, e10),
        "p1_final": _frac_match(traj, -1),
        "top1_share_final": _top1_share(traj, -1),
        "p_correct_final": _final_p_correct(run_dir / "dynamics.pkl"),
    }


def _agg(dirs, seeds):
    rows = [_seed_metrics(d / f"seed_{s}") for s in seeds for d in [dirs]]
    rows = [r for r in rows if r]
    if not rows:
        return None
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default="results/attention_dynamics")
    ap.add_argument("--exp_base", default="results/attn_reg")
    ap.add_argument("--exps", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--out", default=str(
        REPO_ROOT / "reports" / "attention_dynamics" / "INTERVENTION_COMPARISON.md"))
    args = ap.parse_args()

    runs = {"baseline": Path(args.baseline_dir)}
    for e in args.exps:
        runs[e] = Path(args.exp_base) / e

    table = {}
    for name, d in runs.items():
        m = _agg(d, args.seeds)
        if m:
            table[name] = m
            print(f"{name:<10} P1@10={m['p1_ep10']:.2f}  P1@final={m['p1_final']:.2f}  "
                  f"top1_share={m['top1_share_final']:.2f}  p_correct={m['p_correct_final']:.2f}")
        else:
            print(f"{name:<10} (no results yet)")

    # Figure: grouped bars
    if table:
        metrics = ["p1_ep10", "p1_final", "top1_share_final", "p_correct_final"]
        names = list(table)
        x = np.arange(len(metrics))
        w = 0.8 / max(len(names), 1)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, nm in enumerate(names):
            ax.bar(x + i * w, [table[nm][m] for m in metrics], w, label=nm)
        ax.set_xticks(x + w * (len(names) - 1) / 2)
        ax.set_xticklabels(["P1@ep10", "P1@final", "top1 share\n(collapse)", "p_correct\n(final)"])
        ax.set_ylabel("mean over seeds"); ax.legend(); ax.set_ylim(0, 1.05)
        ax.set_title("§33 attention-collapse interventions vs baseline")
        fig.tight_layout()
        out_png = Path(args.exp_base) / "intervention_comparison.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"Saved figure: {out_png}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    L = ["# §33 attention-collapse interventions — comparison vs baseline\n",
         "\nMean over seeds. **top1_share** = fraction of attention on the single top "
         "cell type at the final epoch (higher = more collapsed). **p_correct** = final "
         "train stimulus discriminability (the regularization tradeoff — should stay high).\n",
         "\n| run | P1@ep10 | P1@final | top1_share (collapse↓) | p_correct (final) |\n",
         "|---|---|---|---|---|\n"]
    for name, m in table.items():
        L.append(f"| {name} | {m['p1_ep10']:.2f} | {m['p1_final']:.2f} | "
                 f"{m['top1_share_final']:.2f} | {m['p_correct_final']:.2f} |\n")
    L.append("\n**Read:** a successful intervention raises **P1@final toward P1@ep10 (~0.8)** "
             "and lowers **top1_share** WITHOUT a large drop in **p_correct** vs baseline.\n")
    out.write_text("".join(L))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
