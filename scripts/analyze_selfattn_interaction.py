"""
Analyze the §34 self-attention cell x cell interaction trajectory.

Consumes <run_dir>/interaction_trajectory.pkl (+ attention_trajectory.pkl for the
pooling attention-primary cell types) and produces:

  - G0 go/no-go: off-diagonal (cross-cell-type) attention mass over training.
  - directed cell-type interaction graph per key cytokine (asymmetry).
  - relay direction for known cascades (IL-12/IL-2/IL-15 -> IFN-gamma) + the
    IL-6 / TNF-alpha negative control.
  - ~4 figures + a verdict markdown.

Usage:
    python scripts/analyze_selfattn_interaction.py --run_dir <dir> [--report out.md]
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.attention_dynamics import attention_primary, celltype_recruitment
from cytokine_mil.analysis.attention_interaction import (
    directed_pairs, interaction_matrix, offdiagonal_summary, relay_interaction_direction,
)

KNOWN_CASCADES = [("IL-12", "IFN-gamma"), ("IL-2", "IFN-gamma"), ("IL-15", "IFN-gamma")]
NEG_CONTROL = [("IL-6", "TNF-alpha")]
KEY_CYTOKINES = ["IL-12", "IFN-gamma", "IL-2", "IL-15", "TNF-alpha", "IL-6"]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--report", default=None)
    p.add_argument("--min_offdiag", type=float, default=0.2,
                   help="Go/no-go threshold on final-epoch cross-type attention fraction.")
    return p.parse_args()


def _load(run_dir: Path):
    with open(run_dir / "interaction_trajectory.pkl", "rb") as f:
        inter = pickle.load(f)
    pool = None
    p = run_dir / "attention_trajectory.pkl"
    if p.exists():
        with open(p, "rb") as f:
            pool = pickle.load(f)
    return inter, pool


def _fig_offdiag(offd_summary, offdiag, epochs, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cyt, arr in sorted(offdiag.items()):
        ax.plot(epochs, np.asarray(arr), alpha=0.35, lw=0.8)
    mean_curve = np.mean(np.stack([np.asarray(a) for a in offdiag.values()]), axis=0)
    ax.plot(epochs, mean_curve, color="black", lw=2.5, label="mean across cytokines")
    ax.axhline(offd_summary["min_frac"], color="red", ls="--", label=f"gate={offd_summary['min_frac']}")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-cell-type attention fraction")
    ax.set_title("§34 G0 — off-diagonal (cross-type) self-attention mass over training")
    ax.legend(); ax.set_ylim(0, 1.02); fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)


def _fig_heatmap(inter, cyt, out):
    if cyt not in inter["interaction"]:
        return False
    M, cts = interaction_matrix(inter["interaction"][cyt], epoch_idx=-1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cts))); ax.set_yticklabels(cts, fontsize=6)
    ax.set_xlabel("target type σ (attended-to)"); ax.set_ylabel("source type τ (attending)")
    ax.set_title(f"M[τ,σ] final-epoch interaction — {cyt}")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    return True


def _fig_relay_bars(relay_rows, out):
    if not relay_rows:
        return False
    labels = [f"{r['A']}→{r['B']}" for r in relay_rows]
    vals = [r["D"] if np.isfinite(r["D"]) else 0.0 for r in relay_rows]
    colors = ["seagreen" if v > 0 else "indianred" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("D(A,B) = M^A[T_B,T_A] − M^B[T_A,T_B]  (>0 ⇒ A→B)")
    ax.set_title("§34 relay interaction direction — known cascades + control")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    inter, pool = _load(run_dir)
    epochs = inter["epochs"]
    plots = run_dir / "plots"; plots.mkdir(exist_ok=True)

    # G0 go/no-go
    offd_summary = offdiagonal_summary(inter["offdiag"], epochs, min_frac=args.min_offdiag)
    _fig_offdiag(offd_summary, inter["offdiag"], epochs, plots / "offdiag_mass.png")

    # per-cytokine heatmaps + directed pairs
    for cyt in KEY_CYTOKINES:
        _fig_heatmap(inter, cyt, plots / f"interaction_{cyt.replace('/', '_')}.png")

    # relay direction (needs pooling attention-primary for T_A/T_B)
    relay_rows = []
    pool_traj = pool["trajectory"] if pool else {}
    for A, B in KNOWN_CASCADES + NEG_CONTROL:
        T_A = attention_primary(pool_traj.get(A, {})) if pool_traj else None
        T_B = attention_primary(pool_traj.get(B, {})) if pool_traj else None
        if T_A is None or T_B is None or A not in inter["interaction"] or B not in inter["interaction"]:
            continue
        row = relay_interaction_direction(inter["interaction"], A, B, T_A, T_B)
        row["kind"] = "cascade" if (A, B) in KNOWN_CASCADES else "control"
        relay_rows.append(row)
    _fig_relay_bars(relay_rows, plots / "relay_direction.png")

    # report
    L = ["# §34 self-attention interaction — analysis\n",
         f"Run: `{run_dir}` · {len(inter['cytokines'])} cytokines · "
         f"{len(inter['cell_types'])} cell types · epochs {epochs[0]}..{epochs[-1]}\n\n",
         "## G0 go/no-go — do cells actually interact?\n",
         f"- mean final cross-cell-type attention fraction: **{offd_summary['mean_final']:.3f}** "
         f"(gate ≥ {offd_summary['min_frac']}) → **{'PASS' if offd_summary['gate_pass'] else 'FAIL'}**\n",
         "  (FAIL ⇒ SAB collapsed to self/diagonal; cells don't interact; premise broken)\n\n",
         "## Relay interaction direction (known cascades + control)\n",
         "| pair | kind | T_A | T_B | M^A[T_B,T_A] | M^B[T_A,T_B] | D | call |\n",
         "|---|---|---|---|---:|---:|---:|---|\n"]
    for r in relay_rows:
        L.append(f"| {r['A']}→{r['B']} | {r['kind']} | {r['T_A']} | {r['T_B']} | "
                 f"{(r['m_A_BtoA'] or float('nan')):.4f} | {(r['m_B_AtoB'] or float('nan')):.4f} | "
                 f"{r['D']:.4f} | {r['call']} |\n")
    n_cas = [r for r in relay_rows if r["kind"] == "cascade"]
    n_correct = sum(1 for r in n_cas if r["call"] == "a_to_b")
    L.append(f"\nKnown cascades pointing A→B: **{n_correct}/{len(n_cas)}**\n\n")
    L.append("## Top directed cell-type interactions (final epoch)\n")
    for cyt in KEY_CYTOKINES:
        if cyt not in inter["interaction"]:
            continue
        dp = directed_pairs(inter["interaction"][cyt], top_k=5)
        L.append(f"\n**{cyt}** — {dp['metric_description'][:80]}...\n")
        for src, dst, asym in dp["pairs"]:
            arrow = "→" if asym > 0 else "←"
            L.append(f"- {src} {arrow} {dst}  (|asym|={abs(asym):.4f})\n")
    L.append("\n> Interaction direction ≠ existence (coupling is Path A's job) ≠ causation. "
             "Attention is task-driven, not biology. Seed-noisy — multi-seed before trusting.\n")

    out = Path(args.report) if args.report else run_dir / "SELFATTN_INTERACTION_RESULTS.md"
    out.write_text("".join(L))
    print(f"Saved: {out}")
    print(f"G0 gate: {'PASS' if offd_summary['gate_pass'] else 'FAIL'} "
          f"(mean_final={offd_summary['mean_final']:.3f})")
    print(f"Known cascades A→B: {n_correct}/{len(n_cas)}")


if __name__ == "__main__":
    main()
