"""
Attention training-dynamics analysis (CLAUDE.md §33) — depends on
scripts/extract_attention_trajectory.py.

Reads ``<run_dir>/attention_trajectory.pkl`` (+ optional ``<run_dir>/dynamics.pkl``
for p_correct), computes the three cell-type cascade readouts, evaluates the four
pre-registered predictions (reports/attention_dynamics/PRE_REGISTRATION.md),
renders the figure set, and writes the verdict report.

Readouts:
  1. primary/secondary responder map  (classify_primary_secondary)
  2. relay-recruitment-lag direction  (relay_recruitment_lag) on known cascades
  3. intra-cell-type concentration    (concentration_summary)
Plus P1 (attention-primary vs known direct responders) and P3 (primacy/subtlety).

Usage:
    python scripts/analyze_attention_dynamics.py \
        --run_dir results/oesinghaus_full/run_... \
        --report  reports/attention_dynamics/ATTENTION_DYNAMICS_RESULTS.md
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

from cytokine_mil.analysis.attention_dynamics import (  # noqa: E402
    EXPECTED_DOMINANT,
    attention_primary,
    attention_primary_vs_groundtruth,
    classify_primary_secondary,
    concentration_summary,
    p_correct_by_cytokine,
    primacy_subtlety_correlation,
    recruitment_order,
    relay_recruitment_lag,
)

# Default known directional cascades (A upstream -> B downstream), Fig 4f/i.
# Present in both the Oesinghaus 91-class set and the demo fixture.
DEFAULT_CASCADES = [("IL-12", "IFN-gamma"), ("IL-2", "IFN-gamma"), ("IL-15", "IFN-gamma")]
# Negative control: coupled only via shared activation (§28 negative), no relay.
DEFAULT_NEG_CONTROL = [("IL-6", "TNF"), ("IL-6", "TNF-alpha")]
KEY_CYTOKINES = ["IL-12", "IL-2", "IL-15", "IFN-gamma", "IL-4", "TNF", "TNF-alpha"]
RISE_FRAC = 0.5
N_BOOT = 1000
BOOT_SEED = 0


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--report", default=str(
        REPO_ROOT / "reports" / "attention_dynamics" / "ATTENTION_DYNAMICS_RESULTS.md"))
    p.add_argument("--plots_dir", default=None, help="Default: <run_dir>/plots")
    p.add_argument("--rise_frac", type=float, default=RISE_FRAC)
    return p.parse_args()


def _log(m=""):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load(run_dir: Path):
    with open(run_dir / "attention_trajectory.pkl", "rb") as f:
        at = pickle.load(f)
    records = None
    dyn_path = run_dir / "dynamics.pkl"
    if dyn_path.exists():
        with open(dyn_path, "rb") as f:
            records = pickle.load(f).get("records")
    return at, records


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_recruitment_ladder(at, plots_dir, rise_frac):
    epochs = at["epochs"]
    cyts = [c for c in KEY_CYTOKINES if c in at["trajectory"]]
    if not cyts:
        return
    n = len(cyts)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), squeeze=False)
    for ax, cyt in zip(axes[0], cyts):
        order = recruitment_order(at["trajectory"][cyt], epochs, rise_frac)["order"]
        order = order[:8]
        cts = [o[0] for o in order]
        taus = [o[1] if o[1] is not None else max(epochs) for o in order]
        ax.barh(range(len(cts))[::-1], taus, color="#4477aa")
        ax.set_yticks(range(len(cts))[::-1])
        ax.set_yticklabels(cts, fontsize=7)
        ax.set_xlabel("recruitment epoch τ")
        ax.set_title(cyt, fontsize=9)
    fig.suptitle("Attention recruitment order by cell type (earlier = primary)", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "recruitment_ladder.png", dpi=150)
    plt.close(fig)


def _fig_relay_lag(lag_rows, plots_dir):
    rows = [r for r in lag_rows if np.isfinite(r["mean_lag"])]
    if not rows:
        return
    labels = [f"{r['A']}→{r['B']}\n({r['kind']})" for r in rows]
    means = [r["mean_lag"] for r in rows]
    los = [r["mean_lag"] - r["ci_low"] for r in rows]
    his = [r["ci_high"] - r["mean_lag"] for r in rows]
    colors = ["#228833" if r["kind"] == "cascade" else "#bbbbbb" for r in rows]
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 2, 4))
    ax.bar(range(len(rows)), means, yerr=[los, his], color=colors, capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("relay-recruitment lag (epochs)\n>0 ⇒ A upstream")
    ax.set_title("Direction by relay-recruitment lag (green=cascade, grey=control)", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "relay_lag.png", dpi=150)
    plt.close(fig)


def _fig_primary_secondary(labels, plots_dir):
    cyts = sorted(labels)
    cts = sorted({ct for d in labels.values() for ct in d})
    if not cyts or not cts:
        return
    code = {"primary": 1.0, "secondary": -1.0, "minor": 0.0}
    M = np.array([[code.get(labels[c].get(ct, "minor"), 0.0) for ct in cts] for c in cyts])
    fig, ax = plt.subplots(figsize=(0.5 * len(cts) + 3, 0.4 * len(cyts) + 2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cyts))); ax.set_yticklabels(cyts, fontsize=6)
    ax.set_title("Primary (red) / secondary (blue) responder map", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, ticks=[-1, 0, 1])
    fig.tight_layout()
    fig.savefig(plots_dir / "primary_secondary_map.png", dpi=150)
    plt.close(fig)


def _fig_concentration(at, plots_dir):
    conc = at.get("concentration", {})
    epochs = at["epochs"]
    cyts = [c for c in KEY_CYTOKINES if c in conc]
    if not cyts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for cyt in cyts[:4]:
        prim = attention_primary(at["trajectory"].get(cyt, {}))
        if prim and prim in conc[cyt]:
            ax.plot(epochs, conc[cyt][prim], marker="o", label=f"{cyt} / {prim}")
    ax.set_xlabel("epoch"); ax.set_ylabel("within-type attention Gini")
    ax.set_title("Concentration of attention within the primary cell type", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "concentration.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    at, records = _load(run_dir)
    epochs = at["epochs"]
    traj = at["trajectory"]
    traj_pd = at["trajectory_per_donor"]
    _log(f"Loaded trajectory: {len(traj)} cytokines, {len(at['cell_types'])} cell types, "
         f"{len(epochs)} checkpoints (records={'yes' if records else 'no'}).")

    p_by_cyt = p_correct_by_cytokine(records) if records else None

    # --- Readout 1: primary/secondary map ---
    ps = classify_primary_secondary(traj, epochs, p_by_cyt, rise_frac=args.rise_frac)

    # --- P1: attention-primary vs known direct responders ---
    p1 = attention_primary_vs_groundtruth(traj, epochs, EXPECTED_DOMINANT,
                                          top_k=3, rise_frac=args.rise_frac)

    # --- Readout 2 / P2: relay-recruitment lag on cascades + neg control ---
    lag_rows = []
    for kind, pairs in (("cascade", DEFAULT_CASCADES), ("control", DEFAULT_NEG_CONTROL)):
        for A, B in pairs:
            if A in traj and B in traj:
                r = relay_recruitment_lag(traj, traj_pd, epochs, A, B,
                                          rise_frac=args.rise_frac,
                                          n_boot=N_BOOT, seed=BOOT_SEED)
                r["kind"] = kind
                lag_rows.append(r)
    casc = [r for r in lag_rows if r["kind"] == "cascade"]
    ctrl = [r for r in lag_rows if r["kind"] == "control"]
    n_casc_correct = sum(1 for r in casc if r["call"] == "A->B")
    n_ctrl_null = sum(1 for r in ctrl if r["call"] == "ambiguous")

    # --- Readout 3: concentration summary ---
    conc = concentration_summary(at.get("concentration", {}), epochs)

    # --- P3: primacy / subtlety ---
    p3 = {"rho": float("nan"), "n": 0}
    if records:
        directness = {c: float(np.trapz(t)) for c, t in p_by_cyt.items()}  # learnability AUC
        primary_tau = {}
        for c, by_ct in traj.items():
            prim = attention_primary(by_ct)
            if prim:
                from cytokine_mil.analysis.attention_dynamics import celltype_recruitment
                tau = celltype_recruitment({prim: by_ct[prim]}, epochs, args.rise_frac)[prim]["tau"]
                if tau is not None:
                    primary_tau[c] = tau
        p3 = primacy_subtlety_correlation(primary_tau, directness)

    # --- Figures ---
    _fig_recruitment_ladder(at, plots_dir, args.rise_frac)
    _fig_relay_lag(lag_rows, plots_dir)
    _fig_primary_secondary(ps["labels"], plots_dir)
    _fig_concentration(at, plots_dir)

    # --- Gates (mirror PRE_REGISTRATION.md) ---
    p1_green = (p1["frac_match"] >= 0.8) and (p1["frac_match_and_early"] >= 0.6)
    p1_amber = p1["frac_match"] >= 0.6
    p2_green = (len(casc) > 0 and n_casc_correct >= max(1, int(np.ceil(0.66 * len(casc))))
                and (len(ctrl) == 0 or n_ctrl_null >= 1))
    p2_amber = n_casc_correct >= 1
    p3_green = np.isfinite(p3["rho"]) and p3["rho"] < -0.1
    verdict = ("GREEN" if (p1_green and p2_green)
               else "AMBER" if (p1_amber or p2_amber or p3_green) else "RED")

    _write_report(Path(args.report), run_dir, at, records is not None,
                  ps, p1, p1_green, p1_amber, lag_rows, casc, ctrl,
                  n_casc_correct, n_ctrl_null, p2_green, p2_amber,
                  conc, p3, p3_green, verdict, plots_dir)
    _log(f"\nVerdict: {verdict}")
    _log(f"Report: {args.report}")
    _log(f"Plots:  {plots_dir}")


def _write_report(path, run_dir, at, has_records, ps, p1, p1_green, p1_amber,
                  lag_rows, casc, ctrl, n_casc_correct, n_ctrl_null,
                  p2_green, p2_amber, conc, p3, p3_green, verdict, plots_dir):
    path.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# Attention training-dynamics — cell-type-resolved cascade (CLAUDE.md §33)\n")
    L.append(f"Run: `{run_dir}` · checkpoints {at['epochs']} · "
             f"{len(at['trajectory'])} cytokines · {len(at['cell_types'])} cell types · "
             f"p_correct records: {'yes' if has_records else 'NO (P3 + secondary gate limited)'}\n")
    L.append(f"\n**VERDICT: {verdict}** (GREEN iff P1 and P2 both GREEN; "
             "gates locked in `reports/attention_dynamics/PRE_REGISTRATION.md`).\n")
    L.append(f"\n> Interpret strictly against the pre-registration (P1–P4 gates). "
             f"Checkpoints used: {at['epochs']} — the relay-lag/primacy readouts need the "
             f"full grid (every 10 epochs over a 250-epoch Stage-2) for temporal resolution; "
             f"few checkpoints make P2/P3 uninformative. Trajectories are point estimates "
             f"until multi-seed agreement (see caveats).\n")

    L.append("\n## P1 — attention-primary vs known direct responders\n")
    L.append(f"- frac_match (top-3): **{p1['frac_match']:.2f}** "
             f"({p1['n_evaluated']} cytokines); frac_match_and_early: "
             f"**{p1['frac_match_and_early']:.2f}** → "
             f"{'GREEN' if p1_green else 'AMBER' if p1_amber else 'RED'}\n")
    L.append("\n| cytokine | attention-primary | top-3 | expected | match | early |\n|---|---|---|---|---|---|\n")
    for c, r in p1["per_cytokine"].items():
        L.append(f"| {c} | {r['primary']} | {', '.join(r['top'])} | "
                 f"{', '.join(r['expected'])} | {'✓' if r['match'] else '✗'} | "
                 f"{'✓' if r['primary_early'] else '✗'} |\n")

    L.append("\n## P2 — relay-recruitment-lag direction\n")
    L.append(f"- cascades correct (A→B): **{n_casc_correct}/{len(casc)}**; "
             f"controls null (ambiguous): **{n_ctrl_null}/{len(ctrl)}** → "
             f"{'GREEN' if p2_green else 'AMBER' if p2_amber else 'RED'}\n")
    L.append("\n| pair | kind | relay T_B | mean_lag | 95% CI | sign_consist | n_donors | call |\n"
             "|---|---|---|---|---|---|---|---|\n")
    for r in lag_rows:
        ci = (f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
              if np.isfinite(r["ci_low"]) else "—")
        ml = f"{r['mean_lag']:.2f}" if np.isfinite(r["mean_lag"]) else "—"
        sc = f"{r['sign_consistency']:.2f}" if np.isfinite(r["sign_consistency"]) else "—"
        L.append(f"| {r['A']}→{r['B']} | {r['kind']} | {r['T_B']} | {ml} | {ci} | "
                 f"{sc} | {r['n_donors']} | {r['call']} |\n")

    L.append("\n## P3 — primacy / subtlety\n")
    L.append(f"- Spearman(primary recruitment τ, learnability-AUC directness): "
             f"rho=**{p3['rho']:.3f}** (n={p3['n']}) → "
             f"{'GREEN' if p3_green else 'n/a' if not has_records else 'not green'}\n")

    L.append("\n## Readout 3 — within-type concentration (top of each key cytokine)\n")
    L.append("\n| cytokine | primary cell type | final Gini | slope |\n|---|---|---|---|\n")
    for c in [c for c in KEY_CYTOKINES if c in at["trajectory"]]:
        prim = attention_primary(at["trajectory"][c])
        s = conc["summary"].get(c, {}).get(prim) if prim else None
        if s:
            L.append(f"| {c} | {prim} | {s['final']:.3f} | {s['slope']:+.2e} |\n")

    L.append("\n## Figures\n")
    for fn in ["recruitment_ladder.png", "relay_lag.png",
               "primary_secondary_map.png", "concentration.png"]:
        if (plots_dir / fn).exists():
            L.append(f"- `{plots_dir / fn}`\n")

    L.append("\n## Honest caveats (CLAUDE.md §33)\n")
    L.append("- Attention is task-driven (discriminative), not biology — validate against "
             "held-out donors; late recruitment ≠ secondary unless a p_correct second-rise "
             "co-occurs (lazy/redundant attention).\n")
    L.append("- Frozen-encoder representability: a secondary program is only visible if it "
             "lives in the cell-type-pretrained embedding subspace.\n")
    L.append("- Direction ≠ existence (coupling is Path A's job) and ≠ causation; small donor "
             "N → relay-lag CIs are wide; multi-seed before trusting ordering.\n")

    path.write_text("".join(L))


if __name__ == "__main__":
    main()
