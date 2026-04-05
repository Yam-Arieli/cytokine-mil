"""
Oelen 2022 SA + CA results inspector.

Mirrors notebooks/inspect_oelen_sa_ca_results.ipynb but runs as a standalone
script on locally-pulled pkl files, saving all figures to disk.

Usage
-----
# With only Stage 2 data (Stage 3 not yet complete):
python scripts/inspect_oelen_sa_ca_results.py --results-dir results/oelen_sa_ca/

# After Stage 3 completes and you've pulled dynamics_stage3.pkl too:
python scripts/inspect_oelen_sa_ca_results.py --results-dir results/oelen_sa_ca/

# Custom output directory for figures:
python scripts/inspect_oelen_sa_ca_results.py \
    --results-dir results/oelen_sa_ca/ \
    --output-dir results/oelen_sa_ca/figures/
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from cytokine_mil.analysis.dynamics import (
    aggregate_to_donor_level,
    rank_cytokines_by_learnability,
    compute_confusion_entropy_summary,
    compute_cytokine_entropy_summary,
)


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

DIFFICULTY_MAP = {
    "24hPA":  "EASY",
    "24hCA":  "MED",
    "24hMTB": "HARD",
    "UT":     "CONTROL",
}

ORDERED_CONDITIONS = ["24hPA", "24hCA", "24hMTB", "UT"]

COLOR_MAP = {
    "24hPA":  "steelblue",
    "24hCA":  "darkorange",
    "24hMTB": "tomato",
    "UT":     "gray",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _print_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def _print_ranking(ranking: list, metric_desc: str, label: str) -> None:
    print(f"\nCondition learnability ranking — {label}")
    print(f"Metric: {metric_desc}")
    print()
    print(f"{'Rank':>4}  {'Condition':<12}  {'Train AUC':>9}  Difficulty")
    print("-" * 46)
    for i, (cond, auc) in enumerate(ranking, 1):
        difficulty = DIFFICULTY_MAP.get(cond, "")
        print(f"  {i:2d}.  {cond:<12}  {auc:>9.3f}  {difficulty}")


def _compute_normalized_auc(trajectory: list) -> float:
    """AUC of p_correct(t) / max(p_correct(t)) — shape metric."""
    arr = np.array(trajectory)
    max_val = arr.max()
    if max_val < 1e-10:
        return 0.0
    normed = arr / max_val
    n = len(normed)
    return float(np.trapz(normed) / max(n - 1, 1))


def _aggregate_scatter_metrics(records: list) -> tuple:
    """Per-condition normalized AUC and final-P, donor-aggregated."""
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        grouped[rec["cytokine"]][rec["donor"]].append(rec["p_correct_trajectory"])

    norm_auc_map: dict = {}
    p_final_map:  dict = {}
    for cond, donor_dict in grouped.items():
        donor_norm_aucs, donor_p_finals = [], []
        for trajs in donor_dict.values():
            median_traj = np.median(trajs, axis=0)
            donor_norm_aucs.append(_compute_normalized_auc(median_traj))
            donor_p_finals.append(float(median_traj[-1]))
        norm_auc_map[cond] = float(np.mean(donor_norm_aucs))
        p_final_map[cond]  = float(np.mean(donor_p_finals))
    return norm_auc_map, p_final_map


def _extract_layer_entropy(records: list, cytokine: str) -> tuple:
    """SA and CA entropy curves (one array per tube) for one condition."""
    sa_curves, ca_curves = [], []
    for rec in records:
        if rec["cytokine"] != cytokine:
            continue
        sa_traj = rec.get("entropy_trajectory")
        ca_traj = rec.get("entropy_trajectory_ca")
        if sa_traj is None or ca_traj is None:
            continue
        sa_curves.append(np.array(sa_traj))
        ca_curves.append(np.array(ca_traj))
    return sa_curves, ca_curves


# ---------------------------------------------------------------------------
# Figure routines
# ---------------------------------------------------------------------------

def fig_learning_curves(records: list, logged_epochs: list,
                        stage_label: str, out_dir: Path,
                        val_records: list | None = None) -> Path:
    """Learning curves per condition. Train = solid, val = dashed (same colour)."""
    train_donor_traj = aggregate_to_donor_level(records)
    val_donor_traj   = aggregate_to_donor_level(val_records) if val_records else {}
    tab10 = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, cond in enumerate(ORDERED_CONDITIONS):
        color = COLOR_MAP.get(cond, tab10[ci % len(tab10)])
        label_base = f"{cond}  ({DIFFICULTY_MAP.get(cond, '')})"

        if cond in train_donor_traj:
            train_mean = np.mean(list(train_donor_traj[cond].values()), axis=0)
            ax.plot(logged_epochs, train_mean, color=color, linewidth=2,
                    label=f"{label_base} — train")

        if cond in val_donor_traj:
            val_mean = np.mean(list(val_donor_traj[cond].values()), axis=0)
            ax.plot(logged_epochs, val_mean, color=color, linestyle="--",
                    linewidth=1.5, alpha=0.8, label=f"{label_base} — val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("P(Y_correct | t)")
    ax.set_title(f"{stage_label} — Oelen 2022 pathogen conditions")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    val_note = "Val donors held out (observer-only): donor_6.0, donor_95.0" if val_records else "No val split."
    fig.suptitle(
        "Metric: mean p_correct_trajectory(t), aggregated to donor level\n"
        f"(median per donor, mean across donors). {val_note}",
        fontsize=8,
    )
    plt.tight_layout()

    suffix = stage_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path = out_dir / f"learning_curves_{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_loss_and_ca_norm(loss_traj: list, ca_norm_traj: list,
                         out_dir: Path) -> Path:
    """Stage 3 training loss and CA weight norm side by side."""
    epochs = list(range(1, len(loss_traj) + 1))
    init_norm  = ca_norm_traj[0]
    final_norm = ca_norm_traj[-1]
    delta_norm = final_norm - init_norm

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(epochs, loss_traj, color="steelblue", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean cross-entropy loss")
    axes[0].set_title("Stage 3 CA-only — Training loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, ca_norm_traj, color="darkviolet", lw=2)
    axes[1].axhline(init_norm, color="gray", ls="--", lw=0.8,
                    label=f"Initial = {init_norm:.4f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L2 norm of CA parameters")
    axes[1].set_title(
        f"CA weight norm  |  init={init_norm:.4f}  "
        f"final={final_norm:.4f}  Δ={delta_norm:+.4f}"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Stage 3 CA-only diagnostics — Oelen 2022 pathogen\n"
        "Metric: L2 norm of [V_ca.weight, V_ca.bias, w_ca.weight, U_ca.weight] concatenated",
        fontsize=9,
    )
    plt.tight_layout()
    out_path = out_dir / "loss_and_ca_norm_stage3.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_sa_vs_ca_entropy(records: list, logged_epochs: list,
                         out_dir: Path) -> Path:
    """1×4 grid: SA (blue) vs CA (orange) attention entropy per condition."""
    n_cond = len(ORDERED_CONDITIONS)
    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 4), sharey=True)

    for ax, cond in zip(axes, ORDERED_CONDITIONS):
        difficulty = DIFFICULTY_MAP.get(cond, "")
        sa_curves, ca_curves = _extract_layer_entropy(records, cond)
        sa_delta = 0.0
        if sa_curves:
            mean_sa = np.mean(sa_curves, axis=0)
            mean_ca = np.mean(ca_curves, axis=0)
            sa_delta = float(np.ptp(mean_sa))
            ax.plot(logged_epochs, mean_sa, color="steelblue", linewidth=2,
                    label="SA (frozen)")
            ax.plot(logged_epochs, mean_ca, color="darkorange", linewidth=2,
                    linestyle="--", label="CA (trainable)")
        ax.set_title(f"{cond}\n({difficulty})  SA Δ={sa_delta:.4f}", fontsize=9)
        ax.set_xlabel("Epoch (Stage 3)")
        ax.set_ylabel("H(a) [nats]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Stage 3 CA-only — SA vs CA attention entropy per condition  |  Oelen 2022\n"
        "Metric: H(a) = -sum_i a_i log(a_i). SA is frozen (should be flat, Δ ≈ 0); "
        "CA changes only if it learned signal.\nMean across pseudo-tubes per condition.",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = out_dir / "sa_vs_ca_entropy_stage3.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_scatter_normauc_pmax(records_s2: list, logged_s2: list,
                              records_s3: list | None, logged_s3: list | None,
                              out_dir: Path) -> Path:
    """Normalized AUC vs P_max scatter for Stage 2 (and Stage 3 if available)."""
    stages = [("Stage 2 (SA)", records_s2, logged_s2)]
    if records_s3 is not None:
        stages.append(("Stage 3 CA-only", records_s3, logged_s3))

    fig, axes = plt.subplots(1, len(stages), figsize=(6.5 * len(stages), 5))
    if len(stages) == 1:
        axes = [axes]

    for ax, (stage_label, records, _logged) in zip(axes, stages):
        norm_auc_map, p_final_map = _aggregate_scatter_metrics(records)
        for cond in ORDERED_CONDITIONS:
            if cond not in norm_auc_map:
                continue
            x = norm_auc_map[cond]
            y = p_final_map[cond]
            color = COLOR_MAP.get(cond, "gray")
            ax.scatter(x, y, color=color, s=140, zorder=3)
            ax.annotate(
                f"{cond}\n({DIFFICULTY_MAP.get(cond, '')})",
                (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8,
            )
        ax.set_xlabel(
            "Normalized trajectory AUC  [shape metric]\n"
            "AUC(p_correct(t) / max(p_correct(t))), donor-aggregated"
        )
        ax.set_ylabel(
            "Final P(Y_correct)  [ceiling metric]\n"
            "p_correct(t_final), donor-aggregated"
        )
        ax.set_title(f"{stage_label} — Oelen 2022")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axvline(0.5, color="gray", ls=":", lw=0.8)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Learnability scatter — Normalized AUC (shape) vs Final P (ceiling)\n"
        "Metric: AUC(norm_p_correct_trajectory) and p_correct(t_final), "
        "aggregated to donor level (median per donor, mean across donors)",
        fontsize=9,
    )
    plt.tight_layout()
    out_path = out_dir / "scatter_normauc_pmax.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Analysis routines
# ---------------------------------------------------------------------------

def analyse_stage2(dyn2: dict) -> tuple:
    """Print Stage 2 learnability ranking. Returns (donor_traj, ranking)."""
    _print_section("Stage 2 (SA) Learnability Ranking")
    records_s2    = dyn2["records"]
    logged_s2     = dyn2["logged_epochs"]
    donor_traj_s2 = aggregate_to_donor_level(records_s2)
    learn_s2      = rank_cytokines_by_learnability(donor_traj_s2, exclude=[])
    _print_ranking(learn_s2["ranking"], learn_s2["metric_description"], "Stage 2 (SA)")
    return donor_traj_s2, learn_s2["ranking"]


def analyse_stage3(dyn3: dict) -> tuple:
    """Print Stage 3 learnability and confusion rankings. Returns (donor_traj, ranking)."""
    _print_section("Stage 3 (CA-only) Learnability Ranking")
    records_s3    = dyn3["records"]
    donor_traj_s3 = aggregate_to_donor_level(records_s3)
    learn_s3      = rank_cytokines_by_learnability(donor_traj_s3, exclude=[])
    ranking_s3    = learn_s3["ranking"]
    _print_ranking(ranking_s3, learn_s3["metric_description"], "Stage 3 CA-only")

    # Biological validation
    auc_map = {c: a for c, a in ranking_s3}
    pa_auc  = auc_map.get("24hPA",  float("nan"))
    ca_auc  = auc_map.get("24hCA",  float("nan"))
    mtb_auc = auc_map.get("24hMTB", float("nan"))
    print()
    print("Biological ordering validation (pre-registered: PA > CA > MTB):")
    print(f"  PA > CA:  {pa_auc > ca_auc}   (expected: True)")
    print(f"  CA > MTB: {ca_auc > mtb_auc}  (expected: True)")
    print(f"  PA > MTB: {pa_auc > mtb_auc}  (expected: True)")

    # SA entropy variance sanity check (should be ~0 for frozen SA)
    sample = next((r for r in records_s3 if r.get("entropy_trajectory")), None)
    if sample and len(sample["entropy_trajectory"]) > 1:
        sa_var = float(np.var(sample["entropy_trajectory"]))
        status = "OK" if sa_var < 1e-6 else "WARNING: SA may not be frozen!"
        print(f"\nSA entropy variance (first record): {sa_var:.2e}  [{status}]")

    # Confusion entropy
    _print_section("Confusion Entropy Ranking — Stage 3 CA-only")
    confusion_traj   = dyn3.get("confusion_entropy_trajectory", {})
    confusion_result = compute_confusion_entropy_summary(confusion_traj, exclude=[])
    print(f"Metric: {confusion_result['metric_description']}")
    print()
    print(f"{'Condition':<12}  {'Train AUC(H_c)':>14}  Difficulty")
    print("-" * 38)
    for cond, auc in confusion_result["ranking"]:
        print(f"  {cond:<12}  {auc:>14.3f}  {DIFFICULTY_MAP.get(cond, '')}")

    return donor_traj_s3, ranking_s3


def analyse_stage2_vs_stage3(ranking_s2: list, ranking_s3: list) -> None:
    """Print Stage 2 vs Stage 3 rank comparison and Spearman correlation."""
    _print_section("Stage 2 vs Stage 3 Rank Comparison")
    s2_order = [c for c, _ in ranking_s2]
    s3_order = [c for c, _ in ranking_s3]
    s2_auc   = {c: a for c, a in ranking_s2}
    s3_auc   = {c: a for c, a in ranking_s3}
    s3_rank  = {c: i for i, c in enumerate(s3_order)}

    if set(s2_order) == set(s3_order) and len(s2_order) >= 2:
        s3_aligned = [s3_rank.get(c, len(s3_order)) for c in s2_order]
        rho, pval  = spearmanr(range(len(s2_order)), s3_aligned)
        print(f"Spearman ρ (S2 vs S3): {rho:.3f}  (p = {pval:.3f})")
        print(f"Stable (ρ > 0.7): {rho > 0.7}")
        print()

    print(f"{'Condition':<12}  {'S2 AUC':>8}  {'S2 Rank':>7}  "
          f"{'S3 AUC':>8}  {'S3 Rank':>7}  {'ΔAUC':>7}  {'ΔRank':>6}  Difficulty")
    print("-" * 75)
    for i, cond in enumerate(s2_order, 1):
        s3r   = s3_rank.get(cond, i - 1) + 1
        da    = s3_auc.get(cond, float("nan")) - s2_auc.get(cond, float("nan"))
        dr    = s3r - i
        arrow = "↑" if dr < 0 else ("↓" if dr > 0 else "=")
        print(
            f"  {cond:<12}  {s2_auc.get(cond, float('nan')):>8.3f}  {i:>7d}  "
            f"{s3_auc.get(cond, float('nan')):>8.3f}  {s3r:>7d}  "
            f"{da:>+7.3f}  {dr:>5d}{arrow}  {DIFFICULTY_MAP.get(cond, '')}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Oelen 2022 SA + CA results from locally-pulled pkl files."
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Local directory containing dynamics_stage2.pkl (and optionally "
             "dynamics_stage3.pkl).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save figures (default: same as --results-dir).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output_dir) if args.output_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    s2_pkl = results_dir / "dynamics_stage2.pkl"
    s3_pkl = results_dir / "dynamics_stage3.pkl"

    if not s2_pkl.exists():
        print(f"ERROR: {s2_pkl} not found. Pull it first with cluster_pull.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Stage 2 dynamics from {s2_pkl} ...")
    dyn2 = _load_pkl(s2_pkl)
    records_s2 = dyn2["records"]
    logged_s2  = dyn2["logged_epochs"]
    print(f"  Stage 2 — logged epochs: {len(logged_s2)}, records: {len(records_s2)}")

    dyn3 = None
    if s3_pkl.exists():
        print(f"Loading Stage 3 dynamics from {s3_pkl} ...")
        dyn3 = _load_pkl(s3_pkl)
        records_s3 = dyn3["records"]
        logged_s3  = dyn3["logged_epochs"]
        print(f"  Stage 3 — logged epochs: {len(logged_s3)}, records: {len(records_s3)}")
    else:
        print(f"Stage 3 pkl not found at {s3_pkl} — skipping Stage 3 sections.")
        records_s3 = None
        logged_s3  = None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    _donor_traj_s2, ranking_s2 = analyse_stage2(dyn2)

    ranking_s3 = None
    if dyn3 is not None:
        _donor_traj_s3, ranking_s3 = analyse_stage3(dyn3)
        analyse_stage2_vs_stage3(ranking_s2, ranking_s3)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    _print_section("Saving Figures")
    saved: list[Path] = []

    val_records_s2 = dyn2.get("val_records") or None
    p = fig_learning_curves(records_s2, logged_s2, "Stage 2 (SA)", out_dir,
                            val_records=val_records_s2)
    print(f"  {p.name}")
    saved.append(p)

    if dyn3 is not None:
        val_records_s3 = dyn3.get("val_records") or None
        p = fig_learning_curves(records_s3, logged_s3, "Stage 3 CA-only", out_dir,
                                val_records=val_records_s3)
        print(f"  {p.name}")
        saved.append(p)

        p = fig_loss_and_ca_norm(
            dyn3["loss_trajectory"], dyn3["ca_weight_norm_trajectory"], out_dir
        )
        print(f"  {p.name}")
        saved.append(p)

        p = fig_sa_vs_ca_entropy(records_s3, logged_s3, out_dir)
        print(f"  {p.name}")
        saved.append(p)

    p = fig_scatter_normauc_pmax(records_s2, logged_s2, records_s3, logged_s3, out_dir)
    print(f"  {p.name}")
    saved.append(p)

    print(f"\n{len(saved)} figure(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
