"""Two controls on the direction statistic, computed from sanctioned cascadir outputs only.

Both quantities were quoted in working notes without a source record; this script is that
record. Nothing here recomputes coupling or ``cross_asym`` by hand (see the
``cascadir-values`` skill) - the direction values are read from the packaged benchmark and
from the committed per-axis/direction tables, and the one aggregation applied is the
documented median-across-cell-types rule.

Control 1 - the asymmetric magnitude baseline.
    sign( median_T [ |s_T(a, S_a)| - |s_T(b, S_b)| ] )
    "which condition responds more strongly to its own signature", made antisymmetric and
    aggregated exactly the way cross_asym is. Unlike the symmetric control it *could* score
    above chance, so it is the statistic that tests the "a simply responded harder"
    explanation of the direction recall.

Control 2 - the additive-potential (rank-1) fit.
    Fit one scalar theta per condition minimising sum over observed pairs of
    (cross_asym(a,b) - (theta_a - theta_b))^2, and report the fraction of ||A||^2 it
    explains. A high fraction means the pairwise calls are close to a single global ordering
    of conditions rather than pair-specific judgements - but only relative to the null
    reported alongside it: a panel with barely more pairs than conditions leaves the fit
    almost no freedom to fail, and scores high on random values too.

Usage:  python scripts/compute_direction_controls.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "reports" / "direction_controls"

# --- sanctioned inputs (see .claude/skills/cascadir-values/references/result_files.md) ---
OES_BENCHMARK = REPO / "cascadir/examples/analysis/oesinghaus_per_axis.csv"
OES_PER_CELLTYPE = REPO / "results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv"
SHEU_NATIVE_DIRECTION = REPO / "results/sheu_cascadir_native/5hr/direction_table.csv"
SHEU_LEGACY_PER_AXIS = REPO / "results/sheu_cascade/5hr/pathB/per_axis_summary.csv"
ID_PER_AXIS = REPO / "reports/immune_dictionary/per_axis_summary.csv"

# The thesis scores the 17 audited "strict" pairs less IL-2/IL-15, which was dropped for
# carrying no literature of its own (thesis.tex, appendix "How the directional benchmark
# was labeled").
DROPPED_PAIR = ("IL-15", "IL-2")


def load_oesinghaus_benchmark() -> pd.DataFrame:
    """The 16 pairs the thesis headline is scored on, with their published cross_asym."""
    df = pd.read_csv(OES_BENCHMARK)
    df = df[df["benchmark"] == "strict"].copy()
    keep = ~((df["axis_a"] == DROPPED_PAIR[0]) & (df["axis_b"] == DROPPED_PAIR[1]))
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Control 1 - the asymmetric magnitude baseline
# --------------------------------------------------------------------------------------

def magnitude_baseline_per_axis(per_celltype: pd.DataFrame) -> pd.DataFrame:
    """Median across cell types of |s(a, S_a)| - |s(b, S_b)|, one row per ordered axis."""
    df = per_celltype.copy()
    df["magnitude_diff"] = df["sA_PA_norm"].abs() - df["sB_PB_norm"].abs()
    grouped = df.groupby(["axis_a", "axis_b"], as_index=False).agg(
        magnitude_median=("magnitude_diff", "median"),
        n_cell_types=("magnitude_diff", "size"),
    )
    return grouped


def score_signs(values: np.ndarray, expected: np.ndarray) -> tuple[int, int]:
    """Count sign matches; a value of exactly zero never matches."""
    called = np.sign(values)
    return int(np.sum((called != 0) & (called == np.sign(expected)))), int(len(values))


def run_magnitude_baseline() -> tuple[pd.DataFrame, dict]:
    bench = load_oesinghaus_benchmark()
    per_ct = pd.read_csv(OES_PER_CELLTYPE)
    mag = magnitude_baseline_per_axis(per_ct)

    merged = bench.merge(mag, on=["axis_a", "axis_b"], how="left")
    missing = merged["magnitude_median"].isna().sum()
    if missing:
        raise SystemExit(
            f"{missing} benchmark pairs absent from {OES_PER_CELLTYPE.name}; "
            "the panels no longer line up - stop and check rather than scoring a subset."
        )

    merged["magnitude_sign"] = np.sign(merged["magnitude_median"]).astype(int)
    merged["magnitude_correct"] = merged["magnitude_sign"] == merged["expected_sign"]

    n_mag, n_total = score_signs(
        merged["magnitude_median"].to_numpy(), merged["expected_sign"].to_numpy()
    )
    n_cross = int(merged["correct"].sum())

    summary = {
        "n_pairs": n_total,
        "cross_asym_correct": n_cross,
        "cross_asym_recall": round(n_cross / n_total, 4),
        "magnitude_correct": n_mag,
        "magnitude_recall": round(n_mag / n_total, 4),
        "agreement_with_cross_asym_sign": int(
            (merged["magnitude_sign"] == np.sign(merged["cross_asym_median"]).astype(int)).sum()
        ),
    }
    cols = [
        "axis_a", "axis_b", "expected_direction", "expected_sign",
        "cross_asym_median", "correct", "magnitude_median", "n_cell_types",
        "magnitude_sign", "magnitude_correct",
    ]
    return merged[cols].rename(columns={"correct": "cross_asym_correct"}), summary


# --------------------------------------------------------------------------------------
# Control 2 - the additive-potential (rank-1) fit
# --------------------------------------------------------------------------------------

def fit_potential(pairs: pd.DataFrame) -> tuple[pd.Series, float]:
    """Least-squares theta with A[a,b] ~ theta_a - theta_b, over the observed pairs only.

    Solves the graph-Laplacian normal equations with a pseudo-inverse, which handles the
    incomplete-panel case (not every pair is observed) and the constant shift theta is
    defined up to. Returns theta and the fraction of sum(A^2) the fit explains.
    """
    conditions = sorted(set(pairs["a"]) | set(pairs["b"]))
    index = {c: i for i, c in enumerate(conditions)}
    n = len(conditions)

    laplacian = np.zeros((n, n))
    rhs = np.zeros(n)
    for a, b, value in zip(pairs["a"], pairs["b"], pairs["value"]):
        i, j = index[a], index[b]
        laplacian[i, i] += 1.0
        laplacian[j, j] += 1.0
        laplacian[i, j] -= 1.0
        laplacian[j, i] -= 1.0
        rhs[i] += value
        rhs[j] -= value

    theta = np.linalg.pinv(laplacian) @ rhs
    theta -= theta.mean()

    observed = pairs["value"].to_numpy()
    fitted = np.array([theta[index[a]] - theta[index[b]] for a, b in zip(pairs["a"], pairs["b"])])
    total = float(np.sum(observed ** 2))
    residual = float(np.sum((observed - fitted) ** 2))
    explained = 1.0 - residual / total if total > 0 else float("nan")
    return pd.Series(theta, index=conditions), explained


def oesinghaus_cross_asym_per_axis(per_celltype: pd.DataFrame) -> pd.DataFrame:
    """cross_asym per axis: median across cell types of sA_PB_norm - sB_PA_norm.

    The documented aggregation rule, applied to the committed per-cell-type values - the
    catalog states explicitly that this file carries no pre-derived cross_asym column and
    that this is how to obtain one.
    """
    df = per_celltype.copy()
    df["cross_asym"] = df["sA_PB_norm"] - df["sB_PA_norm"]
    return df.groupby(["axis_a", "axis_b"], as_index=False).agg(
        cross_asym_median=("cross_asym", "median")
    )


def null_explained_fraction(pairs: pd.DataFrame, n_draws: int = 400, seed: int = 0) -> float:
    """Explained fraction the same fit reaches on random values over the same edge set.

    The fit has (n_conditions - 1) free parameters against n_pairs observations, so a panel
    whose pair count barely exceeds its condition count scores high on noise. This is that
    floor, measured rather than assumed.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_draws):
        shuffled = pairs.copy()
        shuffled["value"] = rng.standard_normal(len(pairs))
        _, explained = fit_potential(shuffled)
        scores.append(explained)
    return float(np.mean(scores))


def collect_panels() -> dict[str, pd.DataFrame]:
    """One (a, b, value) frame per panel, value = cross_asym(a, b)."""
    panels: dict[str, pd.DataFrame] = {}

    oes = oesinghaus_cross_asym_per_axis(pd.read_csv(OES_PER_CELLTYPE))
    panels["Oesinghaus (53 axes)"] = oes.rename(
        columns={"axis_a": "a", "axis_b": "b", "cross_asym_median": "value"}
    )

    sheu_native = pd.read_csv(SHEU_NATIVE_DIRECTION)
    panels["Sheu 5h, cascadir-native (21 pairs)"] = sheu_native.rename(
        columns={"condition_a": "a", "condition_b": "b", "cross_asym_median": "value"}
    )[["a", "b", "value"]]

    sheu_legacy = pd.read_csv(SHEU_LEGACY_PER_AXIS)
    panels["Sheu 5h, legacy fit"] = sheu_legacy.rename(
        columns={"axis_a": "a", "axis_b": "b", "cross_median": "value"}
    )[["a", "b", "value"]]

    id_axes = pd.read_csv(ID_PER_AXIS)
    panels["Immune Dictionary"] = id_axes.rename(
        columns={"axis_a": "a", "axis_b": "b", "cross_median": "value"}
    )[["a", "b", "value"]]

    return panels


def run_potential_fit(bench: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    thetas = []
    for name, pairs in collect_panels().items():
        pairs = pairs.dropna(subset=["value"])
        theta, explained = fit_potential(pairs)
        floor = null_explained_fraction(pairs)
        rows.append(
            {
                "panel": name,
                "n_conditions": len(theta),
                "n_pairs": len(pairs),
                "explained_fraction": round(explained, 4),
                "null_explained_fraction": round(floor, 4),
                "excess_over_null": round(explained - floor, 4),
            }
        )
        thetas.append(
            pd.DataFrame({"panel": name, "condition": theta.index, "theta": theta.to_numpy()})
        )

    # Residual direction recall on the Oesinghaus benchmark: how much of the direction call
    # survives once the single global ordering is removed.
    oes_pairs = collect_panels()["Oesinghaus (53 axes)"].dropna(subset=["value"])
    theta, _ = fit_potential(oes_pairs)
    merged = bench.merge(
        oes_pairs.rename(columns={"a": "axis_a", "b": "axis_b", "value": "cross_recomputed"}),
        on=["axis_a", "axis_b"],
        how="left",
    )
    merged["residual"] = merged["cross_recomputed"] - (
        merged["axis_a"].map(theta) - merged["axis_b"].map(theta)
    )
    n_res, n_tot = score_signs(
        merged["residual"].to_numpy(), merged["expected_sign"].to_numpy()
    )
    rows.append(
        {
            "panel": "Oesinghaus benchmark, residual after removing the potential",
            "n_conditions": len(theta),
            "n_pairs": n_tot,
            "explained_fraction": float("nan"),
            "residual_direction_recall": round(n_res / n_tot, 4),
            "residual_correct": n_res,
        }
    )
    return pd.DataFrame(rows), pd.concat(thetas, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_pair, summary = run_magnitude_baseline()
    per_pair.to_csv(OUT_DIR / "oesinghaus_magnitude_baseline.csv", index=False)

    bench = load_oesinghaus_benchmark()
    potential, thetas = run_potential_fit(bench)
    potential.to_csv(OUT_DIR / "potential_fit.csv", index=False)
    thetas.to_csv(OUT_DIR / "potential_theta.csv", index=False)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("=== Control 1: asymmetric magnitude baseline, Oesinghaus benchmark")
    print(json.dumps(summary, indent=2))
    print()
    print(per_pair.to_string(index=False))
    print()
    print("=== Control 2: additive-potential fit")
    print(potential.to_string(index=False))


if __name__ == "__main__":
    main()
