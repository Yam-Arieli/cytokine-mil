"""Rehydrate a fitted `cascadir.CascadeDirection` from the full-90 DAG artifacts.

The fit is staged across SLURM jobs, so the analysis stages never hold a live `fit()`
result. `CascadeDirection.from_artifacts` rebuilds exactly the state `fit()` leaves, which
is what keeps coupling and direction on the orchestrator (the bare module-level
`signature_coupling()` silently falls back to an over-powered cell-level null while still
returning a `coupled` column — cascadir/MANUAL.md §3.1).

No statistics here: this loads files and hands them to cascadir.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _full90_config as C  # noqa: E402
from cytokine_mil.analysis.full90_tube_io import load_tube_set, read_meta  # noqa: E402


def load_signatures(parquet_path, top_n: int = C.TOP_N) -> dict:
    """Read `signatures_all90.parquet` into cascadir's `Signature` dataclass.

    Selection is by the precomputed `rank_ig` written by `derive_signatures` — the IG
    values are read, never recomputed or re-ranked.
    """
    import pandas as pd

    from cascadir.types import Signature

    df = pd.read_parquet(parquet_path)
    out = {}
    for cond, sub in df.groupby("cytokine"):
        sub = sub.sort_values("rank_ig").head(top_n)
        out[str(cond)] = Signature(
            condition=str(cond),
            genes=tuple(str(g) for g in sub["gene"]),
            ig_scores=tuple(float(v) for v in sub["ig"]),
            top_n=top_n,
        )
    return out


def build_estimator(output_dir, verbose: bool = True):
    """Load tubes + signatures and return a fitted `CascadeDirection` (+ provenance dict)."""
    import time

    import cascadir as cd

    out = Path(output_dir)
    shard_dir = out / "tubes"
    tube_meta = read_meta(shard_dir)

    t0 = time.time()
    tube_set = load_tube_set(shard_dir)
    if verbose:
        n_cells = sum(t.n_cells for t in tube_set.tubes)
        print(f"[tubes] {len(tube_set.tubes)} tubes, {n_cells} cells, "
              f"{len(tube_set.donors)} donors, {len(tube_set.cell_types)} cell types "
              f"({time.time()-t0:.0f}s)", flush=True)

    signatures = load_signatures(out / "signatures_all90.parquet")
    if verbose:
        print(f"[signatures] {len(signatures)} conditions x {C.TOP_N} genes", flush=True)

    est = cd.CascadeDirection.from_artifacts(
        tube_set,
        signatures,
        condition_col=C.CONDITION_COL,
        donor_col=C.DONOR_COL,
        celltype_col=C.CELLTYPE_COL,
        control_label=C.CONTROL,
        cross_asym_config=C.cross_asym_config(),
        device="cpu",
        seed=C.SEED,
    )
    provenance = {
        "tubes_shards_sha256": tube_meta["shards_sha256"],
        "n_tubes": tube_meta["n_tubes"],
        "n_conditions": len(signatures),
        "donors": list(tube_set.donors),
        "cell_types": list(tube_set.cell_types),
        "top_n": C.TOP_N,
    }
    return est, provenance
