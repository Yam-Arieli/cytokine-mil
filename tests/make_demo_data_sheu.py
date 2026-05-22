"""
Generate simulated demo data for the Sheu 2024 adapter (`build_pseudotubes_sheu2024.py`).

This represents the *input* to the Sheu adapter's pseudo-tube construction
step — i.e., after BD Rhapsody CSV parsing + samptag metadata join + QC +
normalization + log1p + global Leiden cell typing, but *before*
PBS-relabeling and pseudo-tube construction.

In real use, the adapter runs the full pipeline starting from BD Rhapsody
files. For testing, we skip the platform-specific I/O and synthesize a
small AnnData with the exact column convention the adapter expects:

  obs columns (required):
    - pseudo_donor : str       (e.g., "M0_rep1", "M0_rep2", "M1_IFNg_rep1")
    - cytokine     : str       (raw stimulus: "Unstim", "LPS", "polyIC", "IFNb")
                                  -- adapter will relabel "Unstim" -> "PBS"
                                  -- adapter will relabel any time_point=="0hr" -> "PBS"
    - cell_type    : str       (e.g., "mac_c0", "mac_c1")
    - time_point   : str       (e.g., "0hr", "3hr")
  obs columns (informative, optional):
    - sample_tag   : str
    - batch        : int

  X: log-normalized counts, dtype float32, shape (n_cells, n_genes).

Spec for the demo:
  - 3 pseudo-donors:
      M0_rep1      (train)
      M0_rep2      (train; both M0 reps in train so the §21 M0-only check is well-defined)
      M1_IFNg_rep1 (val   -- mirrors the real plan where val pseudo-donor is a
                              biological context not represented in train)
  - 4 cytokines (raw): Unstim, LPS, polyIC, IFNb
      -- after adapter relabel: PBS (from Unstim+0hr), LPS, polyIC, IFNb
  - 2 cell types: mac_c0, mac_c1
  - 2 time points: 0hr, 3hr
  - 25 cells per (pseudo_donor, raw_cytokine, cell_type, time_point) combo
  - 30 genes
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import anndata as ad
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Demo design (kept small for fast tests)
# ---------------------------------------------------------------------------
PSEUDO_DONORS_TRAIN = ["M0_rep1", "M0_rep2"]
PSEUDO_DONORS_VAL = ["M1_IFNg_rep1"]
PSEUDO_DONORS = PSEUDO_DONORS_TRAIN + PSEUDO_DONORS_VAL

# Raw cytokines as they appear in Sheu metadata (before adapter relabeling)
RAW_CYTOKINES = ["Unstim", "LPS", "polyIC", "IFNb"]

# What the adapter should produce after relabeling
EXPECTED_ACTIVE_CLASSES = ["PBS", "LPS", "polyIC", "IFNb"]

CELL_TYPES = ["mac_c0", "mac_c1"]
TIME_POINTS = ["0hr", "3hr"]

N_GENES = 30
N_CELLS_PER_COMBO = 25  # per (pseudo_donor, cytokine, cell_type, time_point)


def make_demo_adata_sheu(seed: int = 42) -> ad.AnnData:
    """
    Build the synthetic input AnnData for the Sheu adapter's pseudo-tube step.

    Returns an AnnData with `obs[pseudo_donor, cytokine, cell_type, time_point]`
    and a small log-normalized count matrix. No PBS relabeling has been
    applied — that's the adapter's job.
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"gene_{i:03d}" for i in range(N_GENES)]

    rows = []
    Xs = []
    for pd_name in PSEUDO_DONORS:
        for cyt in RAW_CYTOKINES:
            for ct in CELL_TYPES:
                for tp in TIME_POINTS:
                    # Unstim only meaningfully exists at 0hr;
                    # other stimuli only meaningfully at 3hr.
                    if cyt == "Unstim" and tp != "0hr":
                        continue
                    if cyt != "Unstim" and tp != "3hr":
                        continue

                    n = N_CELLS_PER_COMBO
                    # Log-normalized synthetic expression with mild
                    # cytokine-specific signal on the first 5 genes.
                    X = rng.lognormal(
                        mean=0.5, sigma=1.0, size=(n, N_GENES)
                    ).astype(np.float32)
                    cyt_idx = RAW_CYTOKINES.index(cyt)
                    X[:, cyt_idx : cyt_idx + 1] += rng.normal(
                        0.5, 0.1, size=(n, 1)
                    ).astype(np.float32)

                    Xs.append(X)
                    for i in range(n):
                        rows.append(
                            {
                                "pseudo_donor": pd_name,
                                "cytokine": cyt,
                                "cell_type": ct,
                                "time_point": tp,
                                "sample_tag": f"SampleTag{(cyt_idx + 1):02d}_mm",
                                "batch": (cyt_idx % 4) + 1,
                            }
                        )

    X_full = np.concatenate(Xs, axis=0)
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i:06d}" for i in range(len(obs))]
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=X_full, obs=obs, var=var)

    # Shuffle cell order so cell type and donor are not implicit in row order.
    perm = rng.permutation(adata.n_obs)
    adata = adata[perm].copy()
    return adata


def write_demo_anndata(out_dir: str, seed: int = 42) -> str:
    """
    Write the demo AnnData to <out_dir>/sheu_demo_input.h5ad and return the path.

    This is the file that the adapter's pseudo-tube step consumes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = make_demo_adata_sheu(seed=seed)
    out_path = out_dir / "sheu_demo_input.h5ad"
    adata.write_h5ad(str(out_path))
    return str(out_path)


if __name__ == "__main__":
    import sys
    import tempfile

    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="sheu_demo_")
    path = write_demo_anndata(out_dir)
    print(f"Sheu demo input AnnData written to: {path}")
