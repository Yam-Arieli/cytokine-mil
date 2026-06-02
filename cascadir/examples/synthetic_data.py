"""Synthetic single-cell data with a *planted* cascade, for examples and tests.

We plant a known cascade ``upstream -> downstream`` so the ground-truth answer is
known: the upstream stimulus's cells carry **both** its own program and the
downstream program (autocrine relay), while the downstream stimulus's cells carry
only their own program. cross_asym must then call the upstream as upstream.

The matrix is raw integer counts (Poisson) so ``preprocess(assume="raw")`` applies
the real normalize -> log1p path. Gene blocks:
    genes [0:K)        -> upstream program  (high in upstream cells only)
    genes [K:2K)       -> downstream program (high in upstream AND downstream cells)
    genes [2K:)        -> background noise
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd


def make_synthetic_anndata(
    *,
    upstream: str = "CytA",
    downstream: str = "CytB",
    control: str = "PBS",
    n_donors: int = 3,
    n_per_group: int = 40,
    n_program_genes: int = 10,
    n_noise_genes: int = 40,
    base_rate: float = 1.0,
    program_rate: float = 8.0,
    cell_types: tuple[str, ...] = ("Mono", "NK"),
    seed: int = 0,
) -> ad.AnnData:
    """Return a raw-count AnnData with a planted ``upstream -> downstream`` cascade.

    Args:
        upstream / downstream / control: Condition labels. (``upstream`` should sort
            before ``downstream`` if you want the default ``a_to_b`` convention.)
        n_donors: Number of donors (>= 3 to pass validation).
        n_per_group: Cells per (donor, condition, cell_type).
        n_program_genes: Genes per program block (K).
        n_noise_genes: Number of background genes.
        base_rate / program_rate: Poisson means for background / elevated program.
        cell_types: Cell-type labels to stratify by.
        seed: RNG seed.

    Returns:
        AnnData (cells x genes) with ``obs`` columns ``cytokine``, ``donor``,
        ``cell_type`` and integer counts in ``X``.
    """
    rng = np.random.default_rng(seed)
    k = n_program_genes
    n_genes = 2 * k + n_noise_genes
    up_block = slice(0, k)
    down_block = slice(k, 2 * k)

    conditions = [control, upstream, downstream]
    rows_X: list[np.ndarray] = []
    obs_cond: list[str] = []
    obs_donor: list[str] = []
    obs_ct: list[str] = []

    for d in range(n_donors):
        donor = f"donor{d + 1}"
        for cond in conditions:
            for ct in cell_types:
                lam = np.full((n_per_group, n_genes), base_rate, dtype=np.float64)
                if cond == upstream:
                    lam[:, up_block] = program_rate      # own program
                    lam[:, down_block] = program_rate    # autocrine downstream relay
                elif cond == downstream:
                    lam[:, down_block] = program_rate     # own program only
                # control stays at base_rate everywhere
                counts = rng.poisson(lam).astype(np.float32)
                rows_X.append(counts)
                obs_cond += [cond] * n_per_group
                obs_donor += [donor] * n_per_group
                obs_ct += [ct] * n_per_group

    X = np.concatenate(rows_X, axis=0)
    obs = pd.DataFrame(
        {"cytokine": obs_cond, "donor": obs_donor, "cell_type": obs_ct}
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["planted_cascade"] = {
        "upstream": upstream,
        "downstream": downstream,
        "upstream_program_genes": [f"gene{i}" for i in range(0, k)],
        "downstream_program_genes": [f"gene{i}" for i in range(k, 2 * k)],
    }
    return adata
