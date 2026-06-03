"""Signature-space coupling — the second coupling path, unified with cross_asym.

This is the "specific-dimensions" reframe. It builds **one** cross-engagement matrix
in GENE (signature) space and reads BOTH answers off it::

    M[a, b]        = s(a, S_b) - s(PBS, S_b)     # a's cells engaging b's signature,
                                                   PBS-normalized, median over cell types
    coupling(a, b) = M[a, b] + M[b, a]           # SYMMETRIC   -> are a,b mutually
                                                   engaged in each other's specific
                                                   programs?  (existence)
    cross_asym(a,b)= M[a, b] - M[b, a]           # ANTISYMMETRIC -> who is upstream?
                                                   (direction; identical to cross_asym.py)

``M[a, b]`` is exactly the ``sA_PB_norm`` quantity of
:func:`cascadir.cross_asym.directional_asymmetry_test`, so the ``cross_asym`` reported
here matches :func:`cascadir.cross_asym.direction_table` when conditions share cell types.

TWO COUPLING PATHS (pick by dataset; see the MANUAL):
  * **Latent-geometry coupling** (:func:`cascadir.coupling.discover_axes`) — coupling in
    the encoder EMBEDDING. Needs a broad gene panel + several donors. Worked on the broad
    human PBMC data; had **no power** on a targeted mouse panel (every q≈1).
  * **Signature-space coupling** (this module) — coupling in cytokine-SPECIFIC genes.
    Recovered the textbook cascades that latent geometry missed on the targeted panel.
    Its weakness is the mirror image: the gate **over-calls on broad data**.

HONEST CAVEAT — over-power. The cell-level gene-set null is **over-powered**: with
thousands of cells, almost any nonzero asymmetry is "significant", so ~everything passes
and the gate stops discriminating. **The unit of independence is the DONOR.** Use
``donor_level=True`` for an honest (under-powered, conservative) gate; treat the
cell-level ``coupling_null_p`` as exploratory only. This is an open methodological point,
not a settled gate — see the MANUAL.
"""

from __future__ import annotations

from math import comb
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from cascadir.config import CrossAsymConfig
from cascadir.types import Signature


# ---------------------------------------------------------------------------
# Cross-engagement matrix
# ---------------------------------------------------------------------------


def cross_engagement_matrix(
    cells_by_pair: dict[tuple[str, str], np.ndarray],
    sig_idx: dict[str, np.ndarray],
    *,
    control_label: str = "PBS",
    min_cells: int = 10,
) -> tuple[list[str], np.ndarray]:
    """Build ``M[i, j] = s(cyt_i, S_{cyt_j}) - s(PBS, S_{cyt_j})`` (median over cell types).

    Only conditions that (a) have a signature in ``sig_idx`` and (b) appear in
    ``cells_by_pair`` are kept (control excluded from the rows/cols). ``M[i, j]`` uses the
    cell types where both ``cyt_i`` and the control have ``>= min_cells``; ``NaN`` where
    none qualify.

    Returns ``(conditions, M)`` with ``M`` shape ``(n, n)`` ordered by ``conditions``.
    """
    conditions = sorted(
        c for c in sig_idx if c != control_label and any(k[0] == c for k in cells_by_pair)
    )
    n = len(conditions)
    sig_arrs = {c: np.asarray(sig_idx[c], dtype=np.int64) for c in conditions}
    cell_types = sorted({ct for (_, ct) in cells_by_pair})

    # E[t, i, j] then nanmedian over t
    E = np.full((len(cell_types), n, n), np.nan, dtype=np.float64)
    for t, T in enumerate(cell_types):
        cP = cells_by_pair.get((control_label, T))
        if cP is None or len(cP) < min_cells:
            continue
        pbs_score = np.array(
            [float(cP[:, sig_arrs[conditions[j]]].mean()) for j in range(n)],
            dtype=np.float64,
        )
        for i, a in enumerate(conditions):
            cA = cells_by_pair.get((a, T))
            if cA is None or len(cA) < min_cells:
                continue
            for j in range(n):
                E[t, i, j] = float(cA[:, sig_arrs[conditions[j]]].mean()) - pbs_score[j]
    with np.errstate(all="ignore"):
        M = np.nanmedian(E, axis=0)
    n_ct = np.sum(np.isfinite(E), axis=0)  # (n, n): cell types contributing to M[i,j]
    return conditions, M  # noqa: RET504  (n_ct computed for callers via finite check)


def _pair_rows(conditions: list[str], M: np.ndarray) -> list[dict]:
    rows = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            m_ab, m_ba = M[i, j], M[j, i]
            rows.append(
                {
                    "condition_a": conditions[i],
                    "condition_b": conditions[j],
                    "m_ab": float(m_ab),
                    "m_ba": float(m_ba),
                    "coupling": float(m_ab + m_ba),     # symmetric -> existence
                    "cross_asym": float(m_ab - m_ba),   # antisymmetric -> direction
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Gene-set null on the SYMMETRIC coupling (cell-level; exploratory)
# ---------------------------------------------------------------------------


def _coupling_null(
    cells_by_pair: dict[tuple[str, str], np.ndarray],
    conditions: list[str],
    *,
    set_size: int,
    n_genes: int,
    excluded_indices: set[int],
    control_label: str,
    min_cells: int,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    """Null coupling per pair under random gene sets (shared set per permutation).

    Returns ``(n_perm, n, n)`` null coupling = ``N[k,i] + N[k,j]`` where ``N[k,i]`` is
    condition i's PBS-normalized engagement of random set k (median over cell types).
    """
    pool = np.array(
        [i for i in range(n_genes) if i not in excluded_indices], dtype=np.int64
    )
    if len(pool) < set_size:
        raise ValueError(
            f"Null pool too small: {len(pool)} non-signature genes, need {set_size}."
        )
    rng = np.random.default_rng(seed)
    cell_types = sorted({ct for (_, ct) in cells_by_pair})
    n = len(conditions)
    rand_sets = [rng.choice(pool, size=set_size, replace=False) for _ in range(n_perm)]
    Eng = np.full((n_perm, len(cell_types), n), np.nan, dtype=np.float64)
    for t, T in enumerate(cell_types):
        cP = cells_by_pair.get((control_label, T))
        if cP is None or len(cP) < min_cells:
            continue
        for k, R in enumerate(rand_sets):
            pbs_r = float(cP[:, R].mean())
            for i, a in enumerate(conditions):
                cA = cells_by_pair.get((a, T))
                if cA is None or len(cA) < min_cells:
                    continue
                Eng[k, t, i] = float(cA[:, R].mean()) - pbs_r
    with np.errstate(all="ignore"):
        eng_med = np.nanmedian(Eng, axis=1)  # (n_perm, n)
    return eng_med[:, :, None] + eng_med[:, None, :]  # (n_perm, n, n)


# ---------------------------------------------------------------------------
# Public: signature-coupling table
# ---------------------------------------------------------------------------


def _signatures_to_idx(
    signatures: dict[str, Signature], gene_names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    gene_index = {g: i for i, g in enumerate(gene_names)}
    return {
        cond: np.array(
            [gene_index[g] for g in sig.genes if g in gene_index], dtype=np.int64
        )
        for cond, sig in signatures.items()
    }


def signature_coupling(
    cells_by_pair: dict[tuple[str, str], np.ndarray],
    signatures: dict[str, Signature],
    gene_names: tuple[str, ...],
    *,
    control_label: str = "PBS",
    config: CrossAsymConfig | None = None,
    cells_by_pair_per_donor: Optional[dict[str, dict[tuple[str, str], np.ndarray]]] = None,
    coupling_alpha: float = 0.05,
) -> pd.DataFrame:
    """Signature-space coupling + direction for every unordered pair.

    Args:
        cells_by_pair: ``{(condition, cell_type): (n_cells, n_genes)}`` pooled across donors.
        signatures: ``{condition: Signature}`` (the discovered ``S_X``).
        gene_names: gene order of the arrays.
        config: :class:`CrossAsymConfig` (``min_cells``, ``n_null_perms``, ``null_seed``,
            ``top_n`` for the null set size).
        cells_by_pair_per_donor: if given, ``{donor: cells_by_pair}`` — enables the
            **donor-level** gate (recommended): coupling is aggregated per donor and tested
            with a sign test across donors (conservative; respects effective N = donors).
        coupling_alpha: significance threshold for the ``coupled`` flag.

    Returns:
        DataFrame, one row per unordered pair, sorted by descending ``coupling``:
        ``condition_a, condition_b, coupling, cross_asym, coupling_null_p`` (cell-level,
        exploratory); and if donor-level: ``donor_coupling_mean, donor_consensus,
        donor_sign_p, n_donors``; plus ``coupled`` (bool by the best available gate).
    """
    cfg = config or CrossAsymConfig()
    sig_idx = _signatures_to_idx(signatures, gene_names)
    conditions, M = cross_engagement_matrix(
        cells_by_pair, sig_idx, control_label=control_label, min_cells=cfg.min_cells
    )
    if len(conditions) < 2:
        return pd.DataFrame(
            columns=["condition_a", "condition_b", "coupling", "cross_asym",
                     "coupling_null_p", "coupled"]
        )
    rows = _pair_rows(conditions, M)
    idx_of = {c: i for i, c in enumerate(conditions)}

    # cell-level gene-set null (exploratory; over-powered — see module docstring)
    null_cmat = None
    if cfg.n_null_perms and cfg.n_null_perms > 0:
        excluded = {i for idx in sig_idx.values() for i in np.asarray(idx).tolist()}
        sizes = [len(np.asarray(sig_idx[c])) for c in conditions]
        set_size = int(np.median(sizes)) if sizes else 0
        try:
            null_cmat = _coupling_null(
                cells_by_pair, conditions, set_size=set_size, n_genes=len(gene_names),
                excluded_indices=excluded, control_label=control_label,
                min_cells=cfg.min_cells, n_perm=cfg.n_null_perms, seed=cfg.null_seed,
            )
        except ValueError:
            null_cmat = None

    # donor-level coupling (recommended gate)
    donor_M: list[np.ndarray] = []
    if cells_by_pair_per_donor:
        for _d, cbp_d in cells_by_pair_per_donor.items():
            conds_d, M_d = cross_engagement_matrix(
                cbp_d, sig_idx, control_label=control_label, min_cells=cfg.min_cells
            )
            # re-index M_d onto the global `conditions` order (NaN for missing)
            full = np.full((len(conditions), len(conditions)), np.nan)
            local = {c: k for k, c in enumerate(conds_d)}
            for a in conds_d:
                for b in conds_d:
                    full[idx_of[a], idx_of[b]] = M_d[local[a], local[b]]
            donor_M.append(full)

    for r in rows:
        i, j = idx_of[r["condition_a"]], idx_of[r["condition_b"]]
        if null_cmat is not None:
            nc = null_cmat[:, i, j]
            nc = nc[np.isfinite(nc)]
            r["coupling_null_p"] = (
                float(np.mean(nc >= r["coupling"])) if nc.size else float("nan")
            )
        else:
            r["coupling_null_p"] = float("nan")
        if donor_M:
            cpl = np.array([dM[i, j] + dM[j, i] for dM in donor_M], dtype=np.float64)
            cpl = cpl[np.isfinite(cpl)]
            nd = cpl.size
            r["n_donors"] = int(nd)
            if nd:
                r["donor_coupling_mean"] = float(np.mean(cpl))
                n_pos = int(np.sum(cpl > 0))
                r["donor_consensus"] = float(n_pos / nd)
                # one-sided sign test: P(>= n_pos positives | p=0.5)
                r["donor_sign_p"] = float(
                    sum(comb(nd, k) for k in range(n_pos, nd + 1)) / (2 ** nd)
                )
            else:
                r["donor_coupling_mean"] = float("nan")
                r["donor_consensus"] = float("nan")
                r["donor_sign_p"] = float("nan")

    df = pd.DataFrame(rows)
    if donor_M:
        df["coupled"] = (df["donor_sign_p"] <= coupling_alpha) & (df["donor_coupling_mean"] > 0)
    else:
        df["coupled"] = df["coupling_null_p"] < coupling_alpha
    return df.sort_values("coupling", ascending=False).reset_index(drop=True)
