"""Signature-space coupling — the coupling path built on the cross-engagement matrix.

This is the "specific-dimensions" reframe. It builds **one** cross-engagement matrix
in GENE (signature) space and reads coupling off it.

**The cross-engagement entry.** Within one cell type ``T``, condition ``a``'s engagement of
``b``'s signature is ``s_T(a, S_b) = mean(S_b over a's T-cells) - mean(S_b over PBS's
T-cells)``. The matrix entry is the median of that over cell types::

    M[a, b] = median_T s_T(a, S_b)

Each entry medians over the cell types where **``a`` and the control** each have
``>= min_cells``. ``b``'s presence is *not* required, so ``M[a, b]`` and ``M[b, a]`` may run
over **different** cell-type sets — this matters below.

**Coupling (the output of this module).** The raw symmetric coupling is::

    C[a, b] = M[a, b] + M[b, a]                  # -> the ``coupling_raw`` column

and the **reported, gated score** is its degree-corrected (additive double-centred)
residual, ``R[a,b] = C[a,b] - d_a - d_b + g`` (see :func:`_degree_center`) — the
``coupling`` column, on by default. Significance: a donor-level sign test (recommended) or
the cell-level gene-set null (over-powered; see the caveat below).

**Direction is NOT computed here.** The validated direction statistic lives in
:mod:`cascadir.cross_asym`: it is the **median over cell types of the per-cell-type
asymmetry** ``s_T(a, S_b) - s_T(b, S_a)``, taken only over cell types where ``a``, ``b``
*and* the control all qualify. Use :func:`cascadir.cross_asym.direction_table` for
direction — that is the path behind the validated accuracies.

The ``cross_asym`` column returned here is the **difference of medians**
``M[a, b] - M[b, a]``, which is a *fast approximation* of that statistic, not the same
number: the median is not linear, and the two entries may be medianed over different
cell-type sets (above). The two agree when the pair shares its scorable cell types and
diverge otherwise — measured on Sheu 5 h: 6 of 21 pairs differ in **sign**. It is kept for
backwards compatibility with existing result files; prefer ``direction_table``.

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
                    "coupling": float(m_ab + m_ba),     # symmetric -> existence (raw;
                                                        # degree-corrected below)
                    # Difference of medians: an APPROXIMATION of the direction statistic
                    # in cross_asym.py (which medians the per-cell-type difference over
                    # shared cell types). Kept for compatibility; use direction_table().
                    "cross_asym": float(m_ab - m_ba),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Degree (hub) correction
# ---------------------------------------------------------------------------


def _degree_center(C: np.ndarray) -> np.ndarray:
    """Additive double-centering of a SYMMETRIC coupling matrix (diagonal = NaN).

    ``R[i,j] = C[i,j] - d_i - d_j + g``, where ``d_i`` = mean off-diagonal coupling of
    condition ``i`` (its overall engagement "strength"/degree) and ``g`` = grand
    off-diagonal mean. Removes each condition's overall level — the HUB/DEGREE artifact
    where a broadly-engaged signature (e.g. IL-15) appears coupled to everything —
    leaving pair-SPECIFIC residual coupling. NaNs (absent pairs / diagonal) are excluded
    from the means.

    Validated to be the fix for the gate over-call in BOTH regimes (CLAUDE.md §28.1):
    donor-level on broad many-donor data (Oesinghaus: over-call 77%->31%, recall 8->11/17)
    and cell-level on a targeted few-donor panel (Sheu: preserves the 2/2 IFN cascades,
    suppresses all negatives, over-call ~80%->~40%). Being symmetric, it changes only the
    coupling (existence) half — direction is unaffected, both the ``cross_asym`` column
    here and :func:`cascadir.cross_asym.direction_table`, which never sees this matrix.
    """
    with np.errstate(all="ignore"):
        d = np.nanmean(C, axis=1)        # node strength (diagonal NaN -> excluded)
        g = float(np.nanmean(C))
    return C - d[:, None] - d[None, :] + g


def _coupling_matrix(M: np.ndarray) -> np.ndarray:
    """Symmetric coupling C[i,j] = M[i,j] + M[j,i] (diagonal NaN)."""
    C = M + M.T
    np.fill_diagonal(C, np.nan)
    return C


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
    degree_correct: bool = True,
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
        degree_correct: subtract each condition's overall engagement strength (row+column
            "degree") from the coupling matrix before gating (:func:`_degree_center`).
            **Default True** — this is the validated fix for the gate over-call (hub
            conditions otherwise look coupled to everything). Symmetric, so direction is
            unaffected. Set False for the raw ``M[a,b]+M[b,a]`` coupling.

    Returns:
        DataFrame, one row per unordered pair, sorted by descending ``coupling``:
        ``condition_a, condition_b, coupling`` (degree-corrected by default), ``coupling_raw``
        (uncorrected ``m_ab+m_ba``), ``cross_asym`` (the difference-of-medians
        approximation — see the module docstring; for direction use
        :func:`cascadir.cross_asym.direction_table`), ``coupling_null_p`` (cell-level,
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

    # symmetric coupling matrix; degree-correct (remove hub/degree bias) by default.
    # cross_asym (direction) is untouched — degree centering is symmetric. Degree
    # centering is degenerate for < 3 conditions (residual collapses to 0), so it is a
    # no-op there.
    do_degree = degree_correct and len(conditions) >= 3
    C_raw = _coupling_matrix(M)
    C_used = _degree_center(C_raw) if do_degree else C_raw
    for r in rows:
        i, j = idx_of[r["condition_a"]], idx_of[r["condition_b"]]
        r["coupling_raw"] = float(C_raw[i, j])
        r["coupling"] = float(C_used[i, j])  # corrected value drives the gate + sort

    # cell-level gene-set null (exploratory; over-powered — see module docstring).
    # When degree-correcting, the null matrices are degree-centered the SAME way
    # (apples-to-apples), so the p-value tests the pair-SPECIFIC residual.
    null_used = None
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
            if do_degree:
                null_used = np.full_like(null_cmat, np.nan)
                for k in range(null_cmat.shape[0]):
                    Ck = null_cmat[k].copy()
                    np.fill_diagonal(Ck, np.nan)
                    null_used[k] = _degree_center(Ck)
            else:
                null_used = null_cmat
        except ValueError:
            null_used = None

    # donor-level coupling (recommended gate); degree-corrected per donor when enabled.
    donor_C: list[np.ndarray] = []
    if cells_by_pair_per_donor:
        for _d, cbp_d in cells_by_pair_per_donor.items():
            conds_d, M_d = cross_engagement_matrix(
                cbp_d, sig_idx, control_label=control_label, min_cells=cfg.min_cells
            )
            full = np.full((len(conditions), len(conditions)), np.nan)
            local = {c: k for k, c in enumerate(conds_d)}
            for a in conds_d:
                for b in conds_d:
                    full[idx_of[a], idx_of[b]] = M_d[local[a], local[b]]
            C_d = _coupling_matrix(full)
            donor_C.append(_degree_center(C_d) if do_degree else C_d)

    for r in rows:
        i, j = idx_of[r["condition_a"]], idx_of[r["condition_b"]]
        if null_used is not None:
            nc = null_used[:, i, j]
            nc = nc[np.isfinite(nc)]
            r["coupling_null_p"] = (
                float(np.mean(nc >= r["coupling"])) if nc.size else float("nan")
            )
        else:
            r["coupling_null_p"] = float("nan")
        if donor_C:
            cpl = np.array([dC[i, j] for dC in donor_C], dtype=np.float64)
            cpl = cpl[np.isfinite(cpl)]
            nd = cpl.size
            r["n_donors"] = int(nd)
            if nd:
                r["donor_coupling_mean"] = float(np.mean(cpl))
                n_pos = int(np.sum(cpl > 0))
                r["donor_consensus"] = float(n_pos / nd)
                r["donor_sign_p"] = float(
                    sum(comb(nd, k) for k in range(n_pos, nd + 1)) / (2 ** nd)
                )
            else:
                r["donor_coupling_mean"] = float("nan")
                r["donor_consensus"] = float("nan")
                r["donor_sign_p"] = float("nan")

    df = pd.DataFrame(rows)
    if donor_C:
        df["coupled"] = (df["donor_sign_p"] <= coupling_alpha) & (df["donor_coupling_mean"] > 0)
    else:
        df["coupled"] = df["coupling_null_p"] < coupling_alpha
    return df.sort_values("coupling", ascending=False).reset_index(drop=True)
