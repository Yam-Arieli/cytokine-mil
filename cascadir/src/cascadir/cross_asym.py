"""The cross-engagement asymmetry statistic and the direction call.

This is the scientific heart. Given two conditions ``a`` and ``b`` with discovered
signatures ``S_a`` and ``S_b``, and cells grouped per ``(condition, cell_type)``:

    cross_asym(a, b) = s(a, S_b) - s(b, S_a)      # PBS-normalized

where ``s(x, S)`` is the mean expression of gene set ``S`` in condition ``x``'s cells.
This quantity is **antisymmetric**: ``cross_asym(b, a) == -cross_asym(a, b)``, so its
*sign* encodes direction. Biology: an upstream stimulus's cells carry both their own
program *and* the autocrine downstream one, while the downstream ligand's cells carry
mainly their own — so ``s(upstream, S_down) > s(down, S_up)``.

We aggregate per cell type (median + sign-consensus), classify the call
(STRONG / WEAK / AMBIGUOUS), and compare against a random-gene-set null to confirm
the *discovered* signatures carry condition-specific information.

Honest caveat (do not skip): cross_asym answers **direction, not existence**. A pair
that is not biologically coupled can still have a large ``|cross_asym|`` — deciding
*whether* a pair is coupled is a separate problem. Magnitude is not a coupling gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cascadir.config import CrossAsymConfig
from cascadir.exceptions import SignatureError
from cascadir.types import DirectionCall, Signature


# ---------------------------------------------------------------------------
# Per-cell-type directional test
# ---------------------------------------------------------------------------


def directional_asymmetry_test(
    cells_by_pair: dict[tuple[str, str], np.ndarray],
    sig_idx: dict[str, np.ndarray],
    a: str,
    b: str,
    *,
    control_label: str = "PBS",
    min_cells: int = 10,
) -> pd.DataFrame:
    """Per-cell-type cross-engagement table for the ordered pair (a, b).

    Args:
        cells_by_pair: ``{(condition, cell_type): (n_cells, n_genes)}`` expression.
        sig_idx: ``{condition: gene-index array}`` — the signature columns for ``a``
            and ``b`` (indices into the gene axis of the arrays above).
        a, b: The two conditions (``a`` is treated as the candidate upstream).
        control_label: The PBS baseline condition.
        min_cells: Minimum cells per group for a cell type to be scored.

    Returns:
        A DataFrame with one row per usable cell type, columns include
        ``sA_PB_norm`` = s(a, S_b)-PBS, ``sB_PA_norm`` = s(b, S_a)-PBS,
        ``cross_asym`` = ``sA_PB_norm - sB_PA_norm`` (the direction-bearing quantity),
        and ``directional_score`` (the symmetric reference). Empty if no cell type
        has all three of (a, b, control) populated above ``min_cells``.
    """
    idx_a = sig_idx[a]
    idx_b = sig_idx[b]
    cell_types = sorted({ct for (_, ct) in cells_by_pair})
    rows: list[dict] = []
    for T in cell_types:
        keys = [(a, T), (b, T), (control_label, T)]
        if any(k not in cells_by_pair for k in keys):
            continue
        cA = cells_by_pair[(a, T)]
        cB = cells_by_pair[(b, T)]
        cP = cells_by_pair[(control_label, T)]
        if len(cA) < min_cells or len(cB) < min_cells or len(cP) < min_cells:
            continue
        sA_in_PA = float(cA[:, idx_a].mean())
        sA_in_PB = float(cA[:, idx_b].mean())
        sB_in_PA = float(cB[:, idx_a].mean())
        sB_in_PB = float(cB[:, idx_b].mean())
        sP_in_PA = float(cP[:, idx_a].mean())
        sP_in_PB = float(cP[:, idx_b].mean())
        sA_PA_norm = sA_in_PA - sP_in_PA
        sA_PB_norm = sA_in_PB - sP_in_PB
        sB_PA_norm = sB_in_PA - sP_in_PA
        sB_PB_norm = sB_in_PB - sP_in_PB
        asym_PA = sA_PA_norm - sB_PA_norm
        asym_PB = sA_PB_norm - sB_PB_norm
        rows.append(
            {
                "cell_type": T,
                "sA_PA_norm": sA_PA_norm,
                "sA_PB_norm": sA_PB_norm,
                "sB_PA_norm": sB_PA_norm,
                "sB_PB_norm": sB_PB_norm,
                "asym_PA": asym_PA,
                "asym_PB": asym_PB,
                "directional_score": asym_PA - asym_PB,
                "cross_asym": sA_PB_norm - sB_PA_norm,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation + classification
# ---------------------------------------------------------------------------


def aggregate_direction(
    df: pd.DataFrame, *, column: str = "cross_asym"
) -> tuple[float, float, int, int]:
    """Median + sign-consensus over a per-cell-type metric.

    Returns ``(median, consensus, n_pos, n_neg)``. ``median`` is robust to a single
    dominant cell type; ``consensus`` is the fraction of cell types whose sign matches
    the median's sign (for ``median == 0`` it is the fraction of exact zeros).
    """
    if df.empty:
        return float("nan"), float("nan"), 0, 0
    scores = df[column].to_numpy(dtype=np.float64)
    med = float(np.median(scores))
    n_pos = int(np.sum(scores > 0))
    n_neg = int(np.sum(scores < 0))
    n_total = len(scores)
    if med == 0.0:
        consensus = float(np.sum(scores == 0) / n_total)
    elif med > 0:
        consensus = float(n_pos / n_total)
    else:
        consensus = float(n_neg / n_total)
    return med, consensus, n_pos, n_neg


def classify_call(median: float, consensus: float, config: CrossAsymConfig) -> str:
    """STRONG / WEAK / AMBIGUOUS from (|median|, consensus)."""
    if np.isnan(median) or np.isnan(consensus):
        return "AMBIGUOUS"
    if abs(median) < config.magnitude_threshold:
        return "AMBIGUOUS"
    if consensus >= config.strong_consensus:
        return "STRONG"
    if consensus >= config.weak_consensus:
        return "WEAK"
    return "AMBIGUOUS"


# ---------------------------------------------------------------------------
# Random-gene-set null
# ---------------------------------------------------------------------------


def random_gene_set_null(
    cells_by_pair: dict[tuple[str, str], np.ndarray],
    a: str,
    b: str,
    *,
    size_a: int,
    size_b: int,
    n_genes: int,
    excluded_indices: set[int],
    n_perms: int = 100,
    min_cells: int = 10,
    control_label: str = "PBS",
    seed: int = 42,
) -> np.ndarray:
    """Median cross_asym under random signatures of matched size.

    For each of ``n_perms`` draws, replace ``S_a`` and ``S_b`` with random gene sets
    of the same sizes, drawn from genes **disjoint from every observed signature**
    (``excluded_indices``). Tests whether the discovered signatures carry
    condition-specific direction information vs. generic activation-responsive genes.

    Note (reproducibility): the RNG is seeded *per call* (``seed``), so each pair's
    null is independent and reproducible in isolation. This is intentional, but it
    means the resulting ``null_p`` values are not numerically identical to the
    original research runs, which advanced a single shared RNG sequentially across
    all pairs. The direction sign and the STRONG/WEAK/AMBIGUOUS classification do not
    depend on the null and are unaffected.

    Returns:
        ``(n_perms,)`` array of per-permutation median cross_asym (NaNs where a draw
        produced no scorable cell type).
    """
    pool = np.array(
        [i for i in range(n_genes) if i not in excluded_indices], dtype=np.int64
    )
    if len(pool) < max(size_a, size_b):
        raise SignatureError(
            f"Null pool too small: {len(pool)} non-signature genes available, "
            f"need {max(size_a, size_b)}. Reduce top_n or supply more genes."
        )
    rng = np.random.default_rng(seed)
    meds = np.full(n_perms, np.nan, dtype=np.float64)
    for k in range(n_perms):
        idx_a = rng.choice(pool, size=size_a, replace=False)
        idx_b = rng.choice(pool, size=size_b, replace=False)
        df = directional_asymmetry_test(
            cells_by_pair,
            {a: idx_a, b: idx_b},
            a,
            b,
            control_label=control_label,
            min_cells=min_cells,
        )
        meds[k] = aggregate_direction(df)[0]
    return meds


# ---------------------------------------------------------------------------
# High-level direction call
# ---------------------------------------------------------------------------


def _signatures_to_idx(
    signatures: dict[str, Signature], gene_names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    gene_index = {g: i for i, g in enumerate(gene_names)}
    out: dict[str, np.ndarray] = {}
    for cond, sig in signatures.items():
        idx = np.array(
            [gene_index[g] for g in sig.genes if g in gene_index], dtype=np.int64
        )
        out[cond] = idx
    return out


def direction_call(
    cells_by_pair: dict[tuple[str, str], np.ndarray],
    signatures: dict[str, Signature],
    gene_names: tuple[str, ...],
    a: str,
    b: str,
    *,
    control_label: str = "PBS",
    config: CrossAsymConfig | None = None,
) -> DirectionCall:
    """Compute the full cascade-direction call for one pair (a, b).

    The pair is canonicalized alphabetically so the sign convention is fixed:
    a positive median cross_asym means ``condition_a`` (the alphabetically-first one)
    is upstream.

    Args:
        cells_by_pair: ``{(condition, cell_type): (n_cells, n_genes)}`` (must include
            both conditions and the control).
        signatures: ``{condition: Signature}`` — must contain both ``a`` and ``b``.
            All signatures present are used to define the null's excluded gene pool.
        gene_names: Gene order of the arrays in ``cells_by_pair``.
        a, b: The pair of conditions.
        control_label: PBS baseline condition.
        config: :class:`CrossAsymConfig` (defaults if ``None``).

    Returns:
        A :class:`DirectionCall`.

    Raises:
        SignatureError: if a signature is missing/empty, or ``S_a == S_b`` exactly.
    """
    cfg = config or CrossAsymConfig()
    lo, hi = sorted([a, b])
    for cond in (lo, hi):
        if cond not in signatures:
            raise SignatureError(
                f"No signature for {cond!r}; derive it before calling direction_call."
            )

    sig_idx = _signatures_to_idx(signatures, gene_names)
    if sig_idx[lo].size == 0 or sig_idx[hi].size == 0:
        raise SignatureError(
            f"Signature for {lo!r} or {hi!r} has no genes present in gene_names."
        )
    if set(sig_idx[lo].tolist()) == set(sig_idx[hi].tolist()):
        raise SignatureError(
            f"Signatures for {lo!r} and {hi!r} are identical; cross_asym collapses to "
            "0 (S_a == S_b). The pair is not separable by discovered signature."
        )

    df = directional_asymmetry_test(
        cells_by_pair, sig_idx, lo, hi, control_label=control_label,
        min_cells=cfg.min_cells,
    )
    median, consensus, n_pos, n_neg = aggregate_direction(df)
    ds_median, _ds_cons, _dp, _dn = aggregate_direction(df, column="directional_score")
    classification = classify_call(median, consensus, cfg)

    if classification in ("STRONG", "WEAK") and not np.isnan(median) and median != 0:
        if median > 0:
            direction, upstream = "a_to_b", lo
        else:
            direction, upstream = "b_to_a", hi
    else:
        direction, upstream = "ambiguous", None

    null_p: float | None = None
    if cfg.n_null_perms > 0 and not np.isnan(median):
        excluded = {i for idx in sig_idx.values() for i in idx.tolist()}
        null_meds = random_gene_set_null(
            cells_by_pair,
            lo,
            hi,
            size_a=int(sig_idx[lo].size),
            size_b=int(sig_idx[hi].size),
            n_genes=len(gene_names),
            excluded_indices=excluded,
            n_perms=cfg.n_null_perms,
            min_cells=cfg.min_cells,
            control_label=control_label,
            seed=cfg.null_seed,
        )
        valid = null_meds[~np.isnan(null_meds)]
        if valid.size > 0:
            null_p = float(np.mean(np.abs(valid) >= abs(median)))

    return DirectionCall(
        condition_a=lo,
        condition_b=hi,
        cross_asym_median=median,
        directional_score_median=ds_median,
        sign_consensus=consensus,
        n_pos=n_pos,
        n_neg=n_neg,
        classification=classification,
        direction=direction,
        upstream=upstream,
        null_p=null_p,
        per_cell_type=df,
    )


def direction_table(
    cells_by_pair: dict[tuple[str, str], np.ndarray],
    signatures: dict[str, Signature],
    gene_names: tuple[str, ...],
    pairs: list[tuple[str, str]],
    *,
    control_label: str = "PBS",
    config: CrossAsymConfig | None = None,
) -> pd.DataFrame:
    """Run :func:`direction_call` over many pairs and return a tidy summary table.

    Columns: condition_a, condition_b, cross_asym_median, sign_consensus, n_pos,
    n_neg, classification, direction, upstream, null_p. Sorted by descending
    ``|cross_asym_median|``.
    """
    records: list[dict] = []
    for a, b in pairs:
        call = direction_call(
            cells_by_pair, signatures, gene_names, a, b,
            control_label=control_label, config=config,
        )
        records.append(
            {
                "condition_a": call.condition_a,
                "condition_b": call.condition_b,
                "cross_asym_median": call.cross_asym_median,
                "directional_score_median": call.directional_score_median,
                "sign_consensus": call.sign_consensus,
                "n_pos": call.n_pos,
                "n_neg": call.n_neg,
                "classification": call.classification,
                "direction": call.direction,
                "upstream": call.upstream,
                "null_p": call.null_p,
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.reindex(
            df["cross_asym_median"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    return df
