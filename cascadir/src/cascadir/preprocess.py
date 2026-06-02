"""Preprocessing: normalization + HVG selection, with explicit handling of
whether your data is already normalized or not.

The method expects **log-normalized, HVG-subset** expression. This module gets
you there from either raw UMI counts or already-log-normalized values, and is
loud about the one case it cannot guess (so you never silently feed it the
wrong thing).

Decision guide (this is the answer to "what do I do if my data is/ isn't
normalized?"):

* **Raw UMI counts** (integer matrix): call ``preprocess(adata)`` — it stashes the
  counts, selects HVGs (seurat_v3, on counts), then ``normalize_total`` + ``log1p``.
  Equivalent: ``preprocess(adata, assume="raw")``.
* **Already log-normalized** (``normalize_total`` + ``log1p`` done): call
  ``preprocess(adata, assume="lognorm", flavor="seurat")`` — it skips normalization
  and selects HVGs on the log values. (seurat_v3 needs raw counts; if you also kept
  a ``counts`` layer, the default seurat_v3 will use it automatically.)
* **Anything else** (negative/z-scored values, normalized-but-not-logged): the
  ``"auto"`` detector raises :class:`NotPreprocessedError` and tells you which
  ``assume=`` to pass.
"""

from __future__ import annotations

import logging

import numpy as np
import scanpy as sc
from anndata import AnnData

from cascadir.exceptions import NotPreprocessedError

logger = logging.getLogger("cascadir")

# Log-normalized values are bounded by ~log1p(target_sum); 15 is a safe ceiling
# that still separates them from raw counts / CPM.
_LOGNORM_MAX = 15.0
_INT_ATOL = 1e-3


def _dense_sample(adata: AnnData, max_cells: int = 2000) -> np.ndarray:
    """Return a dense float sample of X (up to ``max_cells`` rows) for state checks."""
    n = adata.n_obs
    if n > max_cells:
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(n, size=max_cells, replace=False))
        X = adata[rows].X
    else:
        X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def is_raw_counts(adata: AnnData) -> bool:
    """True if ``adata.X`` looks like raw UMI counts (non-negative, integer-valued).

    Heuristic: a sample of X is all >= 0 and every entry is (within 1e-3 of) an
    integer. Log-normalized data has fractional values, so this is a clean split.
    """
    s = _dense_sample(adata)
    if s.size == 0:
        return False
    if s.min() < 0:
        return False
    return bool(np.allclose(s, np.round(s), atol=_INT_ATOL))


def is_lognormalized(adata: AnnData) -> bool:
    """True if ``adata.X`` looks like ``normalize_total`` + ``log1p`` output.

    Heuristic: a sample of X is non-negative, has fractional (non-integer) values,
    and its max is <= ~15 (log-space ceiling). Also returns True if
    ``adata.uns['log1p']`` is present (scanpy's log1p marker), provided X is
    non-negative.
    """
    s = _dense_sample(adata)
    if s.size == 0 or s.min() < 0:
        return False
    has_fractional = not np.allclose(s, np.round(s), atol=_INT_ATOL)
    if "log1p" in adata.uns and has_fractional and s.max() <= _LOGNORM_MAX:
        return True
    return bool(has_fractional and s.max() <= _LOGNORM_MAX)


def _resolve_state(adata: AnnData, assume: str) -> str:
    """Resolve the X state to 'raw_counts' or 'lognormalized', or raise."""
    if assume == "raw":
        return "raw_counts"
    if assume == "lognorm":
        return "lognormalized"
    if assume != "auto":
        raise ValueError(
            f"assume must be one of 'auto', 'raw', 'lognorm'; got {assume!r}."
        )
    if is_raw_counts(adata):
        return "raw_counts"
    if is_lognormalized(adata):
        return "lognormalized"
    s = _dense_sample(adata)
    raise NotPreprocessedError(
        "Could not auto-detect the normalization state of adata.X "
        f"(min={s.min():.3g}, max={s.max():.3g}). cascadir expects either raw UMI "
        "counts or normalize_total+log1p values. If these are raw counts, pass "
        "assume='raw'; if they are already normalize_total+log1p'd, pass "
        "assume='lognorm'. (Negative or very large non-integer values suggest the "
        "data was z-scored or normalized-but-not-logged, which is not supported — "
        "re-derive from raw counts or log1p the normalized values first.)"
    )


def normalize_log1p(adata: AnnData, *, target_sum: float = 1e4) -> AnnData:
    """Apply ``normalize_total(target_sum)`` then ``log1p`` in place; returns adata.

    Use this only on raw counts. On already-log-normalized data it would double-log;
    prefer :func:`preprocess`, which guards against that.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_hvgs(
    adata: AnnData,
    *,
    n_hvgs: int = 4000,
    flavor: str = "seurat_v3",
    batch_key: str | None = None,
    counts_layer: str | None = None,
) -> AnnData:
    """Mark the top ``n_hvgs`` highly-variable genes in ``adata.var['highly_variable']``.

    Does not subset (so callers can inspect the flags); :func:`preprocess` subsets.
    If the panel already has <= ``n_hvgs`` genes (e.g. a targeted panel), all genes
    are marked highly-variable and selection is a no-op.

    ``flavor='seurat_v3'`` is computed on raw counts (``counts_layer``) and needs the
    optional ``scikit-misc`` package. ``flavor='seurat'`` is computed on
    log-normalized values and needs no extra dependency.
    """
    if adata.n_vars <= n_hvgs:
        logger.info(
            "select_hvgs: panel has %d <= n_hvgs=%d genes; keeping all genes.",
            adata.n_vars,
            n_hvgs,
        )
        adata.var["highly_variable"] = True
        return adata

    kwargs: dict = {"n_top_genes": n_hvgs, "flavor": flavor}
    if counts_layer is not None:
        kwargs["layer"] = counts_layer
    try:
        if batch_key is not None:
            try:
                sc.pp.highly_variable_genes(adata, batch_key=batch_key, **kwargs)
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "select_hvgs: batch_key=%r failed (%s); retrying pooled.",
                    batch_key,
                    exc,
                )
                sc.pp.highly_variable_genes(adata, **kwargs)
        else:
            sc.pp.highly_variable_genes(adata, **kwargs)
    except ImportError as exc:  # seurat_v3 -> skmisc.loess missing
        raise NotPreprocessedError(
            "flavor='seurat_v3' requires the 'scikit-misc' package "
            f"({exc}). Either install it (pip install scikit-misc) or pass "
            "flavor='seurat' to select HVGs on the log-normalized values instead."
        ) from exc
    return adata


def preprocess(
    adata: AnnData,
    *,
    n_hvgs: int = 4000,
    target_sum: float = 1e4,
    flavor: str = "seurat_v3",
    batch_key: str | None = None,
    assume: str = "auto",
    copy: bool = True,
) -> AnnData:
    """Bring an AnnData to the method-ready state: log-normalized + HVG-subset.

    Args:
        adata: cells x genes AnnData. ``X`` may be raw counts or log-normalized.
        n_hvgs: Number of highly-variable genes to keep.
        target_sum: ``normalize_total`` target (only used for raw-count input).
        flavor: HVG flavor. ``'seurat_v3'`` (default, on raw counts) or ``'seurat'``
            (on log values). See :func:`select_hvgs`.
        batch_key: Optional ``obs`` column for within-batch HVG ranking (e.g. donor).
        assume: ``'auto'`` (detect), ``'raw'`` (force raw-count branch), or
            ``'lognorm'`` (force log-normalized branch).
        copy: Work on a copy (default) and return it; if False, mutate ``adata``.

    Returns:
        A new AnnData (or the mutated input) with log-normalized ``X`` subset to the
        selected HVGs. Re-running on already-processed data is a near no-op (the
        normalize/log1p step is skipped for log-normalized input).

    Raises:
        NotPreprocessedError: if ``assume='auto'`` cannot decide the state, or if
            log-normalized input is given with ``flavor='seurat_v3'`` but no
            ``counts`` layer (seurat_v3 needs raw counts).
    """
    state = _resolve_state(adata, assume)
    work = adata.copy() if copy else adata

    if state == "raw_counts":
        if "counts" not in work.layers:
            work.layers["counts"] = work.X.copy()
        counts_layer = "counts" if flavor == "seurat_v3" else None
        select_hvgs(
            work,
            n_hvgs=n_hvgs,
            flavor=flavor,
            batch_key=batch_key,
            counts_layer=counts_layer,
        )
        normalize_log1p(work, target_sum=target_sum)
    else:  # lognormalized
        if flavor == "seurat_v3" and "counts" not in work.layers:
            raise NotPreprocessedError(
                "Input looks log-normalized but flavor='seurat_v3' needs raw counts, "
                "which are not present (no 'counts' layer). Either (1) provide raw "
                "counts and call preprocess(assume='raw'), or (2) pass flavor='seurat' "
                "to select HVGs on the log-normalized values (a valid HVG method)."
            )
        counts_layer = "counts" if flavor == "seurat_v3" else None
        select_hvgs(
            work,
            n_hvgs=n_hvgs,
            flavor=flavor,
            batch_key=batch_key,
            counts_layer=counts_layer,
        )

    if "highly_variable" in work.var:
        work = work[:, work.var["highly_variable"].to_numpy()].copy()
    logger.info(
        "preprocess: state=%s -> %d cells x %d HVGs.",
        state,
        work.n_obs,
        work.n_vars,
    )
    return work
