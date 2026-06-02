"""Data-suitability validation.

``validate_anndata`` is the gate every entry point runs first. It checks that the
AnnData carries everything the cross_asym method needs — the right ``obs`` columns,
a control condition, enough donors and cells, a finite non-negative matrix — and
either returns a :class:`ValidationReport` or (in strict mode, the default) raises
:class:`DataValidationError` listing *all* problems at once.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from cascadir.exceptions import DataValidationError
from cascadir.preprocess import is_lognormalized, is_raw_counts
from cascadir.types import ValidationReport


def _x_finite_nonneg(adata: AnnData) -> tuple[bool, bool, float, float]:
    """Return (all_finite, all_nonneg, min, max) over X (sparse-aware, exact)."""
    X = adata.X
    data = X.data if hasattr(X, "data") else np.asarray(X)
    data = np.asarray(data)
    if data.size == 0:
        return True, True, 0.0, 0.0
    finite = bool(np.isfinite(data).all())
    # For sparse, implicit zeros are non-negative; only stored data can be negative.
    nonneg = bool(np.nanmin(data) >= 0) if finite else False
    dmin = float(np.nanmin(data)) if finite else float("nan")
    dmax = float(np.nanmax(data)) if finite else float("nan")
    if hasattr(X, "data"):  # sparse: account for implicit zeros in min
        dmin = min(dmin, 0.0)
    return finite, nonneg, dmin, dmax


def _x_state(adata: AnnData) -> str:
    if is_raw_counts(adata):
        return "raw_counts"
    if is_lognormalized(adata):
        return "lognormalized"
    return "ambiguous"


def validate_anndata(
    adata: AnnData,
    *,
    condition_col: str,
    donor_col: str,
    celltype_col: str,
    control_label: str = "PBS",
    min_donors: int = 3,
    min_cells: int = 10,
    n_hvgs: int | None = None,
    strict: bool = True,
) -> ValidationReport:
    """Validate that ``adata`` is suitable for the cross_asym method.

    Args:
        adata: cells x genes AnnData.
        condition_col: ``obs`` column with the stimulus label per cell.
        donor_col: ``obs`` column with the biological-replicate id per cell.
        celltype_col: ``obs`` column with the cell-type label per cell.
        control_label: The resting/unstimulated label that must be present
            (it is the PBS baseline for normalization).
        min_donors: Minimum number of distinct donors required (default 3).
        min_cells: Minimum cells per (condition, cell_type) the method can use.
        n_hvgs: If given, warns when ``n_genes < n_hvgs`` (HVG selection no-op).
        strict: If True (default), raise :class:`DataValidationError` when any
            blocking issue is found. If False, return the report regardless.

    Returns:
        A :class:`ValidationReport`. ``report.ok`` is True iff there are no blocking
        issues.

    Raises:
        DataValidationError: in strict mode, if any blocking issue is found. The
            message lists every issue (and any warnings) so they can be fixed in one
            pass.
    """
    issues: list[str] = []
    warnings: list[str] = []
    obs = adata.obs

    # --- obs columns present -------------------------------------------------
    missing_cols = [c for c in (condition_col, donor_col, celltype_col) if c not in obs]
    for c in missing_cols:
        issues.append(
            f"obs is missing required column {c!r}. Present columns: {list(obs.columns)}."
        )

    # --- gene names unique ---------------------------------------------------
    var_names = list(map(str, adata.var_names))
    if len(set(var_names)) != len(var_names):
        warnings.append(
            "var_names are not unique; call adata.var_names_make_unique() before "
            "preprocessing (duplicate gene symbols break signature lookup)."
        )

    # --- X finiteness / sign -------------------------------------------------
    finite, nonneg, dmin, dmax = _x_finite_nonneg(adata)
    if not finite:
        issues.append("adata.X contains NaN or infinite values; clean it before use.")
    if finite and not nonneg:
        issues.append(
            f"adata.X has negative values (min={dmin:.3g}); cascadir expects raw "
            "counts or log-normalized values (both non-negative), not z-scored data."
        )
    x_state = _x_state(adata) if finite else "ambiguous"
    if finite and x_state == "ambiguous":
        warnings.append(
            f"adata.X normalization state is ambiguous (min={dmin:.3g}, max={dmax:.3g}); "
            "pass assume='raw' or assume='lognorm' to preprocess()."
        )

    # --- condition / control / donors / cell types ---------------------------
    n_donors = 0
    n_cell_types = 0
    n_conditions = 0
    control_present = False
    per_condition_cells: dict[str, int] = {}

    if condition_col in obs:
        cond_series = obs[condition_col].astype(str)
        per_condition_cells = {
            str(k): int(v) for k, v in cond_series.value_counts().items()
        }
        conditions = sorted(per_condition_cells)
        n_conditions = len(conditions)
        control_present = control_label in per_condition_cells
        if not control_present:
            issues.append(
                f"control_label {control_label!r} is not in obs[{condition_col!r}]. "
                f"Found conditions: {conditions}. The control is required as the PBS "
                "baseline — relabel your unstimulated/resting cells to this string."
            )
        stim = [c for c in conditions if c != control_label]
        if len(stim) < 1:
            issues.append(
                "Need at least one stimulus condition besides the control; found none."
            )

    if donor_col in obs:
        donors = sorted(obs[donor_col].astype(str).unique())
        n_donors = len(donors)
        if n_donors < min_donors:
            issues.append(
                f"Only {n_donors} distinct donor(s) in obs[{donor_col!r}] "
                f"({donors}); the method needs >= {min_donors} for a meaningful "
                "per-cell-type aggregation. (If you truly have fewer biological "
                "replicates, pool (context x replicate) as pseudo-donors.)"
            )

    # --- per-(condition, cell_type) cell sufficiency -------------------------
    if condition_col in obs and celltype_col in obs:
        ct_series = obs[celltype_col].astype(str)
        n_cell_types = int(ct_series.nunique())
        if n_cell_types < 1:
            issues.append(f"obs[{celltype_col!r}] has no cell-type labels.")
        # Does the control have at least one cell type with >= min_cells?
        if control_present:
            grp = (
                obs.assign(_c=obs[condition_col].astype(str), _t=ct_series)
                .groupby(["_c", "_t"])
                .size()
            )
            ctrl_ok = (grp.loc[control_label] >= min_cells).any() if (
                control_label in grp.index.get_level_values(0)
            ) else False
            if not ctrl_ok:
                issues.append(
                    f"The control {control_label!r} has no (cell_type) group with "
                    f">= {min_cells} cells; cross_asym needs a populated control "
                    "baseline per cell type."
                )
            stim_labels = [
                c for c in per_condition_cells if c != control_label
            ]
            stim_with_enough = [
                c
                for c in stim_labels
                if c in grp.index.get_level_values(0)
                and (grp.loc[c] >= min_cells).any()
            ]
            if not stim_with_enough:
                issues.append(
                    f"No stimulus condition has a (cell_type) group with >= {min_cells} "
                    "cells; nothing can be scored."
                )
            elif len(stim_with_enough) < len(stim_labels):
                thin = sorted(set(stim_labels) - set(stim_with_enough))
                warnings.append(
                    f"These conditions have no cell type reaching {min_cells} cells and "
                    f"will be skipped: {thin}."
                )

    # --- gene count vs HVG target -------------------------------------------
    if n_hvgs is not None and adata.n_vars < n_hvgs:
        warnings.append(
            f"n_genes={adata.n_vars} < n_hvgs={n_hvgs}; HVG selection will keep all "
            "genes (this is fine for targeted panels)."
        )

    report = ValidationReport(
        ok=len(issues) == 0,
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_donors=n_donors,
        n_cell_types=n_cell_types,
        n_conditions=n_conditions,
        control_label=control_label,
        control_present=control_present,
        x_state=x_state,
        issues=tuple(issues),
        warnings=tuple(warnings),
        per_condition_cells=per_condition_cells,
    )

    if strict and not report.ok:
        msg = "AnnData failed cascadir validation:\n" + "\n".join(
            f"  - {m}" for m in issues
        )
        if warnings:
            msg += "\n  (warnings:)\n" + "\n".join(f"    - {m}" for m in warnings)
        raise DataValidationError(msg)
    return report
