#!/usr/bin/env python
"""Prepare the SARS-CoV-2 vaccine CITE atlas for the §32 T-cell cascade run.

Loads `vaccine_cite_raw.h5ad` (from assemble_vaccine_h5ad.py), AUTO-DETECTS the
donor / timepoint / cell-type obs fields, **subsets to T cells**, assigns a T-cell
maturation `state ∈ {Naive, Effector, Memory}` (preferring CITE **surface protein**
gating — independent of the RNA we score on, so the state label can't leak into the
signature — else the fine author annotation), standardizes obs to
`state` / `timepoint` / `subject` / `tcell_lineage`, relabels **day-0 cells to the
control `"Resting"`**, and writes a slim prepared h5ad + `obs_summary.json`.

One prepared h5ad serves BOTH cascadir runs:
  * STATE     framing: condition=`state`,     control=`Resting`
  * TIMEPOINT framing: condition=`timepoint`, control=`D0`

`--inspect_only` reads obs (no heavy write) and reports detected fields +
value_counts + the protein panel + which gating markers matched — run this first on
the cluster to lock the field mapping and the state_source before the GPU fit.

Usage:
  python scripts/prepare_vaccine_tcell.py --inspect_only
  python scripts/prepare_vaccine_tcell.py \
      --raw /cs/.../SARSCoV2_Vaccine/raw/vaccine_cite_raw.h5ad \
      --out /cs/.../SARSCoV2_Vaccine/prepared/vaccine_tcell_prepared.h5ad
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

RAW_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/raw/vaccine_cite_raw.h5ad"
OUT_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/prepared/vaccine_tcell_prepared.h5ad"

CONTROL = "Resting"                       # day-0 baseline (the "PBS"/"Healthy" analog)
STATES = ["Naive", "Effector", "Memory"]  # ordered: Naive -> Effector -> Memory
TIMEPOINTS = ["D0", "D2", "D11", "D28"]   # ordered; D0 is the control for the timepoint run
                                          # (this atlas samples Day0/2/11/28, not Day10)

DONOR_CANDIDATES = [
    "subject_id", "subject", "donor_id", "donor", "patient_id", "patient",
    "sample_id", "Donor", "Subject", "orig.ident",
]
TIMEPOINT_CANDIDATES = [
    "timepoint", "time_point", "Timepoint", "time", "day", "Day", "visit",
    "days_post_vaccination", "vaccine_timepoint", "collection_day",
]
# coarse lineage (for T-cell subsetting): prefer Azimuth l1-style
LINEAGE_CANDIDATES = [
    "celltypel1", "predicted.celltype.l1", "celltype.l1", "initial_clustering",
    "WCTcoursecelltype", "cell_type_coarse", "coarse_celltype", "majority_voting",
    "CellType", "cell_type", "celltype",
]
# fine annotation (for annotation-based state): prefer Azimuth l2-style
FINE_CANDIDATES = [
    "celltypel2", "celltypel3", "predicted.celltype.l2", "celltype.l2", "fine_annotation",
    "cell_type_fine", "annotation_fine", "full_clustering",
]

# ADT surface-protein names for the maturation gating (fuzzy matched). CCR7/CD197 is
# absent from some panels (incl. this one) — CD45RO + CD27 give an equivalent split.
PROT_MARKERS = {
    "CD45RA": ["CD45RA", "Hu.CD45RA", "PTPRC-RA"],
    "CD45RO": ["CD45RO", "Hu.CD45RO", "PTPRC-RO"],
    "CCR7":   ["CCR7", "CD197", "Hu.CD197", "Hu.CCR7"],
    "CD27":   ["CD27", "Hu.CD27"],
    "CD95":   ["CD95", "Hu.CD95", "Fas"],
}


# --------------------------- detection ---------------------------

def _detect_first(obs, candidates, what, required=True):
    for c in candidates:
        if c in obs.columns:
            return c
    if required:
        raise SystemExit(f"Could not detect a {what} column among {candidates}. "
                         f"Columns available: {list(obs.columns)}")
    return None


def _canon_timepoint(v):
    """Map a raw timepoint value (e.g. 'Day11') to D0/D2/D11/D28, or None to drop."""
    s = str(v).strip().lower()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    n = int(m.group(1))
    return {0: "D0", 2: "D2", 11: "D11", 28: "D28"}.get(n, None)


_NONT = re.compile(r"\b(nk|b cell|b_cell|plasma|mono|dc|platelet|hspc|prog|"
                   r"erythro|baso|mast|ilc|neutro|myeloid|megakaryo)", re.I)
_IST = re.compile(r"(cd4|cd8|treg|mait|gdt|gd t|dnt|\bt cell|\bt_cell|\bt\b|"
                  r"tcm|tem|temra|cytotoxic|ctl)", re.I)


def _is_tcell(label: str) -> bool:
    s = str(label)
    if _NONT.search(s):
        return False
    return bool(_IST.search(s))


def _lineage_label(label: str) -> str:
    s = str(label)
    if re.search(r"cd4|treg", s, re.I):
        return "CD4 T"
    if re.search(r"cd8", s, re.I):
        return "CD8 T"
    return "other T"


def _state_from_annotation(label: str):
    """Parse an Azimuth-l2-style fine label into a maturation state (or None)."""
    s = str(label).lower()
    if "naive" in s:
        return "Naive"
    if "temra" in s or "ctl" in s or "cytotoxic" in s or "effector" in s or \
       "proliferat" in s or "cycling" in s:
        return "Effector"
    if "tcm" in s or "tem" in s or "central memory" in s or "effector memory" in s or \
       "memory" in s:
        return "Memory"
    return None


# --------------------------- protein gating ---------------------------

def _match_proteins(protein_names):
    idx = {}
    lower = {p.lower(): i for i, p in enumerate(protein_names)}
    for canon, aliases in PROT_MARKERS.items():
        for a in aliases:
            if a.lower() in lower:
                idx[canon] = lower[a.lower()]
                break
    return idx


def _state_from_protein(adata, tcell_mask):
    """Surface-protein gating -> {Naive, Effector, Memory} on T cells (RNA-independent).

    Preferred (if CCR7 present): CD45RA/CCR7 quadrant
        Naive=RA+CCR7+, Effector=RA+CCR7- (TEMRA), Memory=RA-.
    Fallback (CCR7 absent, this panel): CD45RO + CD27
        Memory=RO+ (Tcm/Tem), Naive=RO- & CD27+ (CD45RA+CD27+), Effector=RO- & CD27-
        (terminal effector that has lost CD27).
    Thresholds = per-protein median over the T-cell subset (binary high/low).
    Returns (state_array_full_object, used_markers_dict, scheme_str) or (None, pidx, "..").
    """
    if "protein" not in adata.obsm or "protein_names" not in adata.uns:
        return None, {}, "none"
    pidx = _match_proteins(list(adata.uns["protein_names"]))
    P = np.asarray(adata.obsm["protein"])
    t = tcell_mask

    def hi(name):
        v = P[:, pidx[name]]
        return v > np.median(v[t])

    state = np.full(adata.n_obs, "NA", dtype=object)
    if "CD45RA" in pidx and "CCR7" in pidx:
        ra_hi, cc_hi = hi("CD45RA"), hi("CCR7")
        state[t & ra_hi & cc_hi] = "Naive"
        state[t & ra_hi & ~cc_hi] = "Effector"
        state[t & ~ra_hi] = "Memory"
        return state, pidx, "CD45RA/CCR7"
    if "CD45RO" in pidx and "CD27" in pidx:
        ro_hi, c27_hi = hi("CD45RO"), hi("CD27")
        state[t & ro_hi] = "Memory"
        state[t & ~ro_hi & c27_hi] = "Naive"
        state[t & ~ro_hi & ~c27_hi] = "Effector"
        return state, pidx, "CD45RO/CD27"
    return None, pidx, "insufficient"


# --------------------------- report ---------------------------

def _report(adata, donor_col, tp_col, lin_col, fine_col):
    obs = adata.obs
    print(f"[detect] donor    = {donor_col!r}")
    print(f"[detect] timepoint= {tp_col!r}")
    print(f"[detect] lineage  = {lin_col!r}")
    print(f"[detect] fine     = {fine_col!r}")
    print(f"\n[timepoint raw value_counts in {tp_col!r}]")
    print(obs[tp_col].value_counts().head(20).to_string())
    canon_tp = obs[tp_col].map(_canon_timepoint)
    print(f"\n[timepoint -> canonical] kept {int(canon_tp.notna().sum())}/{len(obs)}")
    print(canon_tp.value_counts().to_string())
    print(f"\n[lineage value_counts in {lin_col!r}] n={obs[lin_col].nunique()}")
    print(obs[lin_col].value_counts().head(40).to_string())
    tmask = obs[lin_col].map(_is_tcell).to_numpy()
    print(f"\n[T-cell subset] {int(tmask.sum())}/{len(obs)} cells flagged T")
    print(f"[donors] {obs[donor_col].nunique()} unique in {donor_col!r}")
    if fine_col is not None:
        print(f"\n[fine annotation {fine_col!r}] top values")
        print(obs[fine_col].value_counts().head(40).to_string())
        ann_state = obs.loc[tmask, fine_col].map(_state_from_annotation)
        print("[annotation -> state] (T cells):")
        print(ann_state.value_counts(dropna=False).to_string())
    # protein panel
    if "protein_names" in adata.uns:
        names = list(adata.uns["protein_names"])
        pidx = _match_proteins(names)
        print(f"\n[protein] {len(names)} surface proteins; gating markers matched: {pidx}")
        st, _, scheme = _state_from_protein(adata, tmask)
        if st is not None:
            print(f"[protein -> state] scheme={scheme} (T cells):")
            print(pd.Series(st[tmask]).value_counts().to_string())
        else:
            print(f"[protein] insufficient markers (scheme={scheme}) -> gating unavailable")
    else:
        print("\n[protein] no obsm['protein'] -> protein gating unavailable")
    if not adata.isbacked:
        print(f"\n[expr] X min={float(adata.X.min()):.3g} max={float(adata.X.max()):.3g}")


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--state_source", default="auto",
                    choices=["auto", "protein", "annotation"])
    ap.add_argument("--donor_field", default=None)
    ap.add_argument("--timepoint_field", default=None)
    ap.add_argument("--lineage_field", default=None)
    ap.add_argument("--fine_field", default=None)
    ap.add_argument("--inspect_only", action="store_true")
    args = ap.parse_args()

    print(f"[load] {args.raw}")
    adata = ad.read_h5ad(args.raw, backed="r" if args.inspect_only else None)
    obs = adata.obs
    print(f"[load] {adata.n_obs} cells × {adata.n_vars} genes; obs cols: {list(obs.columns)}")

    donor_col = args.donor_field or _detect_first(obs, DONOR_CANDIDATES, "donor")
    tp_col = args.timepoint_field or _detect_first(obs, TIMEPOINT_CANDIDATES, "timepoint")
    lin_col = args.lineage_field or _detect_first(obs, LINEAGE_CANDIDATES, "lineage")
    fine_col = args.fine_field or _detect_first(obs, FINE_CANDIDATES, "fine", required=False)

    _report(adata, donor_col, tp_col, lin_col, fine_col)

    if args.inspect_only:
        print("\n[inspect_only] done — no file written.")
        return 0

    # --- subset to T cells ---
    tmask = obs[lin_col].map(_is_tcell).to_numpy()
    if tmask.sum() < 1000:
        raise SystemExit(f"Only {int(tmask.sum())} T cells flagged — check --lineage_field.")
    sub = adata[tmask].copy()
    sub_obs = sub.obs

    # --- timepoint (canonical) ---
    canon_tp = sub_obs[tp_col].map(_canon_timepoint)
    keep_tp = canon_tp.notna().to_numpy()
    sub = sub[keep_tp].copy()
    sub.obs["timepoint"] = canon_tp[keep_tp].astype(str).values
    sub.obs["subject"] = sub.obs[donor_col].astype(str).values
    sub.obs["tcell_lineage"] = sub.obs[lin_col].map(_lineage_label).astype(str).values

    # --- maturation state ---
    tmask_sub = np.ones(sub.n_obs, dtype=bool)  # all are T cells now
    src = args.state_source
    state = None
    if src in ("auto", "protein"):
        state, pidx, scheme = _state_from_protein(sub, tmask_sub)
        if state is not None:
            print(f"[state] using PROTEIN gating scheme={scheme} (markers {pidx})")
        elif src == "protein":
            raise SystemExit("--state_source protein but no usable marker set in panel.")
    if state is None:  # auto-fallback or explicit annotation
        if fine_col is None:
            raise SystemExit("No protein gating and no fine annotation for state labeling.")
        state = sub.obs[fine_col].map(_state_from_annotation).to_numpy()
        print(f"[state] using ANNOTATION '{fine_col}' parsing")

    state = np.asarray(state, dtype=object)
    # day-0 T cells -> the control 'Resting' (kept distinct from the state conditions)
    is_d0 = (sub.obs["timepoint"].values == "D0")
    state[is_d0] = CONTROL
    sub.obs["state"] = state

    # keep only rows with a usable state (Resting + the 3 states); drop 'NA'/None
    valid = np.isin(sub.obs["state"].astype(str).values, [CONTROL] + STATES)
    sub = sub[valid].copy()

    # sanity: non-negative expression (Seurat RNA @counts is raw; just verify)
    xmin = float(sub.X.min())
    if xmin < 0:
        raise SystemExit(f"RNA X has negatives (min={xmin}); expected raw counts/log-norm.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # slim obs to the standardized columns cascadir uses (+ keep originals lean)
    sub.obs = sub.obs[["state", "timepoint", "subject", "tcell_lineage"]].copy()
    sub.write_h5ad(args.out)
    print(f"[write] {args.out}  ({sub.n_obs} T cells × {sub.n_vars} genes)")

    summary = {
        "raw": args.raw, "out": args.out, "state_source": args.state_source,
        "detected": {"donor": donor_col, "timepoint": tp_col, "lineage": lin_col,
                     "fine": fine_col},
        "control_label": CONTROL, "states_ordered": STATES, "timepoints_ordered": TIMEPOINTS,
        "n_cells": int(sub.n_obs), "n_genes": int(sub.n_vars),
        "cells_per_state": sub.obs["state"].value_counts().to_dict(),
        "cells_per_timepoint": sub.obs["timepoint"].value_counts().to_dict(),
        "lineages": sub.obs["tcell_lineage"].value_counts().to_dict(),
        "n_subjects": int(sub.obs["subject"].nunique()),
        "subjects_per_state": sub.obs.groupby("state")["subject"].nunique().to_dict(),
        "subjects_per_timepoint": sub.obs.groupby("timepoint")["subject"].nunique().to_dict(),
    }
    summ_path = Path(args.out).parent / "obs_summary.json"
    summ_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[write] {summ_path}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
