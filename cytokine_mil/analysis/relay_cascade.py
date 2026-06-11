"""
Directed cross-cell relay cascades (gene-level), Immune Dictionary.

SETTING (the user's reframe). Predict a held-out cell's per-cell-type expression from the OTHER
cells of its own pseudo-tube ("soup"), with each gene predicted only from *other* genes (a hollow
(cell_type, gene) diagonal). Direction is read from the asymmetry of the learned relay-influence
tensor (source_type, gene) -> (target_type, gene). The leave-one-out (target excluded from the soup
aggregate) is what creates the population->cell asymmetry.

IDENTIFIABILITY (proven in the toy; appendix of the plan). Direction is recovered for the ONE-HOP
cell-autonomous-source -> population-responder relation. It collapses (becomes symmetric) when both
nodes are tube-level (stimulus-driven) -- the same shared-activation confound as the cytokine work.
This module is the apparatus to test, honestly, whether real ID snapshots carry such a signal.

PRIMARY MODEL. Closed-form **hollow ridge**: for each output (T, g) regress on all inputs EXCEPT its
own (T, g) entry. Self-exclusion is exact and per-output via the leave-one-covariate-out inverse
formula (no SGD, deterministic). The cMLP-hollow (nonlinear) is the secondary model in the driver.

Hollow mask: only the EXACT (cell_type, gene) self entry is removed; cross-type-same-gene
(T', g) -> (T, g) is KEPT -- that is the relay (e.g. NK Ifng -> macrophage ISG).

Allowed imports: numpy only (analysis layer).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Curated relay panel (mouse symbols). Exact list lives here (not in CLAUDE.md).
# ---------------------------------------------------------------------------
RELAY_PANEL: Dict[str, List[str]] = {
    "ifng_producer": ["Ifng", "Tbx21", "Il12rb2", "Il18r1"],
    "ifng_transactivator": ["Stat1", "Irf1", "Irf8", "Nlrc5", "Ciita"],
    "isg_target": ["Isg15", "Mx1", "Mx2", "Oasl2", "Oas1a", "Oas2", "Oas3",
                    "Gbp2", "Gbp3", "Usp18", "Rsad2", "Ifit1", "Ifit3", "Irf7", "Stat2"],
    "il12_axis": ["Il12a", "Il12b", "Stat4", "Ebi3"],
    "il6_axis": ["Il6", "Stat3", "Socs3", "Il6ra"],
    "nfkb_axis": ["Tnf", "Il1b", "Nfkbia", "Tnfaip3", "Cxcl10", "Ccl5"],
    "nk_marker": ["Ncr1", "Klrb1c", "Klrk1", "Gzmb", "Prf1", "Eomes"],
    "mac_marker": ["Adgre1", "Csf1r", "Itgam", "Lyz2", "C1qa", "C1qb"],
    "t_marker": ["Cd3e", "Cd8a", "Cd4", "Foxp3", "Tcf7"],
    "b_marker": ["Cd19", "Ms4a1", "Cd79a", "Ighm"],
    "dc_marker": ["Itgax", "Xcr1", "Batf3", "Flt3"],
    "th2_negative": ["Gata3", "Il4", "Il13", "Il5"],
}


def panel_gene_list() -> List[str]:
    """Flat, de-duplicated curated panel (order-stable)."""
    seen, out = set(), []
    for genes in RELAY_PANEL.values():
        for g in genes:
            if g not in seen:
                seen.add(g)
                out.append(g)
    return out


def panel_indices(panel: Sequence[str]) -> Dict[str, np.ndarray]:
    """Map each RELAY_PANEL category -> ndarray of positions within `panel`."""
    pos = {g: i for i, g in enumerate(panel)}
    out = {}
    for name, genes in RELAY_PANEL.items():
        idx = [pos[g] for g in genes if g in pos]
        if idx:
            out[name] = np.array(idx, dtype=np.int64)
    return out


# ---------------------------------------------------------------------------
# Flattened (cell_type, gene) indexing
# ---------------------------------------------------------------------------
def flat_index(ct_pos: int, gene: int, n_genes: int) -> int:
    return ct_pos * n_genes + gene


def build_flat_layout(cell_types: List[str], n_genes: int) -> Dict[str, object]:
    """Helpers for the flattened C*G space."""
    ct_pos = {ct: i for i, ct in enumerate(cell_types)}
    P = len(cell_types) * n_genes
    # (P,) arrays mapping each flat index back to its cell-type position and gene
    type_of = np.repeat(np.arange(len(cell_types)), n_genes)
    gene_of = np.tile(np.arange(n_genes), len(cell_types))
    return {"ct_pos": ct_pos, "P": P, "type_of": type_of, "gene_of": gene_of,
            "cell_types": cell_types, "n_genes": n_genes}


# ---------------------------------------------------------------------------
# Per-tube leave-one-out sample construction
# ---------------------------------------------------------------------------
def usable_cell_types(tubes: List[dict], candidate_types: Sequence[str],
                      min_cells: int, min_tube_frac: float = 0.6) -> List[str]:
    """Cell types present with >= min_cells in >= min_tube_frac of tubes."""
    counts = {ct: 0 for ct in candidate_types}
    for tube in tubes:
        for ct in candidate_types:
            arr = tube["by_type"].get(ct)
            if arr is not None and len(arr) >= min_cells:
                counts[ct] += 1
    thr = min_tube_frac * len(tubes)
    return [ct for ct in candidate_types if counts[ct] >= thr]


def compute_pbs_type_means(tubes: List[dict], cell_types: List[str], n_genes: int,
                           pbs_label: str = "PBS") -> Dict[str, np.ndarray]:
    """Per-cell-type mean expression over all PBS cells (resting baseline)."""
    acc = {ct: [] for ct in cell_types}
    for tube in tubes:
        if tube["cytokine"] != pbs_label:
            continue
        for ct in cell_types:
            arr = tube["by_type"].get(ct)
            if arr is not None and len(arr):
                acc[ct].append(arr)
    means = {}
    for ct in cell_types:
        means[ct] = (np.concatenate(acc[ct], 0).mean(0) if acc[ct]
                     else np.zeros(n_genes, dtype=np.float64))
    return means


def build_relay_samples(
    tubes: List[dict],
    cell_types: List[str],
    n_genes: int,
    pbs_means: Dict[str, np.ndarray],
    held_frac: float = 0.3,
    min_cells: int = 10,
    n_draws: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Per-tube leave-one-out samples in the flattened C*G space.

    For each tube where ALL `cell_types` have >= min_cells, and each of n_draws random
    splits: held-out per-type MEAN -> target Y; per-type mean over the REST -> input A.
    Both PBS-residualised per type. Returns dense A, Y (n_samples, C*G) + meta.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    P = len(cell_types) * n_genes
    A_rows, Y_rows, cyto, donor = [], [], [], []
    for tube in tubes:
        bt = tube["by_type"]
        if any((bt.get(ct) is None or len(bt[ct]) < max(min_cells, 4)) for ct in cell_types):
            continue
        for _ in range(n_draws):
            a = np.empty(P, dtype=np.float64)
            y = np.empty(P, dtype=np.float64)
            for ci, ct in enumerate(cell_types):
                cells = bt[ct]
                N = len(cells)
                n_held = max(1, int(round(held_frac * N)))
                n_held = min(n_held, N - 1)
                perm = rng.permutation(N)
                held, rest = perm[:n_held], perm[n_held:]
                base = pbs_means[ct]
                a[ci * n_genes:(ci + 1) * n_genes] = cells[rest].mean(0) - base
                y[ci * n_genes:(ci + 1) * n_genes] = cells[held].mean(0) - base
            A_rows.append(a); Y_rows.append(y)
            cyto.append(tube["cytokine"]); donor.append(tube["donor"])
    if not A_rows:
        raise ValueError("No usable tubes — check min_cells / cell_types coverage.")
    return {"A": np.array(A_rows), "Y": np.array(Y_rows),
            "cytokine": np.array(cyto), "donor": np.array(donor),
            "cell_types": cell_types, "n_genes": n_genes, "P": P}


# ---------------------------------------------------------------------------
# Closed-form hollow ridge  (exact per-output self-exclusion)
# ---------------------------------------------------------------------------
def hollow_ridge_influence(A: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Influence M (P, P): M[o, j] = ridge coefficient of input j when predicting output o,
    with input o (= same (cell_type, gene)) EXCLUDED. Exact via the leave-one-covariate-out
    inverse identity:  beta^{-o}_j = B[j,o] - Gi[j,o]/Gi[o,o] * B[o,o],  where
    B = Gi A^T Y (full multi-output ridge) and Gi = (A^T A + alpha I)^{-1}.
    """
    n, P = A.shape
    G = A.T @ A + alpha * np.eye(P)
    Gi = np.linalg.inv(G)
    B = Gi @ (A.T @ Y)                       # (P_in, P_out): B[j, o]
    d = np.diag(Gi).copy()                    # (P,)
    d[d == 0] = 1e-12
    bdiag = np.diag(B).copy()                 # B[o, o]
    # M[o, j] = B[j, o] - Gi[j, o]/d[o] * bdiag[o]
    M = B.T - (Gi.T / d[:, None]) * bdiag[:, None]
    np.fill_diagonal(M, 0.0)
    return M


def predict(M: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Yhat[o] = sum_j M[o, j] A[j]  ->  Yhat = A @ M.T."""
    return A @ M.T


def r2(M: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
    pred = predict(M, A)
    ss_res = float(((Y - pred) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def r2_per_target_type(M, A, Y, layout) -> Dict[str, float]:
    pred = predict(M, A)
    out = {}
    for ci, ct in enumerate(layout["cell_types"]):
        sl = slice(ci * layout["n_genes"], (ci + 1) * layout["n_genes"])
        ss_res = float(((Y[:, sl] - pred[:, sl]) ** 2).sum())
        ss_tot = float(((Y[:, sl] - Y[:, sl].mean(0, keepdims=True)) ** 2).sum())
        out[ct] = 1.0 - ss_res / max(ss_tot, 1e-12)
    return out


# ---------------------------------------------------------------------------
# Direction readout
# ---------------------------------------------------------------------------
def edge_strength(M, layout, src_type, src_genes, tgt_type, tgt_genes) -> float:
    """Mean |influence| from (src_type, src_genes) onto (tgt_type, tgt_genes)."""
    ct = layout["ct_pos"]
    if src_type not in ct or tgt_type not in ct:
        return float("nan")
    ng = layout["n_genes"]
    rows = [ct[tgt_type] * ng + g for g in tgt_genes]
    cols = [ct[src_type] * ng + g for g in src_genes]
    sub = M[np.ix_(rows, cols)]
    return float(np.mean(np.abs(sub)))


def relay_direction(M, layout, src_type, src_genes, tgt_type, tgt_genes) -> Dict[str, float]:
    """Coefficient-magnitude asymmetry. NOTE: |coef| is variance-ratio-confounded
    (regressing Y~X vs X~Y differ by (sigma_Y/sigma_X)^2), so this reflects which node
    has larger variance, NOT causal direction. Kept only as a secondary/attribution
    readout. Use relay_direction_pred (predictability) for direction."""
    fwd = edge_strength(M, layout, src_type, src_genes, tgt_type, tgt_genes)
    rev = edge_strength(M, layout, tgt_type, tgt_genes, src_type, src_genes)
    return {"forward": fwd, "reverse": rev, "asymmetry": fwd - rev,
            "direction": "src->tgt" if fwd > rev else "tgt->src"}


def node_predictability(M, A, Y, layout=None) -> np.ndarray:
    """Per-output R^2: how well each (cell_type, gene) is predicted from the soup with
    its OWN (type, gene) excluded (the hollow M). Variance-NORMALISED, so unlike raw
    coefficient magnitude it is NOT confounded by variance ratios."""
    pred = predict(M, A)
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0, keepdims=True)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)          # (P,)


def relay_direction_pred(M, A, Y, layout, src_type, src_genes, tgt_type, tgt_genes) -> Dict[str, float]:
    """Direction from PREDICTABILITY asymmetry (the correct one-hop readout): a
    downstream/responder node is predictable from the soup; an upstream cell-autonomous
    source is not. pred_tgt - pred_src > 0  =>  src -> tgt. Symmetric (both predictable)
    => no direction (tube-level collapse)."""
    r2v = node_predictability(M, A, Y, layout)
    ct, ng = layout["ct_pos"], layout["n_genes"]
    if src_type not in ct or tgt_type not in ct:
        return {"pred_tgt": float("nan"), "pred_src": float("nan"),
                "asymmetry": float("nan"), "direction": "NA"}
    pt = float(np.mean([r2v[ct[tgt_type] * ng + g] for g in tgt_genes]))
    ps = float(np.mean([r2v[ct[src_type] * ng + g] for g in src_genes]))
    return {"pred_tgt": pt, "pred_src": ps, "asymmetry": pt - ps,
            "direction": "src->tgt" if pt > ps else "tgt->src"}


# ---------------------------------------------------------------------------
# Nulls (numpy)
# ---------------------------------------------------------------------------
def signal_permutation_null(A, Y, alpha, n_perm=200, rng=None) -> Dict[str, float]:
    """Break the soup->cell pairing (shuffle Y rows), refit, recompute global R^2.
    Tests G1: is held-out expression predictable from the soup beyond chance?"""
    if rng is None:
        rng = np.random.default_rng(0)
    obs = r2(hollow_ridge_influence(A, Y, alpha), A, Y)
    null = np.empty(n_perm)
    for k in range(n_perm):
        Yp = Y[rng.permutation(len(Y))]
        null[k] = r2(hollow_ridge_influence(A, Yp, alpha), A, Yp)
    return {"observed_r2": obs, "null_q95": float(np.quantile(null, 0.95)),
            "null_mean": float(null.mean()),
            "p_emp": float((np.sum(null >= obs) + 1) / (n_perm + 1))}


def direction_bootstrap(A, Y, layout, alpha, edge, n_boot=200, rng=None) -> Dict[str, float]:
    """Bootstrap -> CI on the PREDICTABILITY-asymmetry of the relay. `edge` =
    (src_type, src_genes, tgt_type, tgt_genes). Direction reliable if the CI excludes 0;
    sign of asym_mean gives the direction (>0 => src->tgt)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(A)
    asy = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        M = hollow_ridge_influence(A[idx], Y[idx], alpha)
        asy[k] = relay_direction_pred(M, A[idx], Y[idx], layout, *edge)["asymmetry"]
    lo, hi = np.quantile(asy, [0.025, 0.975])
    return {"asym_mean": float(asy.mean()), "asym_q025": float(lo), "asym_q975": float(hi),
            "reliable": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------------------
# Synthetic relay generator (local self-test of the apparatus)
# ---------------------------------------------------------------------------
def simulate_relay_tubes(
    mode: str,                       # "cell_autonomous" (recoverable) | "tube_level" (collapses)
    n_tubes_stim: int = 60,
    n_tubes_pbs: int = 30,
    n_cells: int = 90,
    beta: float = 8.0,
    noise: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Two cell types S (source) and T (target) + a distractor D; 6 genes:
    h0 (source signal), g1 (target relay response), plus 4 background. Planted relay
    S.h0 -> T.g1 in 'stim' tubes only.
      cell_autonomous: each S cell draws h0 independently; T.g1 = beta * leave-out-mean(S.h0).
                       -> held-out S.h0 is NOT predictable from soup -> direction recoverable.
      tube_level:      S.h0 = one tube value shared by all S cells (stimulus-driven);
                       T.g1 = beta * that.  -> S.h0 recoverable from T.g1 -> SYMMETRIC.
    Returns tubes in the build_relay_samples format + the ground-truth indices.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cell_types = ["S", "T", "D"]
    G = 6
    H0, G1 = 0, 1                        # source gene, target relay gene
    base = {"S": np.array([0.2, 0.1, 0.3, 0.3, 0.3, 0.3]),
            "T": np.array([0.1, 0.2, 0.3, 0.3, 0.3, 0.3]),
            "D": np.array([0.1, 0.1, 0.3, 0.3, 0.3, 0.3])}
    tubes = []

    def make_tube(cyto, stim):
        bt = {}
        per = n_cells // 3
        for ct in cell_types:
            X = base[ct][None, :] + rng.normal(0, noise, size=(per, G))
            bt[ct] = X
        if stim:
            # source signal in S cells
            if mode == "cell_autonomous":
                s_h0 = rng.gamma(1.0, 1.0, size=per)            # per-cell independent
            else:  # tube_level
                tube_val = rng.gamma(1.0, 1.0)                  # one value per tube
                s_h0 = tube_val + rng.normal(0, 0.05, size=per)
            bt["S"][:, H0] += s_h0
            agg = s_h0.mean()                                   # paracrine: T responds to aggregate
            bt["T"][:, G1] += beta * agg + rng.normal(0, noise, size=per)
        return {"cytokine": cyto, "donor": "rep01", "by_type": {k: v.astype(np.float32)
                                                                for k, v in bt.items()}}

    for _ in range(n_tubes_stim):
        tubes.append(make_tube("STIM", stim=True))
    for _ in range(n_tubes_pbs):
        tubes.append(make_tube("PBS", stim=False))
    return {"tubes": tubes, "cell_types": cell_types, "n_genes": G,
            "src": ("S", [H0]), "tgt": ("T", [G1])}
