"""Gene layout, cell-type means, per-label programs, responder masks, and planting.

Everything here is in **log1p space** (constants calibrated to real log1p Oesinghaus
data, reused from ``cytokine_mil/data/synthetic_cascade_sim.py``): a cell's expression is

    expr = celltype_mean[type] + donor_offset[donor] + N(0, cell_noise)
           + effect_size * sum_X  activation[X] * responder_mask[X, type] * program[X]

where the **cell-type signature** (marker genes, ``marker_high_mu ~ 1.2`` over a
``housekeeping_mu ~ 0.30`` baseline) is the DOMINANT term, and each label's **program**
is a deliberately weak additive bump (``effect_size ~ 0.30`` at full activation, << the
marker gap) — weaker than the cell type but detectable at the bag (pseudo-tube) level
where cascadir's MIL + Integrated Gradients operate.

The direction signal: an applied label ``L`` has ``activation[L] = 1`` (its own program
fully on) plus partial activation of its downstream labels (the autocrine relay). A
downstream label ``Y``, applied on its own, does NOT switch on its upstream ``X`` — that
asymmetry is exactly what ``cross_asym = s(a, S_b) - s(b, S_a)`` reads as direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

from .dynamics import ActivationVector
from .graph import NormEdges


@dataclass
class ExpressionConfig:
    """Calibrated expression parameters (log1p space). All overridable."""

    # Gene-pool sizes.
    n_housekeeping: int = 50
    n_markers_per_type: int = 20
    n_program_genes: int = 50        # per label; matches cascadir top_n so S_X ~ program
    n_background_genes: int = 200
    n_shared_genes: int = 30         # shared-activation pool (used iff shared_activation > 0)

    # Baseline means (log1p space; from real log1p Oesinghaus data).
    housekeeping_mu: float = 0.30
    marker_high_mu: float = 1.20
    marker_low_mu: float = 0.02
    program_rest_mu: float = 0.12    # resting expression of program genes (not silent at rest)
    background_mu: float = 0.05
    shared_rest_mu: float = 0.12

    # Noise / replicate structure.
    cell_noise_sigma: float = 0.15
    donor_offset_sigma: float = 0.06
    program_loading_jitter: float = 0.20   # program-gene loadings ~ U(1-j, 1+j)

    # Perturbation strength.
    effect_size: float = 0.30        # weak program bump at full activation (<< marker gap 1.2)
    shared_activation: float = 0.0   # fraction of effect_size that every active label adds
                                     # to the SHARED pool (the §22/§28 confound); 0 = clean
    non_responder_effect: float = 0.05   # soft effect for non-responder cell types (v4 fix)
    responder_frac: float = 0.6      # receptor mode: fraction of cell types that respond


@dataclass
class GeneModel:
    """Everything fixed once per simulation: layout, means, programs, responders."""

    gene_names: List[str]
    cell_types: List[str]
    labels: List[str]
    celltype_mean: np.ndarray                    # (K, G), log1p space
    donor_offset: np.ndarray                     # (D, G)
    program_vecs: Dict[str, np.ndarray]          # label -> (G,), nonzero on its program block
    program_genes: Dict[str, List[str]]          # label -> list of gene names
    responders: Dict[str, List[str]]             # label -> cell-type names that respond
    mask_by_type: Dict[str, np.ndarray]          # label -> (K,) soft responder mask
    shared_idx: np.ndarray                       # shared-pool gene indices ((0,) if unused)
    shared_loadings: np.ndarray                  # (n_shared,) loadings
    cfg: ExpressionConfig = field(default_factory=ExpressionConfig)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)


def _assign_responders(
    labels: Sequence[str],
    edges: NormEdges,
    cell_types: Sequence[str],
    *,
    mode: str,
    responder_frac: float,
    user_responders: Optional[Mapping[str, Sequence[str]]],
    rng: np.random.Generator,
) -> Dict[str, Set[str]]:
    """Which cell types respond to each label.

    ``mode="all"``: every cell type responds to every label.
    ``mode="receptor"``: each label responds in a random subset (size
    ``ceil(K * responder_frac)``, >= 1), then ``resp(src) subset resp(dst)`` is enforced
    for every edge by a union-to-fixpoint so the cascade direction stays correct (the v5
    subset constraint). ``user_responders`` overrides the random subset for named labels.
    """
    cell_types = list(cell_types)
    if mode == "all":
        return {lab: set(cell_types) for lab in labels}
    if mode != "receptor":
        raise ValueError(f"responder_mode must be 'all' or 'receptor', got {mode!r}")

    k = len(cell_types)
    n_resp = max(1, int(np.ceil(responder_frac * k)))
    resp: Dict[str, Set[str]] = {}
    for lab in labels:
        if user_responders and lab in user_responders:
            chosen = set(str(c) for c in user_responders[lab])
            missing = chosen - set(cell_types)
            if missing:
                raise ValueError(f"responders[{lab!r}] names unknown cell types: {sorted(missing)}")
            resp[lab] = chosen or set(cell_types)
        else:
            idx = rng.choice(k, size=n_resp, replace=False)
            resp[lab] = {cell_types[i] for i in idx}

    # Enforce resp(src) subset resp(dst) for every edge: union to fixpoint.
    changed = True
    while changed:
        changed = False
        for src, downstream in edges.items():
            for dst in downstream:
                before = len(resp[dst])
                resp[dst] |= resp[src]
                if len(resp[dst]) != before:
                    changed = True
    return resp


def build_gene_model(
    labels: Sequence[str],
    edges: NormEdges,
    *,
    n_cell_types: int,
    n_donors: int,
    n_program_genes: Optional[int] = None,
    effect_size: Optional[float] = None,
    responder_mode: str = "all",
    responders: Optional[Mapping[str, Sequence[str]]] = None,
    cfg: Optional[ExpressionConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeneModel:
    """Construct the fixed gene model (layout + means + programs + responders)."""
    rng = np.random.default_rng(0) if rng is None else rng
    cfg = ExpressionConfig() if cfg is None else cfg
    if n_program_genes is not None:
        cfg.n_program_genes = int(n_program_genes)
    if effect_size is not None:
        cfg.effect_size = float(effect_size)
    labels = list(labels)
    cell_types = [f"celltype_{i}" for i in range(n_cell_types)]
    use_shared = cfg.shared_activation > 0 and cfg.n_shared_genes > 0

    # --- Gene index layout ---
    idx = 0
    housekeeping_idx = np.arange(idx, idx + cfg.n_housekeeping); idx += cfg.n_housekeeping
    marker_idx: Dict[int, np.ndarray] = {}
    for t in range(n_cell_types):
        marker_idx[t] = np.arange(idx, idx + cfg.n_markers_per_type); idx += cfg.n_markers_per_type
    program_idx: Dict[str, np.ndarray] = {}
    for lab in labels:
        program_idx[lab] = np.arange(idx, idx + cfg.n_program_genes); idx += cfg.n_program_genes
    if use_shared:
        shared_idx = np.arange(idx, idx + cfg.n_shared_genes); idx += cfg.n_shared_genes
    else:
        shared_idx = np.arange(0, 0)
    background_idx = np.arange(idx, idx + cfg.n_background_genes); idx += cfg.n_background_genes
    n_genes = idx
    gene_names = [f"gene_{i:05d}" for i in range(n_genes)]

    # --- Cell-type mean matrix (log1p space) ---
    mu = np.zeros((n_cell_types, n_genes), dtype=np.float64)
    mu[:, housekeeping_idx] = cfg.housekeeping_mu
    mu[:, background_idx] = cfg.background_mu
    for lab in labels:
        mu[:, program_idx[lab]] = cfg.program_rest_mu
    if use_shared:
        mu[:, shared_idx] = cfg.shared_rest_mu
    for t in range(n_cell_types):
        for t2 in range(n_cell_types):
            mu[t, marker_idx[t2]] = cfg.marker_high_mu if t2 == t else cfg.marker_low_mu

    # --- Donor offsets (per donor, per gene) ---
    donor_offset = rng.normal(0.0, cfg.donor_offset_sigma, size=(n_donors, n_genes))

    # --- Per-label program vectors (loadings on the label's own block only) ---
    program_vecs: Dict[str, np.ndarray] = {}
    program_genes: Dict[str, List[str]] = {}
    for lab in labels:
        vec = np.zeros(n_genes, dtype=np.float64)
        j = cfg.program_loading_jitter
        loadings = rng.uniform(1.0 - j, 1.0 + j, size=cfg.n_program_genes)
        vec[program_idx[lab]] = loadings
        program_vecs[lab] = vec
        program_genes[lab] = [gene_names[i] for i in program_idx[lab]]
    shared_loadings = (
        rng.uniform(1.0 - cfg.program_loading_jitter, 1.0 + cfg.program_loading_jitter,
                    size=len(shared_idx))
        if use_shared else np.zeros(0)
    )

    # --- Responder assignment + soft masks ---
    resp = _assign_responders(
        labels, edges, cell_types,
        mode=responder_mode, responder_frac=cfg.responder_frac,
        user_responders=responders, rng=rng,
    )
    mask_by_type: Dict[str, np.ndarray] = {}
    for lab in labels:
        m = np.full(n_cell_types, cfg.non_responder_effect, dtype=np.float64)
        for ci, ct in enumerate(cell_types):
            if ct in resp[lab]:
                m[ci] = 1.0
        mask_by_type[lab] = m

    return GeneModel(
        gene_names=gene_names,
        cell_types=cell_types,
        labels=labels,
        celltype_mean=mu,
        donor_offset=donor_offset,
        program_vecs=program_vecs,
        program_genes=program_genes,
        responders={lab: sorted(resp[lab]) for lab in labels},
        mask_by_type=mask_by_type,
        shared_idx=shared_idx,
        shared_loadings=shared_loadings,
        cfg=cfg,
    )


def generate_tube(
    model: GeneModel,
    donor_idx: int,
    activation: Optional[ActivationVector],
    *,
    n_cells: int,
    output: str,
    rng: np.random.Generator,
    sparse: bool = False,
):
    """Generate one tube (bag) of cells for a (donor, condition).

    Args:
        model: the fixed gene model.
        donor_idx: which donor (indexes ``model.donor_offset``).
        activation: ``{label: activation}`` at the snapshot time, or ``None`` for the
            control (PBS) condition (base tube, no program added).
        n_cells: number of cells in the tube.
        output: ``"raw"`` (Poisson integer counts) or ``"lognorm"`` (log1p-space floats).
        rng: random generator.
        sparse: if True, return ``X`` as a ``scipy.sparse.csr_matrix`` (bit-identical to
            the dense draw; useful for large raw-count runs). Default dense ndarray.

    Returns:
        ``(X, cell_type_names)`` with ``X`` shape ``(n_cells, n_genes)`` (dense ndarray
        or CSR matrix depending on ``sparse``).
    """
    cfg = model.cfg
    k = len(model.cell_types)
    # Stratify cells ~equally across cell types.
    type_idx = np.arange(n_cells) % k
    rng.shuffle(type_idx)
    cell_type_names = [model.cell_types[t] for t in type_idx]

    X = model.celltype_mean[type_idx] + model.donor_offset[donor_idx][None, :]
    X = X + rng.normal(0.0, cfg.cell_noise_sigma, size=X.shape)

    if activation:
        contrib = np.zeros_like(X)
        total_act = 0.0
        for lab, act in activation.items():
            if act <= 0.0:
                continue
            total_act += act
            mask_per_cell = model.mask_by_type[lab][type_idx]            # (n_cells,)
            contrib += (act * mask_per_cell)[:, None] * model.program_vecs[lab][None, :]
        X = X + cfg.effect_size * contrib
        if cfg.shared_activation > 0 and len(model.shared_idx) > 0 and total_act > 0:
            bump = cfg.effect_size * cfg.shared_activation * total_act
            X[:, model.shared_idx] += bump * model.shared_loadings[None, :]

    X = np.clip(X, 0.0, None)
    if output == "lognorm":
        out = X.astype(np.float32)
    elif output == "raw":
        rate = np.expm1(X)                       # so log1p(counts) ~ X in expectation
        out = rng.poisson(rate).astype(np.float32)
    else:
        raise ValueError(f"output must be 'raw' or 'lognorm', got {output!r}")
    if sparse:
        from scipy import sparse as sp
        out = sp.csr_matrix(out)
    return out, cell_type_names
