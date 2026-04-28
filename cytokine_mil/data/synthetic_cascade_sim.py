"""
Synthetic cytokine pseudo-tube simulator with known cascade ground truth.
v2 — calibrated to real Oesinghaus data statistics.

Key design changes vs v1:
  - Expression scale matched to real log1p Oesinghaus data (marker_high ≈ 1.2,
    not 4.0; housekeeping ≈ 0.3, not 2.0; cell_noise ≈ 0.15, not 0.4).
  - Programs drawn from a SHARED response pool (hub-gene architecture), not
    exclusive per-cytokine blocks.  Gives realistic program overlap and SVD
    effective rank ≈ 20 matching real data (was artificially orthogonal).
  - n_genes = 1000: 100 HK + 12×25 markers + 400 response pool + 100 background.

Hub-gene architecture:
  - Response pool split into n_hub_genes "popular" genes (shared by ~60% of
    cytokines) and a specific-gene remainder.
  - Each cytokine activates hub genes with prob hub_activation_prob and draws
    n_specific_per_cytokine unique genes from the specific pool.
  - Similar pairs additionally share ≥70% of each other's specific genes.
  - Cascade A→B: B's full δ vector (hub + specific) is added to B's responders
    inside A-tubes at β scale, AFTER A's own primary perturbation (two-pass).

Cascade ordering (unchanged from v1):
  PASS 0: baseline (μ_T + donor_offset) — no perturbation
  PASS 1: _apply_primary_perturbation — cytokine's program → own responders
  PASS 2: _apply_cascade_perturbation — β × child δ → child's responders
  PASS 3: _finalize_tube — per-cell noise + clip + log1p
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Top-level parameters for the synthetic dataset (v2 — calibrated)."""

    n_cell_types: int = 12
    n_cytokines: int = 20            # excluding PBS
    n_genes: int = 1000              # HK(100) + markers(300) + pool(400) + bg(200)
    n_donors: int = 6
    n_pseudo_tubes: int = 8
    n_per_cell_type: int = 30

    # Gene-pool sizes.
    n_housekeeping: int = 100
    n_markers_per_type: int = 25     # 12 × 25 = 300

    # Shared response pool (replaces per-cytokine exclusive blocks in v1).
    n_response_pool: int = 400       # total shared pool
    n_hub_genes: int = 30            # first n_hub_genes of pool — many cytokines use these
    hub_activation_prob: float = 0.60  # P(cytokine activates a given hub gene)
    n_specific_per_cytokine: int = 20  # genes drawn from the non-hub pool per cytokine

    # Calibrated expression parameters (matched to real log1p Oesinghaus data).
    housekeeping_mu: float = 0.30
    housekeeping_sigma: float = 0.15
    marker_high_mu: float = 1.20     # real ct_means p95 ≈ 0.41, max ≈ 6.1 → use 1.2
    marker_high_sigma: float = 0.25
    marker_low_mu: float = 0.02      # real ct_means median ≈ 0.002
    marker_low_sigma: float = 0.02
    response_baseline_mu: float = 0.05   # resting expression of pool genes
    response_baseline_sigma: float = 0.05
    background_mu: float = 0.02
    background_sigma: float = 0.03

    program_magnitude_mu: float = 0.50   # per-gene; L2 ≈ √38 × 0.5 ≈ 3.1 → matches real 4.5
    program_magnitude_sigma: float = 0.10
    fraction_with_downreg: float = 0.20
    n_responders_min: int = 2
    n_responders_max: int = 4
    effect_min: float = 0.6
    effect_max: float = 1.0

    cell_noise_sigma: float = 0.15   # real mean within-CT std ≈ 0.156
    donor_offset_sigma: float = 0.06

    apply_log1p: bool = True
    seed: int = 0


# ---------------------------------------------------------------------------
# Cascade graph
# ---------------------------------------------------------------------------


@dataclass
class CascadeGraph:
    """
    Ground-truth cascade structure.

    cascades : list of (src, dst, beta) — src tube gets a β-scaled dst program
               applied to dst's responder cells, AFTER src's primary effect.
    similar  : list of (a, b) cytokine pairs sharing ≥similar_share_frac of
               their specific (non-hub) program genes (jittered magnitudes),
               without any cascade edge.
    isolated : cytokines with no incoming or outgoing cascade edges.
    """

    cascades: List[Tuple[str, str, float]] = field(default_factory=list)
    similar: List[Tuple[str, str]] = field(default_factory=list)
    isolated: List[str] = field(default_factory=list)
    similar_share_frac: float = 0.70

    def out_edges(self, src: str) -> List[Tuple[str, float]]:
        return [(b, beta) for (a, b, beta) in self.cascades if a == src]

    def to_json(self) -> dict:
        return {
            "cascades": [
                {"src": a, "dst": b, "beta": beta} for (a, b, beta) in self.cascades
            ],
            "similar": [list(pair) for pair in self.similar],
            "isolated": list(self.isolated),
            "similar_share_frac": self.similar_share_frac,
        }


def default_cascade_graph() -> CascadeGraph:
    """The baked-in cascade structure."""
    return CascadeGraph(
        cascades=[
            ("cy1",  "cy2",  0.45),                          # short, strong
            ("cy3",  "cy4",  0.45),                          # 2-step part 1
            ("cy4",  "cy5",  0.35),                          # 2-step part 2
            ("cy6",  "cy7",  0.40),
            ("cy8",  "cy9",  0.40), ("cy8", "cy10", 0.40),  # fan-out
            ("cy11", "cy12", 0.40), ("cy13", "cy12", 0.40), # fan-in
        ],
        similar=[("cy14", "cy15"), ("cy16", "cy17")],
        isolated=["cy18", "cy19", "cy20"],
    )


# ---------------------------------------------------------------------------
# Gene-pool layout (v2: shared response pool, no exclusive cytokine blocks)
# ---------------------------------------------------------------------------


@dataclass
class GeneLayout:
    """Index ranges (relative to [0, n_genes)) for each gene pool."""

    housekeeping: np.ndarray
    markers_by_type: Dict[str, np.ndarray]
    response_pool: np.ndarray        # all pool genes (hub + specific)
    hub_genes: np.ndarray            # first n_hub_genes of response_pool
    specific_pool: np.ndarray        # response_pool[n_hub_genes:]
    background: np.ndarray
    gene_names: List[str]


def _build_gene_layout(
    cfg: SimConfig, cell_types: List[str], cytokines: List[str]
) -> GeneLayout:
    used = 0
    housekeeping = np.arange(used, used + cfg.n_housekeeping)
    used += cfg.n_housekeeping

    markers_by_type: Dict[str, np.ndarray] = {}
    for ct in cell_types:
        markers_by_type[ct] = np.arange(used, used + cfg.n_markers_per_type)
        used += cfg.n_markers_per_type

    # Shared response pool (no per-cytokine exclusive blocks).
    response_pool = np.arange(used, used + cfg.n_response_pool)
    used += cfg.n_response_pool
    hub_genes    = response_pool[: cfg.n_hub_genes]
    specific_pool = response_pool[cfg.n_hub_genes:]

    if used > cfg.n_genes:
        raise ValueError(
            f"Gene pools require {used} genes but n_genes={cfg.n_genes}. "
            "Increase n_genes or reduce pool sizes."
        )
    background = np.arange(used, cfg.n_genes)

    gene_names = [f"gene_{i:04d}" for i in range(cfg.n_genes)]
    return GeneLayout(
        housekeeping=housekeeping,
        markers_by_type=markers_by_type,
        response_pool=response_pool,
        hub_genes=hub_genes,
        specific_pool=specific_pool,
        background=background,
        gene_names=gene_names,
    )


# ---------------------------------------------------------------------------
# Cell-type baseline means
# ---------------------------------------------------------------------------


def _make_cell_type_means(
    cfg: SimConfig,
    cell_types: List[str],
    layout: GeneLayout,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Return per-cell-type baseline mean vector μ_T ∈ R^n_genes."""
    means: Dict[str, np.ndarray] = {}
    for ct in cell_types:
        mu = np.zeros(cfg.n_genes, dtype=np.float32)
        # Housekeeping (shared across all types).
        mu[layout.housekeeping] = rng.normal(
            cfg.housekeeping_mu, cfg.housekeeping_sigma, size=len(layout.housekeeping)
        )
        # This type's markers: high.
        mu[layout.markers_by_type[ct]] = rng.normal(
            cfg.marker_high_mu, cfg.marker_high_sigma,
            size=len(layout.markers_by_type[ct]),
        )
        # Other types' markers: low.
        for other in cell_types:
            if other == ct:
                continue
            mu[layout.markers_by_type[other]] = rng.normal(
                cfg.marker_low_mu, cfg.marker_low_sigma,
                size=len(layout.markers_by_type[other]),
            )
        # Shared response pool: low resting baseline for all cell types.
        mu[layout.response_pool] = rng.normal(
            cfg.response_baseline_mu, cfg.response_baseline_sigma,
            size=len(layout.response_pool),
        )
        # Background filler.
        mu[layout.background] = rng.normal(
            cfg.background_mu, cfg.background_sigma, size=len(layout.background)
        )
        means[ct] = np.clip(mu, 0.0, None).astype(np.float32)
    return means


# ---------------------------------------------------------------------------
# Cytokine programs (v2: hub + specific pool architecture)
# ---------------------------------------------------------------------------


@dataclass
class CytokinePrograms:
    """δ_C, responders(C), and per-(C,T) effect gain. PBS not stored."""

    delta: Dict[str, np.ndarray]                   # cyt → R^n_genes (sparse)
    responders: Dict[str, List[str]]
    effect: Dict[Tuple[str, str], float]
    hub_genes_used: Dict[str, List[int]]            # which hub genes each cyt uses
    specific_genes_used: Dict[str, List[int]]       # which specific-pool genes each cyt uses

    def to_json(self, layout: GeneLayout) -> dict:
        out = {}
        for cyt, vec in self.delta.items():
            nz = np.flatnonzero(vec).tolist()
            out[cyt] = {
                "responders":          self.responders[cyt],
                "program_gene_indices": nz,
                "program_gene_values": [float(vec[i]) for i in nz],
                "hub_gene_indices":    self.hub_genes_used[cyt],
                "specific_gene_indices": self.specific_genes_used[cyt],
                "effect_gain": {
                    ct: float(self.effect[(cyt, ct)])
                    for ct in self.responders[cyt]
                },
            }
        return out


def _make_cytokine_programs(
    cfg: SimConfig,
    cytokines: List[str],
    cell_types: List[str],
    layout: GeneLayout,
    graph: CascadeGraph,
    rng: np.random.Generator,
) -> CytokinePrograms:
    delta: Dict[str, np.ndarray] = {}
    responders: Dict[str, List[str]] = {}
    effect: Dict[Tuple[str, str], float] = {}
    hub_genes_used: Dict[str, List[int]] = {}
    specific_genes_used: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ #
    # 1. Build primary program δ_C for each cytokine.
    #    Hub genes: activated with prob hub_activation_prob (shared signal).
    #    Specific genes: n_specific_per_cytokine drawn from specific pool.
    # ------------------------------------------------------------------ #
    for cyt in cytokines:
        vec = np.zeros(cfg.n_genes, dtype=np.float32)

        # Hub genes.
        hub_mask = rng.random(len(layout.hub_genes)) < cfg.hub_activation_prob
        activated_hubs = layout.hub_genes[hub_mask].tolist()

        # Specific genes.
        n_spec = min(cfg.n_specific_per_cytokine, len(layout.specific_pool))
        spec_genes = rng.choice(layout.specific_pool, size=n_spec, replace=False).tolist()

        all_program_genes = np.array(activated_hubs + spec_genes, dtype=int)

        if len(all_program_genes) == 0:
            # Fallback: pick at least a few genes.
            all_program_genes = rng.choice(layout.response_pool, size=5, replace=False)

        # Assign positive magnitudes.
        mags = np.abs(
            rng.normal(cfg.program_magnitude_mu, cfg.program_magnitude_sigma,
                       size=len(all_program_genes))
        ).astype(np.float32)
        vec[all_program_genes] = mags

        # Optional down-regulation (a few genes flipped negative).
        if rng.random() < cfg.fraction_with_downreg and len(all_program_genes) >= 3:
            n_down = int(rng.integers(1, min(4, len(all_program_genes))))
            down_idx = rng.choice(all_program_genes, size=n_down, replace=False)
            vec[down_idx] = -np.abs(
                rng.normal(cfg.program_magnitude_mu * 0.5,
                           cfg.program_magnitude_sigma, size=n_down)
            ).astype(np.float32)

        delta[cyt] = vec
        hub_genes_used[cyt] = activated_hubs
        specific_genes_used[cyt] = spec_genes

        # Responder cell types and per-(cyt, ct) effect gain.
        n_resp = int(rng.integers(cfg.n_responders_min, cfg.n_responders_max + 1))
        resp = sorted(rng.choice(cell_types, size=n_resp, replace=False).tolist())
        responders[cyt] = resp
        for ct in cell_types:
            effect[(cyt, ct)] = (
                float(rng.uniform(cfg.effect_min, cfg.effect_max))
                if ct in resp else 0.0
            )

    # ------------------------------------------------------------------ #
    # 2. Similar-pair sharing: give each partner ≥ similar_share_frac of
    #    the other's SPECIFIC (non-hub) program genes (with jitter).
    # ------------------------------------------------------------------ #
    def _jitter(v: np.ndarray) -> np.ndarray:
        return (v + rng.normal(0.0, 0.12, size=v.shape)).astype(np.float32)

    for a, b in graph.similar:
        spec_a = np.array(specific_genes_used[a], dtype=int)
        spec_b = np.array(specific_genes_used[b], dtype=int)

        n_share_a = max(1, int(round(graph.similar_share_frac * len(spec_a))))
        n_share_b = max(1, int(round(graph.similar_share_frac * len(spec_b))))

        if len(spec_a) >= n_share_a:
            share_from_a = rng.choice(spec_a, size=n_share_a, replace=False)
            delta[b][share_from_a] = _jitter(delta[a][share_from_a])

        if len(spec_b) >= n_share_b:
            share_from_b = rng.choice(spec_b, size=n_share_b, replace=False)
            delta[a][share_from_b] = _jitter(delta[b][share_from_b])

    return CytokinePrograms(
        delta=delta,
        responders=responders,
        effect=effect,
        hub_genes_used=hub_genes_used,
        specific_genes_used=specific_genes_used,
    )


# ---------------------------------------------------------------------------
# Tube sampling (PASS 0–3)
# ---------------------------------------------------------------------------


def _sample_baseline_tube(
    cell_types: List[str],
    cell_type_means: Dict[str, np.ndarray],
    donor_offset: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[str]]:
    """PASS 0: baseline (μ_T + donor_offset). No noise, no cytokine effect."""
    rows: List[np.ndarray] = []
    cell_type_per_row: List[str] = []
    for ct in cell_types:
        mu = cell_type_means[ct] + donor_offset
        block = np.broadcast_to(mu, (cfg.n_per_cell_type, cfg.n_genes)).copy()
        rows.append(block)
        cell_type_per_row.extend([ct] * cfg.n_per_cell_type)
    X = np.concatenate(rows, axis=0).astype(np.float32)
    return X, cell_type_per_row


def _apply_primary_perturbation(
    X: np.ndarray,
    cell_type_per_row: List[str],
    cytokine: str,
    programs: CytokinePrograms,
) -> None:
    """PASS 1: add the cytokine's own program to its responder cells (in-place)."""
    if cytokine == "PBS":
        return
    delta = programs.delta[cytokine]
    for i, ct in enumerate(cell_type_per_row):
        gain = programs.effect[(cytokine, ct)]
        if gain > 0.0:
            X[i] += gain * delta


def _apply_cascade_perturbation(
    X: np.ndarray,
    cell_type_per_row: List[str],
    cytokine: str,
    graph: CascadeGraph,
    programs: CytokinePrograms,
    max_hops: int = 2,
) -> None:
    """
    PASS 2: for every cascade edge cytokine→child, add β·δ(child) to child's
    responders.  Recurse one extra hop (β² decay).  Applied AFTER PASS 1.
    """
    if cytokine == "PBS":
        return

    def _recurse(src: str, accumulated_beta: float, hops_left: int) -> None:
        if hops_left == 0:
            return
        for (dst, beta) in graph.out_edges(src):
            effective_beta = accumulated_beta * beta
            delta_dst = programs.delta[dst]
            for i, ct in enumerate(cell_type_per_row):
                gain = programs.effect[(dst, ct)]
                if gain > 0.0:
                    X[i] += effective_beta * gain * delta_dst
            _recurse(dst, effective_beta, hops_left - 1)

    _recurse(cytokine, 1.0, max_hops)


def _finalize_tube(
    X: np.ndarray, cfg: SimConfig, rng: np.random.Generator
) -> np.ndarray:
    """PASS 3: per-cell Gaussian noise → clip ≥ 0 → optional log1p."""
    noise = rng.normal(0.0, cfg.cell_noise_sigma, size=X.shape).astype(np.float32)
    X = X + noise
    np.clip(X, 0.0, None, out=X)
    if cfg.apply_log1p:
        X = np.log1p(X)
    return X


def _build_anndata(
    X: np.ndarray,
    cell_type_per_row: List[str],
    donor: str,
    cytokine: str,
    tube_idx: int,
    gene_names: List[str],
    rng: np.random.Generator,
) -> ad.AnnData:
    n = X.shape[0]
    perm = rng.permutation(n)
    X = X[perm]
    cell_type_per_row = [cell_type_per_row[i] for i in perm]

    obs = pd.DataFrame(
        {"cell_type": cell_type_per_row, "donor": donor, "cytokine": cytokine},
        index=[f"{donor}_{cytokine}_{tube_idx}_cell{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=gene_names)
    return ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def generate_dataset(
    out_dir: str,
    cfg: Optional[SimConfig] = None,
    graph: Optional[CascadeGraph] = None,
) -> str:
    """
    Generate the full synthetic dataset on disk.  Returns the manifest.json path.
    """
    cfg   = cfg   or SimConfig()
    graph = graph or default_cascade_graph()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    cell_types     = [f"ct{i+1}" for i in range(cfg.n_cell_types)]
    cytokines      = [f"cy{i+1}" for i in range(cfg.n_cytokines)]
    donors         = [f"Donor{i+1}" for i in range(cfg.n_donors)]
    all_conditions = cytokines + ["PBS"]

    layout         = _build_gene_layout(cfg, cell_types, cytokines)
    cell_type_means = _make_cell_type_means(cfg, cell_types, layout, rng)
    programs        = _make_cytokine_programs(cfg, cytokines, cell_types, layout, graph, rng)

    donor_offsets: Dict[str, np.ndarray] = {
        d: rng.normal(0.0, cfg.donor_offset_sigma, size=cfg.n_genes).astype(np.float32)
        for d in donors
    }

    manifest: List[dict] = []
    for donor in donors:
        for cytokine in all_conditions:
            folder = out_path / donor / cytokine
            folder.mkdir(parents=True, exist_ok=True)
            for tube_idx in range(cfg.n_pseudo_tubes):
                X, ct_per_row = _sample_baseline_tube(
                    cell_types, cell_type_means, donor_offsets[donor], cfg, rng
                )
                _apply_primary_perturbation(X, ct_per_row, cytokine, programs)
                _apply_cascade_perturbation(X, ct_per_row, cytokine, graph, programs)
                X = _finalize_tube(X, cfg, rng)

                adata = _build_anndata(
                    X, ct_per_row, donor, cytokine, tube_idx, layout.gene_names, rng
                )
                tube_path = folder / f"pseudotube_{tube_idx}.h5ad"
                adata.write_h5ad(str(tube_path))

                manifest.append({
                    "path":               str(tube_path),
                    "donor":              donor,
                    "cytokine":           cytokine,
                    "n_cells":            int(adata.n_obs),
                    "cell_types_included": cell_types,
                    "tube_idx":           tube_idx,
                })

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    with open(out_path / "cascade_ground_truth.json", "w") as f:
        json.dump(graph.to_json(), f, indent=2)

    with open(out_path / "cytokine_programs.json", "w") as f:
        json.dump(programs.to_json(layout), f, indent=2)

    with open(out_path / "hvg_list.json", "w") as f:
        json.dump(layout.gene_names, f, indent=2)

    with open(out_path / "sim_config.json", "w") as f:
        json.dump({k: v for k, v in cfg.__dict__.items()}, f, indent=2, default=str)

    return str(manifest_path)


__all__ = [
    "SimConfig",
    "CascadeGraph",
    "default_cascade_graph",
    "generate_dataset",
]
