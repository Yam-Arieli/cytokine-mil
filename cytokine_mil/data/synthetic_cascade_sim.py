"""
Synthetic cytokine pseudo-tube simulator with known cascade ground truth.

Generates a drop-in replacement for the Oesinghaus pseudo-tube dataset where
every cytokine response, similarity pair, and cascade edge is known by
construction. Used to validate the full pipeline (encoder → MIL → confusion
dynamics → PBS-RC latent geometry) end-to-end.

Design (see /Users/yam/.claude/plans/cheeky-roaming-bunny.md):

  - Cell-type identity dominates the embedding: each of `n_cell_types` cell
    types has its own marker-gene block; within-type noise < between-type
    distance.
  - Each cytokine has a primary "program" (a sparse perturbation vector over
    response-pool genes) and a list of responder cell types.
  - Cascade edges A → B literally inject B's program into B-responder cells
    inside A-tubes, applied AFTER the primary perturbation and at a smaller
    magnitude β. Two-hop cascades (A → B → C) decay as β².
  - Similar-but-non-cascading pairs share ~70% of their program genes with
    jittered magnitudes; no cascade edge between them.
  - Isolated cytokines have no outgoing or incoming cascade.
  - PBS tubes get baseline + noise only — they ARE the resting state.

Ground truth is written to `cascade_ground_truth.json` and `cytokine_programs.json`
alongside the manifest, so analysis scripts can compare predicted asymmetry /
cascade edges against the construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Top-level parameters for the synthetic dataset."""

    n_cell_types: int = 12
    n_cytokines: int = 20  # excluding PBS
    n_genes: int = 512
    n_donors: int = 6
    n_pseudo_tubes: int = 8
    n_per_cell_type: int = 30

    # Gene-pool sizes (must sum ≤ n_genes; the remainder is background filler).
    n_housekeeping: int = 100
    n_markers_per_type: int = 25      # 12 × 25 = 300
    n_program_per_cytokine: int = 8   # 20 × 8 = 160

    # Distribution parameters.
    housekeeping_mu: float = 2.0
    housekeeping_sigma: float = 0.3
    marker_high_mu: float = 4.0
    marker_high_sigma: float = 0.5
    marker_low_mu: float = 0.2
    marker_low_sigma: float = 0.1
    response_baseline_mu: float = 1.0
    response_baseline_sigma: float = 0.3
    background_mu: float = 0.5
    background_sigma: float = 0.2

    program_magnitude_mu: float = 1.5
    program_magnitude_sigma: float = 0.3
    fraction_with_downreg: float = 0.20  # 20% of cytokines also down-regulate 1–2 genes
    n_responders_min: int = 2
    n_responders_max: int = 4
    effect_min: float = 0.6
    effect_max: float = 1.0

    cell_noise_sigma: float = 0.4
    donor_offset_sigma: float = 0.15

    apply_log1p: bool = True
    seed: int = 0


@dataclass
class CascadeGraph:
    """
    Ground-truth cascade structure.

    cascades : list of (src, dst, beta) — src tube gets a β-scaled dst program
               applied to dst's responder cells, AFTER src's primary effect.
    similar  : list of (a, b) cytokine pairs that share ~`similar_share_frac`
               of their program genes (jittered magnitudes), without any
               cascade edge.
    isolated : cytokines with no incoming OR outgoing cascade edges. They may
               still be similar-paired with another isolated cytokine.
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
    """The baked-in cascade structure documented in the plan."""
    return CascadeGraph(
        cascades=[
            ("cy1", "cy2", 0.40),                    # short, strong
            ("cy3", "cy4", 0.40),                    # 2-step part 1
            ("cy4", "cy5", 0.30),                    # 2-step part 2
            ("cy6", "cy7", 0.30),
            ("cy8", "cy9", 0.35), ("cy8", "cy10", 0.35),   # fan-out
            ("cy11", "cy12", 0.35), ("cy13", "cy12", 0.35), # fan-in
        ],
        similar=[("cy14", "cy15"), ("cy16", "cy17")],
        isolated=["cy18", "cy19", "cy20"],
    )


# ----------------------------------------------------------------------------
# Gene-pool layout
# ----------------------------------------------------------------------------


@dataclass
class GeneLayout:
    """Index ranges (relative to [0, n_genes)) for each gene pool."""

    housekeeping: np.ndarray
    markers_by_type: Dict[str, np.ndarray]
    program_by_cytokine: Dict[str, np.ndarray]
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

    program_by_cytokine: Dict[str, np.ndarray] = {}
    for cyt in cytokines:
        program_by_cytokine[cyt] = np.arange(used, used + cfg.n_program_per_cytokine)
        used += cfg.n_program_per_cytokine

    if used > cfg.n_genes:
        raise ValueError(
            f"Gene pools require {used} genes but n_genes={cfg.n_genes}. "
            "Reduce pool sizes or increase n_genes."
        )
    background = np.arange(used, cfg.n_genes)

    gene_names = [f"gene_{i:04d}" for i in range(cfg.n_genes)]
    return GeneLayout(
        housekeeping=housekeeping,
        markers_by_type=markers_by_type,
        program_by_cytokine=program_by_cytokine,
        background=background,
        gene_names=gene_names,
    )


# ----------------------------------------------------------------------------
# Cell-type baseline means
# ----------------------------------------------------------------------------


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
        # Housekeeping (shared across types).
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
        # Response-pool baseline (perturbed when cytokine acts).
        for cyt_genes in layout.program_by_cytokine.values():
            mu[cyt_genes] = rng.normal(
                cfg.response_baseline_mu, cfg.response_baseline_sigma,
                size=len(cyt_genes),
            )
        # Background filler.
        mu[layout.background] = rng.normal(
            cfg.background_mu, cfg.background_sigma, size=len(layout.background)
        )
        means[ct] = mu.astype(np.float32)
    return means


# ----------------------------------------------------------------------------
# Cytokine programs
# ----------------------------------------------------------------------------


@dataclass
class CytokinePrograms:
    """δ_C, responders(C), and per-(C,T) effect gain. PBS not stored — its program is zero."""

    delta: Dict[str, np.ndarray]                   # cyt -> R^n_genes (sparse, on program genes)
    responders: Dict[str, List[str]]               # cyt -> list[cell_type]
    effect: Dict[Tuple[str, str], float]           # (cyt, ct) -> gain (0 if not responder)

    def to_json(self, layout: GeneLayout) -> dict:
        out = {}
        for cyt, vec in self.delta.items():
            program_genes = layout.program_by_cytokine[cyt]
            # Also include any "borrowed" genes from a similar-pair partner
            # (i.e. all non-zero indices in vec).
            nz = np.flatnonzero(vec).tolist()
            out[cyt] = {
                "responders": self.responders[cyt],
                "program_gene_indices": nz,
                "program_gene_values": [float(vec[i]) for i in nz],
                "primary_program_indices": program_genes.tolist(),
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

    # 1. Build the primary program δ_C for each cytokine.
    for cyt in cytokines:
        vec = np.zeros(cfg.n_genes, dtype=np.float32)
        program_genes = layout.program_by_cytokine[cyt]
        # Up-regulated genes: positive Gaussian magnitudes.
        up_mags = rng.normal(
            cfg.program_magnitude_mu, cfg.program_magnitude_sigma,
            size=len(program_genes),
        ).astype(np.float32)
        up_mags = np.abs(up_mags)  # ensure positive
        vec[program_genes] = up_mags
        # Optional: 1–2 down-regulated genes (chosen from the same program block).
        if rng.random() < cfg.fraction_with_downreg and len(program_genes) >= 2:
            n_down = int(rng.integers(1, 3))
            n_down = min(n_down, len(program_genes))
            down_idx = rng.choice(program_genes, size=n_down, replace=False)
            vec[down_idx] = -np.abs(
                rng.normal(
                    cfg.program_magnitude_mu * 0.5, cfg.program_magnitude_sigma,
                    size=n_down,
                )
            ).astype(np.float32)
        delta[cyt] = vec

        # Responder cell types and gains.
        n_resp = int(rng.integers(cfg.n_responders_min, cfg.n_responders_max + 1))
        resp = sorted(rng.choice(cell_types, size=n_resp, replace=False).tolist())
        responders[cyt] = resp
        for ct in cell_types:
            if ct in resp:
                effect[(cyt, ct)] = float(
                    rng.uniform(cfg.effect_min, cfg.effect_max)
                )
            else:
                effect[(cyt, ct)] = 0.0

    # 2. Apply similarity-pair sharing: pair (a, b) shares ~similar_share_frac
    #    of their program genes, with jittered magnitudes (so they are similar
    #    but NOT identical; classifier can confuse them but no cascade exists).
    for a, b in graph.similar:
        program_a = layout.program_by_cytokine[a]
        program_b = layout.program_by_cytokine[b]
        n_share = int(round(graph.similar_share_frac * len(program_a)))
        if n_share <= 0:
            continue
        share_a_idx = rng.choice(program_a, size=n_share, replace=False)
        share_b_idx = rng.choice(program_b, size=n_share, replace=False)
        # b copies a's magnitudes onto its own genes (jittered)
        # AND additionally fires on a's shared genes (jittered).
        # This way both A-tubes and B-tubes show overlap on the same gene
        # indices — softmax confusion is symmetric.
        vec_a = delta[a]
        vec_b = delta[b]
        jitter = lambda v: v + rng.normal(0.0, 0.15, size=v.shape).astype(np.float32)
        # Make b also strongly express on a's shared genes (and vice versa).
        vec_b[share_a_idx] = jitter(vec_a[share_a_idx])
        vec_a[share_b_idx] = jitter(vec_b[share_b_idx])
        delta[a] = vec_a
        delta[b] = vec_b

    return CytokinePrograms(delta=delta, responders=responders, effect=effect)


# ----------------------------------------------------------------------------
# Tube sampling
# ----------------------------------------------------------------------------


def _sample_baseline_tube(
    cell_types: List[str],
    cell_type_means: Dict[str, np.ndarray],
    donor_offset: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[str]]:
    """
    PASS 0: sample baseline expression (no cytokine effect, no per-cell noise yet).

    Returns:
        X: (N, G) baseline matrix  — equals μ_T + donor_offset for each cell
        cell_type_per_row: list of length N
    """
    rows: List[np.ndarray] = []
    cell_type_per_row: List[str] = []
    for ct in cell_types:
        mu = cell_type_means[ct] + donor_offset  # (G,)
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
    """PASS 1: add the cytokine's own program to its responder cells (in place)."""
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
    PASS 2: for every cascade edge cytokine → child, add β · program(child) to
    child's responders. Recurse one extra hop (decay β · β'). Applied AFTER PASS 1.
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
    """Add per-cell noise, clip ≥ 0, optional log1p."""
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
        {
            "cell_type": cell_type_per_row,
            "donor": donor,
            "cytokine": cytokine,
        },
        index=[f"{donor}_{cytokine}_{tube_idx}_cell{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=gene_names)
    return ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)


# ----------------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------------


def generate_dataset(
    out_dir: str,
    cfg: Optional[SimConfig] = None,
    graph: Optional[CascadeGraph] = None,
) -> str:
    """
    Generate the full synthetic dataset on disk.

    Returns the manifest.json path.
    """
    cfg = cfg or SimConfig()
    graph = graph or default_cascade_graph()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    cell_types = [f"ct{i+1}" for i in range(cfg.n_cell_types)]
    cytokines = [f"cy{i+1}" for i in range(cfg.n_cytokines)]
    donors = [f"Donor{i+1}" for i in range(cfg.n_donors)]
    all_conditions = cytokines + ["PBS"]

    layout = _build_gene_layout(cfg, cell_types, cytokines)
    cell_type_means = _make_cell_type_means(cfg, cell_types, layout, rng)
    programs = _make_cytokine_programs(
        cfg, cytokines, cell_types, layout, graph, rng
    )

    # Per-donor offset shared across that donor's cells.
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

                manifest.append(
                    {
                        "path": str(tube_path),
                        "donor": donor,
                        "cytokine": cytokine,
                        "n_cells": int(adata.n_obs),
                        "cell_types_included": cell_types,
                        "tube_idx": tube_idx,
                    }
                )

    # Manifest + ground truth + HVG list (all 512 genes, no filtering).
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
        json.dump(
            {k: v for k, v in cfg.__dict__.items()},
            f, indent=2, default=str,
        )

    return str(manifest_path)


__all__ = [
    "SimConfig",
    "CascadeGraph",
    "default_cascade_graph",
    "generate_dataset",
]
