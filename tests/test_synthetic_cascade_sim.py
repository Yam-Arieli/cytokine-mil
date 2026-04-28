"""
Tests for the synthetic cytokine cascade simulator.

Uses a small SimConfig so the suite runs quickly. Verifies:
  - shapes / dtypes / on-disk layout
  - PBS tube means ≈ baseline μ_T per cell type
  - cascade tube cy1 elevates cy2-program genes in cy2-responder cells more
    than an isolated tube cy18 does (cascade signal exists by construction)
  - similar pair (cy14, cy15) shares ≥ similar_share_frac of program magnitude
  - 2-step cascade cy3 → cy4 → cy5 also leaks into cy3-tubes (decayed β²)
  - manifest loads cleanly with PseudoTubeDataset
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from cytokine_mil.data.synthetic_cascade_sim import (
    CascadeGraph,
    SimConfig,
    default_cascade_graph,
    generate_dataset,
)
from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel


# Smaller config for fast tests.
def _small_cfg() -> SimConfig:
    return SimConfig(
        n_cell_types=6,
        n_cytokines=20,
        n_genes=400,           # 100 + 6*25 + 20*8 = 410 -> need ≥ 410
        n_housekeeping=80,
        n_markers_per_type=20, # 6*20 = 120
        n_program_per_cytokine=8,  # 20*8 = 160 -> total 360, fits in 400
        n_donors=3,
        n_pseudo_tubes=2,
        n_per_cell_type=12,
        apply_log1p=False,     # easier to compare baselines
        cell_noise_sigma=0.1,  # quieter for stable tests
        donor_offset_sigma=0.05,
        seed=123,
    )


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_cascade")
    cfg = _small_cfg()
    manifest_path = generate_dataset(out_dir=str(out), cfg=cfg, graph=default_cascade_graph())
    return Path(manifest_path).parent, cfg


def test_files_and_manifest(synthetic_dataset):
    out_dir, cfg = synthetic_dataset
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "cascade_ground_truth.json").exists()
    assert (out_dir / "cytokine_programs.json").exists()
    assert (out_dir / "hvg_list.json").exists()
    assert (out_dir / "sim_config.json").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    n_conditions = cfg.n_cytokines + 1  # +PBS
    expected = cfg.n_donors * n_conditions * cfg.n_pseudo_tubes
    assert len(manifest) == expected

    entry = manifest[0]
    for key in ("path", "donor", "cytokine", "n_cells", "cell_types_included", "tube_idx"):
        assert key in entry
    assert entry["n_cells"] == cfg.n_cell_types * cfg.n_per_cell_type


def test_h5ad_shapes_and_obs(synthetic_dataset):
    out_dir, cfg = synthetic_dataset
    manifest = json.loads((out_dir / "manifest.json").read_text())
    adata = ad.read_h5ad(manifest[0]["path"])
    assert adata.n_vars == cfg.n_genes
    assert adata.n_obs == cfg.n_cell_types * cfg.n_per_cell_type
    assert {"cell_type", "donor", "cytokine"}.issubset(set(adata.obs.columns))
    assert adata.X.dtype == np.float32


def _load_tubes(manifest, donor, cytokine):
    paths = [e["path"] for e in manifest if e["donor"] == donor and e["cytokine"] == cytokine]
    return [ad.read_h5ad(p) for p in paths]


def test_pbs_tube_clusters_at_baseline(synthetic_dataset):
    """
    PBS cells of cell type T should sit close to the cell-type baseline μ_T.
    Equivalently: within a PBS tube, cells of the same type cluster much more
    tightly than cells of different types.
    """
    out_dir, cfg = synthetic_dataset
    manifest = json.loads((out_dir / "manifest.json").read_text())
    pbs_tubes = _load_tubes(manifest, donor="Donor1", cytokine="PBS")
    assert len(pbs_tubes) >= 1
    adata = pbs_tubes[0]
    X = np.asarray(adata.X)
    cts = adata.obs["cell_type"].values

    # Mean per cell type.
    means = {}
    for ct in np.unique(cts):
        means[ct] = X[cts == ct].mean(axis=0)

    # Within-type spread should be << between-type spread.
    types = list(means.keys())
    between = []
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            between.append(np.linalg.norm(means[types[i]] - means[types[j]]))
    within = []
    for ct in types:
        rows = X[cts == ct]
        within.append(np.linalg.norm(rows - means[ct], axis=1).mean())

    assert np.mean(between) > 3.0 * np.mean(within), (
        f"Cell-type clusters not separated: between={np.mean(between):.3f}, "
        f"within={np.mean(within):.3f}"
    )


def test_cascade_signal_present_in_source_tube(synthetic_dataset):
    """
    cy1 → cy2 cascade: cy1-tubes should show elevated expression on cy2's
    program genes IN cy2's responder cells, beyond what an isolated cytokine
    (cy18) shows on cy2's program genes.
    """
    out_dir, cfg = synthetic_dataset
    manifest = json.loads((out_dir / "manifest.json").read_text())
    programs_meta = json.loads((out_dir / "cytokine_programs.json").read_text())

    cy2_program_idx = np.array(programs_meta["cy2"]["primary_program_indices"])
    cy2_responders = set(programs_meta["cy2"]["responders"])

    def _signal_on_cy2_genes_in_cy2_responders(adata):
        X = np.asarray(adata.X)
        cts = adata.obs["cell_type"].values
        mask = np.array([ct in cy2_responders for ct in cts])
        if not mask.any():
            return 0.0
        return float(X[mask][:, cy2_program_idx].mean())

    cy1_tubes = _load_tubes(manifest, donor="Donor1", cytokine="cy1")
    cy18_tubes = _load_tubes(manifest, donor="Donor1", cytokine="cy18")
    pbs_tubes = _load_tubes(manifest, donor="Donor1", cytokine="PBS")

    cy1_signal = np.mean([_signal_on_cy2_genes_in_cy2_responders(t) for t in cy1_tubes])
    cy18_signal = np.mean([_signal_on_cy2_genes_in_cy2_responders(t) for t in cy18_tubes])
    pbs_signal = np.mean([_signal_on_cy2_genes_in_cy2_responders(t) for t in pbs_tubes])

    assert cy1_signal > pbs_signal, (
        f"cy1 (cascade source) shows no elevation on cy2 program genes "
        f"vs PBS: cy1={cy1_signal:.3f}, PBS={pbs_signal:.3f}"
    )
    assert cy1_signal > cy18_signal, (
        f"cy1 cascade signal not greater than isolated cy18: "
        f"cy1={cy1_signal:.3f}, cy18={cy18_signal:.3f}"
    )


def test_two_hop_cascade_leaks_into_source(synthetic_dataset):
    """
    cy3 → cy4 → cy5: cy3-tubes should elevate cy5's program genes in cy5's
    responders (decayed by β·β'), more than PBS does.
    """
    out_dir, cfg = synthetic_dataset
    manifest = json.loads((out_dir / "manifest.json").read_text())
    programs_meta = json.loads((out_dir / "cytokine_programs.json").read_text())

    cy5_program_idx = np.array(programs_meta["cy5"]["primary_program_indices"])
    cy5_responders = set(programs_meta["cy5"]["responders"])

    def _signal(adata):
        X = np.asarray(adata.X)
        cts = adata.obs["cell_type"].values
        mask = np.array([ct in cy5_responders for ct in cts])
        if not mask.any():
            return 0.0
        return float(X[mask][:, cy5_program_idx].mean())

    cy3_tubes = _load_tubes(manifest, donor="Donor1", cytokine="cy3")
    pbs_tubes = _load_tubes(manifest, donor="Donor1", cytokine="PBS")

    cy3_signal = np.mean([_signal(t) for t in cy3_tubes])
    pbs_signal = np.mean([_signal(t) for t in pbs_tubes])
    assert cy3_signal > pbs_signal, (
        f"2-hop cascade cy3→cy4→cy5 not detectable: "
        f"cy3={cy3_signal:.3f}, PBS={pbs_signal:.3f}"
    )


def test_similar_pair_shares_program_genes(synthetic_dataset):
    """
    Similar pair (cy14, cy15): each cytokine's δ vector should be non-zero on
    a substantial fraction of the partner's program-gene block (≥ 50%, with
    similar_share_frac=0.7 and jitter; threshold loose to avoid flakiness).
    """
    out_dir, cfg = synthetic_dataset
    programs_meta = json.loads((out_dir / "cytokine_programs.json").read_text())

    a, b = "cy14", "cy15"
    a_program = set(programs_meta[a]["primary_program_indices"])
    b_program = set(programs_meta[b]["primary_program_indices"])
    a_nz = set(programs_meta[a]["program_gene_indices"])
    b_nz = set(programs_meta[b]["program_gene_indices"])

    # cy14 should now be active on a substantial subset of cy15's primary genes.
    a_on_b = len(a_nz & b_program) / len(b_program)
    b_on_a = len(b_nz & a_program) / len(a_program)
    assert a_on_b >= 0.5, f"cy14 active on only {a_on_b:.2f} of cy15's program genes"
    assert b_on_a >= 0.5, f"cy15 active on only {b_on_a:.2f} of cy14's program genes"


def test_cascade_ground_truth_json_roundtrip(synthetic_dataset):
    out_dir, _ = synthetic_dataset
    truth = json.loads((out_dir / "cascade_ground_truth.json").read_text())
    assert "cascades" in truth and "similar" in truth and "isolated" in truth
    edges = {(c["src"], c["dst"]) for c in truth["cascades"]}
    assert ("cy1", "cy2") in edges
    assert ("cy3", "cy4") in edges
    assert ("cy4", "cy5") in edges
    assert {"cy18", "cy19", "cy20"} <= set(truth["isolated"])


def test_pseudotube_dataset_loads(synthetic_dataset):
    """End-to-end: PseudoTubeDataset loads the synthetic manifest cleanly."""
    out_dir, cfg = synthetic_dataset
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    encoder = CytokineLabel().fit(manifest)
    ds = PseudoTubeDataset(str(manifest_path), label_encoder=encoder)
    X, label, donor, cytokine = ds[0]
    assert X.shape == (cfg.n_cell_types * cfg.n_per_cell_type, cfg.n_genes)
    assert isinstance(label, int)
    assert donor.startswith("Donor")
    assert cytokine in {f"cy{i+1}" for i in range(cfg.n_cytokines)} | {"PBS"}
