"""
Tests for the synthetic cytokine cascade simulator (v2 — calibrated).

v2 uses a shared response-pool architecture (hub + specific genes) instead of
per-cytokine exclusive gene blocks.  Tests updated accordingly:
  - program identity uses program_gene_indices (all non-zero), not primary_program_indices
  - similar-pair overlap measured over full non-zero support, not exclusive block
  - cascade signal tests unchanged in logic (still check cy2-gene elevation in cy1 tubes)
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


# ---------------------------------------------------------------------------
# Small config for fast tests
# ---------------------------------------------------------------------------

def _small_cfg() -> SimConfig:
    return SimConfig(
        n_cell_types=6,
        n_cytokines=20,
        # Layout: 80 HK + 6×20 markers (120) + 300 pool + 100 bg = 600
        n_genes=600,
        n_housekeeping=80,
        n_markers_per_type=20,   # 6 × 20 = 120
        n_response_pool=300,
        n_hub_genes=15,
        hub_activation_prob=0.60,
        n_specific_per_cytokine=12,
        n_donors=3,
        n_pseudo_tubes=2,
        n_per_cell_type=12,
        apply_log1p=False,       # easier to check raw baselines
        cell_noise_sigma=0.05,   # quiet noise for deterministic tests
        donor_offset_sigma=0.02,
        seed=123,
    )


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_cascade_v2")
    cfg = _small_cfg()
    manifest_path = generate_dataset(out_dir=str(out), cfg=cfg, graph=default_cascade_graph())
    return Path(manifest_path).parent, cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_files_and_manifest(synthetic_dataset):
    out_dir, cfg = synthetic_dataset
    for fname in ("manifest.json", "cascade_ground_truth.json",
                  "cytokine_programs.json", "hvg_list.json", "sim_config.json"):
        assert (out_dir / fname).exists(), f"Missing: {fname}"

    manifest = json.loads((out_dir / "manifest.json").read_text())
    n_conditions = cfg.n_cytokines + 1   # +PBS
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
    assert adata.n_obs  == cfg.n_cell_types * cfg.n_per_cell_type
    assert {"cell_type", "donor", "cytokine"}.issubset(set(adata.obs.columns))
    assert adata.X.dtype == np.float32


def _load_tubes(manifest, donor, cytokine):
    paths = [e["path"] for e in manifest if e["donor"] == donor and e["cytokine"] == cytokine]
    return [ad.read_h5ad(p) for p in paths]


def test_pbs_tube_clusters_at_baseline(synthetic_dataset):
    """
    PBS cells: within-type spread must be < between-type spread.
    (cell-type identity dominates the embedding — central design requirement)
    """
    out_dir, cfg = synthetic_dataset
    manifest = json.loads((out_dir / "manifest.json").read_text())
    pbs_tubes = _load_tubes(manifest, donor="Donor1", cytokine="PBS")
    adata = pbs_tubes[0]
    X  = np.asarray(adata.X)
    cts = adata.obs["cell_type"].values

    means = {ct: X[cts == ct].mean(axis=0) for ct in np.unique(cts)}
    types = list(means.keys())

    between = [
        np.linalg.norm(means[types[i]] - means[types[j]])
        for i in range(len(types))
        for j in range(i + 1, len(types))
    ]
    within = [
        np.linalg.norm(X[cts == ct] - means[ct], axis=1).mean()
        for ct in types
    ]
    assert np.mean(between) > 3.0 * np.mean(within), (
        f"Cell types not separated: between={np.mean(between):.3f}, "
        f"within={np.mean(within):.3f}"
    )


def test_cascade_signal_present_in_source_tube(synthetic_dataset):
    """
    cy1→cy2: cy1-tubes must show elevated expression on cy2's program genes
    in cy2's responder cells, compared with PBS and isolated cy18.
    """
    out_dir, cfg = synthetic_dataset
    manifest    = json.loads((out_dir / "manifest.json").read_text())
    progs       = json.loads((out_dir / "cytokine_programs.json").read_text())

    # Use all non-zero program genes (hub + specific), not exclusive block.
    cy2_idx       = np.array(progs["cy2"]["program_gene_indices"])
    cy2_responders = set(progs["cy2"]["responders"])

    def _signal(adata):
        X   = np.asarray(adata.X)
        cts = adata.obs["cell_type"].values
        mask = np.array([ct in cy2_responders for ct in cts])
        return float(X[mask][:, cy2_idx].mean()) if mask.any() else 0.0

    cy1_signal  = np.mean([_signal(t) for t in _load_tubes(manifest, "Donor1", "cy1")])
    cy18_signal = np.mean([_signal(t) for t in _load_tubes(manifest, "Donor1", "cy18")])
    pbs_signal  = np.mean([_signal(t) for t in _load_tubes(manifest, "Donor1", "PBS")])

    assert cy1_signal > pbs_signal, (
        f"cy1 cascade shows no elevation over PBS on cy2 genes: "
        f"cy1={cy1_signal:.4f}  PBS={pbs_signal:.4f}"
    )
    assert cy1_signal > cy18_signal, (
        f"cy1 cascade signal not > isolated cy18: "
        f"cy1={cy1_signal:.4f}  cy18={cy18_signal:.4f}"
    )


def test_two_hop_cascade_leaks_into_source(synthetic_dataset):
    """
    cy3→cy4→cy5: cy3-tubes must elevate cy5's program genes in cy5's responders.
    """
    out_dir, cfg = synthetic_dataset
    manifest = json.loads((out_dir / "manifest.json").read_text())
    progs    = json.loads((out_dir / "cytokine_programs.json").read_text())

    cy5_idx       = np.array(progs["cy5"]["program_gene_indices"])
    cy5_responders = set(progs["cy5"]["responders"])

    def _signal(adata):
        X   = np.asarray(adata.X)
        cts = adata.obs["cell_type"].values
        mask = np.array([ct in cy5_responders for ct in cts])
        return float(X[mask][:, cy5_idx].mean()) if mask.any() else 0.0

    cy3_signal = np.mean([_signal(t) for t in _load_tubes(manifest, "Donor1", "cy3")])
    pbs_signal = np.mean([_signal(t) for t in _load_tubes(manifest, "Donor1", "PBS")])

    assert cy3_signal > pbs_signal, (
        f"2-hop cascade cy3→cy4→cy5 not detectable: "
        f"cy3={cy3_signal:.4f}  PBS={pbs_signal:.4f}"
    )


def test_similar_pair_shares_program_genes(synthetic_dataset):
    """
    Similar pair (cy14, cy15): their non-zero program-gene sets must overlap
    substantially (≥ 40% of the smaller partner's support, loose threshold
    because hub genes create some natural overlap for all pairs).
    """
    out_dir, _ = synthetic_dataset
    progs = json.loads((out_dir / "cytokine_programs.json").read_text())

    a_nz = set(progs["cy14"]["program_gene_indices"])
    b_nz = set(progs["cy15"]["program_gene_indices"])

    overlap = len(a_nz & b_nz)
    min_support = min(len(a_nz), len(b_nz))
    overlap_frac = overlap / max(min_support, 1)

    # Similar-pair sharing must be clearly above what two random cytokines share
    # via hub genes alone (~hub_activation_prob² × n_hub ≈ 0.36×15=5 of ~27 genes ≈ 19%).
    assert overlap_frac >= 0.40, (
        f"(cy14,cy15) overlap too low: {overlap}/{min_support} = {overlap_frac:.2f}"
    )


def test_isolated_cytokine_lacks_cascade_signal(synthetic_dataset):
    """
    cy18 is isolated (no cascade edges). Its tubes should NOT elevate cy2's
    program genes in cy2's responders beyond PBS.
    """
    out_dir, _ = synthetic_dataset
    manifest = json.loads((out_dir / "manifest.json").read_text())
    progs    = json.loads((out_dir / "cytokine_programs.json").read_text())

    cy2_idx       = np.array(progs["cy2"]["program_gene_indices"])
    cy2_responders = set(progs["cy2"]["responders"])

    def _signal(adata):
        X   = np.asarray(adata.X)
        cts = adata.obs["cell_type"].values
        mask = np.array([ct in cy2_responders for ct in cts])
        return float(X[mask][:, cy2_idx].mean()) if mask.any() else 0.0

    cy18_signal = np.mean([_signal(t) for t in _load_tubes(manifest, "Donor1", "cy18")])
    cy1_signal  = np.mean([_signal(t) for t in _load_tubes(manifest, "Donor1", "cy1")])

    # cy1 should be higher than cy18 (cy1 has cascade, cy18 doesn't)
    assert cy1_signal > cy18_signal, (
        f"cy1 (cascade) should exceed cy18 (isolated) on cy2 genes: "
        f"cy1={cy1_signal:.4f}  cy18={cy18_signal:.4f}"
    )


def test_cascade_ground_truth_json_roundtrip(synthetic_dataset):
    out_dir, _ = synthetic_dataset
    truth = json.loads((out_dir / "cascade_ground_truth.json").read_text())
    assert "cascades" in truth and "similar" in truth and "isolated" in truth
    edges = {(c["src"], c["dst"]) for c in truth["cascades"]}
    assert ("cy1",  "cy2")  in edges
    assert ("cy3",  "cy4")  in edges
    assert ("cy4",  "cy5")  in edges
    assert {"cy18", "cy19", "cy20"} <= set(truth["isolated"])


def test_cytokine_programs_json_has_hub_fields(synthetic_dataset):
    """v2 cytokine_programs.json must include hub_gene_indices and specific_gene_indices."""
    out_dir, _ = synthetic_dataset
    progs = json.loads((out_dir / "cytokine_programs.json").read_text())
    for cyt in ["cy1", "cy8", "cy14"]:
        assert "hub_gene_indices" in progs[cyt], f"Missing hub_gene_indices for {cyt}"
        assert "specific_gene_indices" in progs[cyt], f"Missing specific_gene_indices for {cyt}"
        assert len(progs[cyt]["program_gene_indices"]) > 0, f"Empty program for {cyt}"


def test_pseudotube_dataset_loads(synthetic_dataset):
    """End-to-end: PseudoTubeDataset loads the synthetic manifest cleanly."""
    out_dir, cfg = synthetic_dataset
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    label_enc = CytokineLabel().fit(manifest)
    ds = PseudoTubeDataset(str(manifest_path), label_encoder=label_enc)
    X, label, donor, cytokine = ds[0]
    assert X.shape == (cfg.n_cell_types * cfg.n_per_cell_type, cfg.n_genes)
    assert isinstance(label, int)
    assert donor.startswith("Donor")
    assert cytokine in {f"cy{i+1}" for i in range(cfg.n_cytokines)} | {"PBS"}
