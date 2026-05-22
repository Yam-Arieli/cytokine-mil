"""
End-to-end pytest suite for the Sheu 2024 adapter (`scripts/build_pseudotubes_sheu2024.py`).

Run with:
    pip install -e ".[dev]"
    pytest tests/test_demo_sheu.py -v

These tests exercise the adapter's pseudo-tube construction half on a
synthesized AnnData input (skipping the BD Rhapsody CSV-parsing layer,
which requires real BD-format files). They pin the contract:

  1. The adapter's `relabel_to_pbs` step turns all (Unstim cells)
     and all (0hr cells) into cytokine == "PBS".
  2. The adapter's pseudo-tube builder produces a manifest readable
     by PseudoTubeDataset.
  3. CytokineLabel.fit on the manifest puts PBS at index 90.
  4. split_manifest_by_donor works with pseudo-donor names.
  5. The pseudo-tube .h5ad files carry the required obs columns
     (cell_type, donor, cytokine) so downstream code (Stage 1 / Stage 2
     / dynamics / latent_geometry) works without modification.
"""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

# Make the script importable as a module without installing it.
SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from tests.make_demo_data_sheu import (  # noqa: E402
    CELL_TYPES,
    EXPECTED_ACTIVE_CLASSES,
    N_CELLS_PER_COMBO,
    N_GENES,
    PSEUDO_DONORS,
    PSEUDO_DONORS_TRAIN,
    PSEUDO_DONORS_VAL,
    RAW_CYTOKINES,
    TIME_POINTS,
    make_demo_adata_sheu,
)

# Imports from the adapter script under test
import build_pseudotubes_sheu2024 as sheu_adapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def raw_adata():
    """The synthetic Sheu-style input AnnData (no PBS relabeling applied yet)."""
    return make_demo_adata_sheu(seed=0)


@pytest.fixture(scope="session")
def relabeled_adata(raw_adata):
    """After running the adapter's relabel step."""
    return sheu_adapter.relabel_to_pbs(raw_adata.copy())


@pytest.fixture(scope="session")
def pseudotube_dir(tmp_path_factory, relabeled_adata):
    """Run the adapter's pseudo-tube construction and return the output directory."""
    out_dir = tmp_path_factory.mktemp("sheu_pseudotubes")
    sheu_adapter.build_pseudo_tubes_sheu(
        adata=relabeled_adata,
        base_path=str(out_dir),
        n_per_cell_type=10,           # small for tests
        min_cells_threshold=5,         # small for tests
        n_pseudo_tubes=2,              # small for tests
        rng=np.random.default_rng(0),
    )
    return out_dir


@pytest.fixture(scope="session")
def manifest_path(pseudotube_dir):
    return str(Path(pseudotube_dir) / "manifest.json")


@pytest.fixture(scope="session")
def manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Relabeling contract
# ---------------------------------------------------------------------------

class TestRelabel:
    """The adapter must relabel Unstim cells and all 0hr cells to PBS."""

    def test_unstim_becomes_pbs(self, raw_adata, relabeled_adata):
        unstim_mask = raw_adata.obs["cytokine"] == "Unstim"
        assert unstim_mask.sum() > 0, "demo should contain Unstim cells"
        assert (relabeled_adata.obs.loc[unstim_mask.values, "cytokine"] == "PBS").all(), (
            "All Unstim cells must be relabeled to PBS"
        )

    def test_0hr_cells_become_pbs(self, raw_adata, relabeled_adata):
        zero_mask = raw_adata.obs["time_point"] == "0hr"
        assert zero_mask.sum() > 0
        assert (relabeled_adata.obs.loc[zero_mask.values, "cytokine"] == "PBS").all(), (
            "All 0hr cells must be relabeled to PBS regardless of original stimulus"
        )

    def test_3hr_stimulated_cells_keep_their_stimulus(self, raw_adata, relabeled_adata):
        for stim in ("LPS", "polyIC", "IFNb"):
            mask = (raw_adata.obs["cytokine"] == stim) & (raw_adata.obs["time_point"] == "3hr")
            assert mask.sum() > 0, f"demo should contain 3hr {stim} cells"
            assert (relabeled_adata.obs.loc[mask.values, "cytokine"] == stim).all(), (
                f"3hr {stim} cells must keep cytokine={stim} after relabel"
            )

    def test_active_class_set(self, relabeled_adata):
        seen = set(relabeled_adata.obs["cytokine"].unique())
        assert seen == set(EXPECTED_ACTIVE_CLASSES), (
            f"Expected active classes {EXPECTED_ACTIVE_CLASSES}, got {sorted(seen)}"
        )


# ---------------------------------------------------------------------------
# Manifest contract
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_exists(self, manifest_path):
        assert Path(manifest_path).exists()

    def test_manifest_nonempty(self, manifest):
        assert len(manifest) > 0

    def test_manifest_entry_keys(self, manifest):
        required = {"path", "donor", "cytokine", "n_cells", "cell_types_included", "tube_idx"}
        for entry in manifest:
            assert required.issubset(entry.keys()), (
                f"Missing keys in manifest entry: {required - set(entry.keys())}"
            )

    def test_manifest_uses_pseudo_donor_names(self, manifest):
        donors = {e["donor"] for e in manifest}
        # adapter prefixes pseudo-donor names directly (no "donor_" prefix added)
        assert donors == set(PSEUDO_DONORS), (
            f"Expected donors {PSEUDO_DONORS}, got {sorted(donors)}"
        )

    def test_manifest_cytokines_are_active_classes(self, manifest):
        cyts = {e["cytokine"] for e in manifest}
        assert cyts == set(EXPECTED_ACTIVE_CLASSES), (
            f"Expected cytokines {EXPECTED_ACTIVE_CLASSES}, got {sorted(cyts)}"
        )

    def test_every_donor_has_all_classes(self, manifest):
        """Each pseudo-donor must have all active classes (complete-donor filter)."""
        from collections import defaultdict
        by_donor = defaultdict(set)
        for e in manifest:
            by_donor[e["donor"]].add(e["cytokine"])
        for donor, cyts in by_donor.items():
            assert cyts == set(EXPECTED_ACTIVE_CLASSES), (
                f"Donor {donor} missing classes: {set(EXPECTED_ACTIVE_CLASSES) - cyts}"
            )

    def test_n_pseudo_tubes_per_donor_cytokine(self, manifest):
        """With n_pseudo_tubes=2, each (donor, cytokine) should have 2 tubes."""
        from collections import defaultdict
        counts = defaultdict(int)
        for e in manifest:
            counts[(e["donor"], e["cytokine"])] += 1
        for key, n in counts.items():
            assert n == 2, f"{key} has {n} tubes, expected 2"


# ---------------------------------------------------------------------------
# Pseudo-tube file contract
# ---------------------------------------------------------------------------

class TestPseudoTubeFiles:
    def test_files_exist(self, manifest):
        for entry in manifest:
            assert Path(entry["path"]).exists(), f"missing pseudo-tube file: {entry['path']}"

    def test_required_obs_columns(self, manifest):
        """Each pseudo-tube must carry cell_type, donor, cytokine in obs
        so downstream code (post-hoc analysis, latent_geometry) can use them."""
        entry = manifest[0]
        adata = ad.read_h5ad(entry["path"])
        for col in ("cell_type", "donor", "cytokine"):
            assert col in adata.obs.columns, (
                f"pseudo-tube missing obs column '{col}': has {list(adata.obs.columns)}"
            )

    def test_donor_column_matches_manifest(self, manifest):
        for entry in manifest[:5]:  # sample a few
            adata = ad.read_h5ad(entry["path"])
            assert (adata.obs["donor"] == entry["donor"]).all()

    def test_cytokine_column_matches_manifest(self, manifest):
        for entry in manifest[:5]:
            adata = ad.read_h5ad(entry["path"])
            assert (adata.obs["cytokine"] == entry["cytokine"]).all()

    def test_n_cells_matches_manifest(self, manifest):
        for entry in manifest[:5]:
            adata = ad.read_h5ad(entry["path"])
            assert adata.n_obs == entry["n_cells"]

    def test_gene_count_unchanged(self, manifest):
        """The adapter does not perform HVG selection for Sheu — keep all genes."""
        adata = ad.read_h5ad(manifest[0]["path"])
        assert adata.n_vars == N_GENES, (
            f"Expected {N_GENES} genes (no HVG filter), got {adata.n_vars}"
        )

    def test_dtype_is_float32(self, manifest):
        adata = ad.read_h5ad(manifest[0]["path"])
        # X may be dense or sparse; cast safely
        x = adata.X
        if hasattr(x, "toarray"):
            x = x.toarray()
        assert x.dtype == np.float32

    def test_cell_types_included_is_list_of_strings(self, manifest):
        for entry in manifest[:5]:
            cti = entry["cell_types_included"]
            assert isinstance(cti, list)
            assert all(isinstance(ct, str) for ct in cti)


# ---------------------------------------------------------------------------
# Downstream-pipeline compatibility
# ---------------------------------------------------------------------------

class TestPipelineCompat:
    def test_pseudotube_dataset_loads(self, manifest_path):
        from cytokine_mil.data.label_encoder import CytokineLabel
        from cytokine_mil.data.dataset import PseudoTubeDataset

        with open(manifest_path) as f:
            manifest = json.load(f)
        label_encoder = CytokineLabel().fit(manifest)
        dataset = PseudoTubeDataset(manifest_path, label_encoder)
        assert len(dataset) == len(manifest)

        import torch
        X, label, donor, cyt_name = dataset[0]
        assert isinstance(X, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(donor, str)
        assert isinstance(cyt_name, str)

    def test_pbs_index_is_90(self, manifest_path):
        from cytokine_mil.data.label_encoder import CytokineLabel
        with open(manifest_path) as f:
            manifest = json.load(f)
        label_encoder = CytokineLabel().fit(manifest)
        assert label_encoder.encode("PBS") == 90

    def test_n_classes_is_91(self, manifest_path):
        from cytokine_mil.data.label_encoder import CytokineLabel
        with open(manifest_path) as f:
            manifest = json.load(f)
        label_encoder = CytokineLabel().fit(manifest)
        # PBS_INDEX + 1 = 91 regardless of how many active cytokines
        assert label_encoder.n_classes() == 91

    def test_non_pbs_indices_are_below_90(self, manifest_path):
        from cytokine_mil.data.label_encoder import CytokineLabel
        with open(manifest_path) as f:
            manifest = json.load(f)
        label_encoder = CytokineLabel().fit(manifest)
        for cyt in [c for c in EXPECTED_ACTIVE_CLASSES if c != "PBS"]:
            assert label_encoder.encode(cyt) < 90

    def test_split_manifest_by_donor_with_pseudo_donor_names(self, manifest_path):
        from cytokine_mil.experiment_setup import split_manifest_by_donor
        with open(manifest_path) as f:
            manifest = json.load(f)
        train_m, val_m = split_manifest_by_donor(manifest, val_donors=PSEUDO_DONORS_VAL)
        assert {e["donor"] for e in train_m} == set(PSEUDO_DONORS_TRAIN)
        assert {e["donor"] for e in val_m} == set(PSEUDO_DONORS_VAL)
        # Cytokine sets preserved in both splits
        assert {e["cytokine"] for e in train_m} == set(EXPECTED_ACTIVE_CLASSES)
        assert {e["cytokine"] for e in val_m} == set(EXPECTED_ACTIVE_CLASSES)


# ---------------------------------------------------------------------------
# PBS-RC compatibility (relabeling-to-"PBS" must let pbs_rc work unchanged)
# ---------------------------------------------------------------------------

class TestPbsRcCompat:
    """The adapter relabels resting cells to literal `"PBS"` so the existing
    PBS-RC code in `cytokine_mil/analysis/pbs_rc.py` (which hard-checks
    `cytokine == "PBS"`) works without modification."""

    def test_pbs_cells_present_in_some_tube(self, manifest):
        pbs_entries = [e for e in manifest if e["cytokine"] == "PBS"]
        assert len(pbs_entries) > 0, (
            "Adapter must produce at least one pseudo-tube with cytokine='PBS' "
            "so PBS-RC code (analysis/pbs_rc.py:59) can compute resting centroids."
        )

    def test_pbs_tube_obs_cytokine_is_pbs(self, manifest):
        pbs_entries = [e for e in manifest if e["cytokine"] == "PBS"]
        adata = ad.read_h5ad(pbs_entries[0]["path"])
        # The internal obs column must also be the literal "PBS" string,
        # because pbs_rc.precompute_transform_means walks records by
        # decoded label name (`label_encoder._idx_to_label[entry["label"]]`).
        assert (adata.obs["cytokine"] == "PBS").all()
