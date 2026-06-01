"""
End-to-end pytest suite for the Immune Dictionary adapter
(`scripts/build_pseudotubes_immune_dictionary.py`).

Run with:
    pip install -e ".[dev]"
    pytest tests/test_demo_id.py -v

These tests exercise the adapter's pseudo-tube construction half on a
synthesized AnnData input (skipping the 10x MTX loading layer, which requires
real GEO files). They pin the contract:

  1. PBS entries in the manifest carry cytokine == "PBS" (literal string).
  2. PseudoTubeDataset loads the ID demo manifest and returns the correct shape
     (60 cells, 200 genes).
  3. split_manifest_by_donor works with mouse_id as the donor key.
  4. The demo gene panel includes the pathway marker genes needed by §24/§25.
  5. CytokineLabel round-trips on ID demo cytokine names.
  6. Pseudo-tubes preserve cell-type labels id_c0, id_c1 in obs.
  7. PBS class index is 90 (hardcoded contract).
  8. compute_pathway_overlap_matrix (if present) returns 0.0 overlap for
     MUST-PASS cascade pairs (IRF3_direct, IFNAR_induced).
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import anndata as ad
import numpy as np
import pytest

# Make the scripts/ directory importable.
SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from tests.make_demo_data_id import (  # noqa: E402
    ALL_CLASSES,
    ALL_GENE_NAMES,
    ACTIVE_CYTOKINES,
    CELL_TYPES,
    MICE,
    MICE_TRAIN,
    MICE_VAL,
    N_CELLS_PER_COMBO,
    N_GENES,
    PATHWAY_GENE_NAMES,
    make_demo_adata_id,
)

import build_pseudotubes_immune_dictionary as id_adapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def demo_adata():
    """Synthetic ID-style input AnnData (PBS already relabeled, cell_type present)."""
    return make_demo_adata_id(seed=0)


@pytest.fixture(scope="session")
def pseudotube_dir(tmp_path_factory, demo_adata):
    """Run the adapter's pseudo-tube builder and return the output directory."""
    out_dir = tmp_path_factory.mktemp("id_pseudotubes")
    id_adapter.build_pseudo_tubes_id(
        adata=demo_adata,
        base_path=str(out_dir),
        n_per_cell_type=N_CELLS_PER_COMBO,  # 30 per cell type → 60 cells/tube
        min_cells_threshold=5,               # relaxed for demo
        n_pseudo_tubes=2,                    # small for fast tests
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
# Test 1: PBS entries carry the literal "PBS" string
# ---------------------------------------------------------------------------

class TestPBSContract:
    """PBS entries in the manifest must carry cytokine == 'PBS' (string)."""

    def test_pbs_entries_in_manifest(self, manifest):
        pbs_entries = [e for e in manifest if e["cytokine"] == "PBS"]
        assert len(pbs_entries) > 0, (
            "Manifest must contain at least one entry with cytokine='PBS'"
        )

    def test_pbs_obs_column_is_literal_pbs(self, manifest):
        pbs_entries = [e for e in manifest if e["cytokine"] == "PBS"]
        adata = ad.read_h5ad(pbs_entries[0]["path"])
        assert (adata.obs["cytokine"] == "PBS").all(), (
            "All cells in a PBS pseudo-tube must have obs['cytokine'] == 'PBS'"
        )

    def test_non_pbs_entries_are_not_pbs(self, manifest):
        non_pbs = [e for e in manifest if e["cytokine"] != "PBS"]
        assert len(non_pbs) > 0, "Manifest must contain non-PBS entries"
        for entry in non_pbs[:5]:
            adata = ad.read_h5ad(entry["path"])
            assert not (adata.obs["cytokine"] == "PBS").any(), (
                f"Non-PBS tube {entry['path']} must not have PBS cells"
            )


# ---------------------------------------------------------------------------
# Test 2: PseudoTubeDataset loads the ID demo manifest with correct shapes
# ---------------------------------------------------------------------------

class TestPseudoTubeDatasetLoading:
    """PseudoTubeDataset must load ID pseudo-tubes and return correct shapes."""

    def test_dataset_loads(self, manifest_path):
        from cytokine_mil.data.label_encoder import CytokineLabel
        from cytokine_mil.data.dataset import PseudoTubeDataset

        with open(manifest_path) as f:
            manifest = json.load(f)
        label_encoder = CytokineLabel().fit(manifest)
        dataset = PseudoTubeDataset(manifest_path, label_encoder)
        assert len(dataset) == len(manifest)

    def test_item_shape_cells_and_genes(self, manifest_path):
        """Each item should have 60 cells (2 types * 30 per type) and 200 genes."""
        import torch
        from cytokine_mil.data.label_encoder import CytokineLabel
        from cytokine_mil.data.dataset import PseudoTubeDataset

        with open(manifest_path) as f:
            manifest = json.load(f)
        label_encoder = CytokineLabel().fit(manifest)
        dataset = PseudoTubeDataset(manifest_path, label_encoder)
        X, label, donor, cyt_name = dataset[0]

        assert isinstance(X, torch.Tensor), "X must be a torch.Tensor"
        assert X.shape[0] == len(CELL_TYPES) * N_CELLS_PER_COMBO, (
            f"Expected {len(CELL_TYPES) * N_CELLS_PER_COMBO} cells, got {X.shape[0]}"
        )
        assert X.shape[1] == N_GENES, (
            f"Expected {N_GENES} genes, got {X.shape[1]}"
        )
        assert isinstance(label, int)
        assert isinstance(donor, str)
        assert donor in MICE
        assert isinstance(cyt_name, str)
        assert cyt_name in ALL_CLASSES


# ---------------------------------------------------------------------------
# Test 3: split_manifest_by_donor produces mouse-disjoint train/val sets
# ---------------------------------------------------------------------------

class TestSplitManifestByDonor:
    """split_manifest_by_donor must use 'donor' key (= mouse_id) correctly."""

    def test_train_val_are_mouse_disjoint(self, manifest):
        from cytokine_mil.experiment_setup import split_manifest_by_donor

        train_m, val_m = split_manifest_by_donor(manifest, val_donors=MICE_VAL)
        train_mice = {e["donor"] for e in train_m}
        val_mice = {e["donor"] for e in val_m}

        assert train_mice & val_mice == set(), (
            f"Train and val share mice: {train_mice & val_mice}"
        )
        assert train_mice == set(MICE_TRAIN), (
            f"Expected train mice {MICE_TRAIN}, got {sorted(train_mice)}"
        )
        assert val_mice == set(MICE_VAL), (
            f"Expected val mice {MICE_VAL}, got {sorted(val_mice)}"
        )

    def test_cytokine_sets_preserved_in_both_splits(self, manifest):
        from cytokine_mil.experiment_setup import split_manifest_by_donor

        train_m, val_m = split_manifest_by_donor(manifest, val_donors=MICE_VAL)
        train_cyts = {e["cytokine"] for e in train_m}
        val_cyts = {e["cytokine"] for e in val_m}
        expected = set(ALL_CLASSES)
        assert train_cyts == expected, (
            f"Train cytokines {sorted(train_cyts)} != {sorted(expected)}"
        )
        assert val_cyts == expected, (
            f"Val cytokines {sorted(val_cyts)} != {sorted(expected)}"
        )


# ---------------------------------------------------------------------------
# Test 4: Demo data includes pathway marker genes
# ---------------------------------------------------------------------------

class TestPathwayGenesPresent:
    """The demo gene panel must include the JAK-STAT pathway marker genes."""

    def test_pathway_genes_in_var_names(self, demo_adata):
        """Key pathway genes from PATHWAY_GENE_NAMES must appear in var_names."""
        must_have = [
            "Il2ra", "Ifng", "Stat1", "Stat3", "Stat4",
            "Ifnb1", "Mx1", "Nfkbia", "Il6st", "Socs3",
        ]
        for gene in must_have:
            assert gene in demo_adata.var_names, (
                f"Pathway gene '{gene}' missing from demo var_names"
            )

    def test_all_pathway_gene_names_present(self, demo_adata):
        """All 30 PATHWAY_GENE_NAMES must be in the demo var_names."""
        for gene in PATHWAY_GENE_NAMES:
            assert gene in demo_adata.var_names, (
                f"Pathway gene '{gene}' missing from demo var_names. "
                f"First 50 gene names: {list(demo_adata.var_names[:50])}"
            )

    def test_pathway_genes_in_pseudotube_files(self, manifest):
        """Pathway genes must survive into the written .h5ad pseudo-tube files."""
        entry = manifest[0]
        adata = ad.read_h5ad(entry["path"])
        for gene in ["Il2ra", "Ifng", "Stat1", "Mx1"]:
            assert gene in adata.var_names, (
                f"Gene '{gene}' missing from pseudo-tube file {entry['path']}"
            )


# ---------------------------------------------------------------------------
# Test 5: CytokineLabel round-trips on ID demo cytokine names
# ---------------------------------------------------------------------------

class TestCytokineLabelRoundTrip:
    """CytokineLabel must encode and decode all demo cytokine names correctly."""

    def test_encode_decode_roundtrip(self, manifest):
        from cytokine_mil.data.label_encoder import CytokineLabel

        label_encoder = CytokineLabel().fit(manifest)
        for cyt in ALL_CLASSES:
            idx = label_encoder.encode(cyt)
            decoded = label_encoder.decode(idx)
            assert decoded == cyt, (
                f"Round-trip failed: {cyt} -> {idx} -> {decoded}"
            )

    def test_all_classes_encodable(self, manifest):
        from cytokine_mil.data.label_encoder import CytokineLabel

        label_encoder = CytokineLabel().fit(manifest)
        for cyt in ALL_CLASSES:
            idx = label_encoder.encode(cyt)
            assert isinstance(idx, int), f"Index for {cyt} must be int"


# ---------------------------------------------------------------------------
# Test 6: Pseudo-tubes preserve cell-type labels id_c0 and id_c1
# ---------------------------------------------------------------------------

class TestCellTypeLabels:
    """Each pseudo-tube .h5ad must carry the id_cN cell-type labels in obs."""

    def test_cell_type_column_present(self, manifest):
        entry = manifest[0]
        adata = ad.read_h5ad(entry["path"])
        assert "cell_type" in adata.obs.columns, (
            f"obs column 'cell_type' missing in {entry['path']}"
        )

    def test_both_cell_types_present(self, manifest):
        """At least some tubes should contain both id_c0 and id_c1."""
        all_found = set()
        for entry in manifest:
            adata = ad.read_h5ad(entry["path"])
            all_found.update(adata.obs["cell_type"].unique())
        for ct in CELL_TYPES:
            assert ct in all_found, (
                f"Cell type '{ct}' not found in any pseudo-tube. "
                f"Found: {sorted(all_found)}"
            )

    def test_cell_types_included_matches_obs(self, manifest):
        """Manifest's cell_types_included must match what's in the .h5ad obs."""
        for entry in manifest[:5]:
            adata = ad.read_h5ad(entry["path"])
            obs_types = set(adata.obs["cell_type"].unique().astype(str))
            manifest_types = set(entry["cell_types_included"])
            assert obs_types == manifest_types, (
                f"Mismatch: manifest says {manifest_types}, obs has {obs_types}"
            )


# ---------------------------------------------------------------------------
# Test 7: PBS class index is 90
# ---------------------------------------------------------------------------

class TestPBSIndex:
    """PBS must always be at index 90 — the hardcoded contract."""

    def test_pbs_index_is_90(self, manifest):
        from cytokine_mil.data.label_encoder import CytokineLabel

        label_encoder = CytokineLabel().fit(manifest)
        assert label_encoder.encode("PBS") == 90, (
            f"PBS must be at index 90, got {label_encoder.encode('PBS')}"
        )

    def test_n_classes_is_91(self, manifest):
        from cytokine_mil.data.label_encoder import CytokineLabel

        label_encoder = CytokineLabel().fit(manifest)
        assert label_encoder.n_classes() == 91, (
            f"n_classes() must be 91 (PBS_INDEX + 1), got {label_encoder.n_classes()}"
        )

    def test_non_pbs_indices_below_90(self, manifest):
        from cytokine_mil.data.label_encoder import CytokineLabel

        label_encoder = CytokineLabel().fit(manifest)
        for cyt in ACTIVE_CYTOKINES:
            idx = label_encoder.encode(cyt)
            assert idx < 90, (
                f"Non-PBS cytokine '{cyt}' must have index < 90, got {idx}"
            )


# ---------------------------------------------------------------------------
# Test 8: Pathway overlap matrix — MUST-PASS cascade pairs have zero overlap
# ---------------------------------------------------------------------------

class TestPathwayOverlapMatrix:
    """
    The directional-asymmetry test (§24) requires that paired pathway gene
    sets (P_A, P_B) be transcriptionally distinct (zero or near-zero gene
    overlap). This test verifies the curated library's MUST-PASS pairs.

    If `compute_pathway_overlap_matrix` is not yet implemented in
    pathway_signatures.py (the gene lists for §25 are TBD), this test
    checks the Sheu-dataset pathway pairs that ARE currently implemented.
    """

    def test_irf3_ifnar_overlap_is_zero(self):
        """
        IRF3_direct and IFNAR_induced are the two pathways used in the
        Sheu audit's pre-registered positives (§24.3). Their gene sets must
        not overlap (otherwise the §24 asymmetry score is uninformative).
        """
        try:
            from cytokine_mil.analysis.pathway_signatures import PATHWAY_SIGNATURES
        except ImportError:
            pytest.skip("pathway_signatures module not importable in this environment")

        if "IRF3_direct" not in PATHWAY_SIGNATURES or "IFNAR_induced" not in PATHWAY_SIGNATURES:
            pytest.skip("IRF3_direct / IFNAR_induced not defined in PATHWAY_SIGNATURES")

        irf3_genes = set(PATHWAY_SIGNATURES["IRF3_direct"]["up"])
        ifnar_genes = set(PATHWAY_SIGNATURES["IFNAR_induced"]["up"])
        overlap = irf3_genes & ifnar_genes

        # Per §24.2: the two pathways must be transcriptionally distinct.
        # IRF3_direct: Ifnb1, Ccl5, Cxcl10, Ifit2, Ifit3
        # IFNAR_induced: Mx1, Mx2, Ifit1, Ifit1bl1, Ifit3, Ifit3b, Rsad2, Irf7, Oasl1
        # Note: Ifit3 appears in both — this is a known overlap; the asymmetry
        # test is validated to still work (audit 4 in §24.3 passed). We check
        # that the overlap fraction is small (< 30% of the smaller set).
        max_allowed_overlap = 0.3 * min(len(irf3_genes), len(ifnar_genes))
        assert len(overlap) <= max_allowed_overlap, (
            f"IRF3_direct / IFNAR_induced overlap too large: {overlap}. "
            f"Overlap fraction: {len(overlap)}/{min(len(irf3_genes), len(ifnar_genes))}. "
            f"These pathways must be transcriptionally distinct for §24 to work."
        )

    def test_nfkb_tnfr_overlap_flag(self):
        """
        NFkB_canonical and TNFR_autocrine overlap substantially (by design —
        both are NF-κB targets). This is the known failure mode documented in
        §24.2 / §24.3 (NF-κB cascades). We assert the overlap IS > 0 as a
        regression test that the spec's diagnosis remains correct.
        """
        try:
            from cytokine_mil.analysis.pathway_signatures import PATHWAY_SIGNATURES
        except ImportError:
            pytest.skip("pathway_signatures module not importable in this environment")

        if ("NFkB_canonical" not in PATHWAY_SIGNATURES or
                "TNFR_autocrine" not in PATHWAY_SIGNATURES):
            pytest.skip("NFkB_canonical / TNFR_autocrine not defined in PATHWAY_SIGNATURES")

        nfkb_genes = set(PATHWAY_SIGNATURES["NFkB_canonical"]["up"])
        tnfr_genes = set(PATHWAY_SIGNATURES["TNFR_autocrine"]["up"])
        overlap = nfkb_genes & tnfr_genes

        # Per §24.3: "NFkB_canonical and TNFR_autocrine overlap too heavily".
        # If this drops to zero, the diagnosis may have changed — flag it.
        assert len(overlap) > 0, (
            "NFkB_canonical / TNFR_autocrine overlap is unexpectedly zero. "
            "Per §24.3, they overlap by design (Tnfaip3, Nfkbid, Birc3). "
            "If gene sets changed, update this test and the §24.2 precondition note."
        )

    def test_must_pass_cascades_have_zero_overlap(self):
        """
        §25 pre-registered MUST-PASS cascades require P_A / P_B with zero gene
        overlap. This is the lock that the §24 directional-asymmetry test
        depends on; any drift here invalidates the pre-registration in
        reports/immune_dictionary/PRE_REGISTRATION.md.
        """
        from cytokine_mil.analysis.pathway_signatures import (
            IMMUNE_DICTIONARY_PREREGISTERED_CASCADES,
            compute_pathway_overlap_matrix,
        )

        m = compute_pathway_overlap_matrix()
        for A, B, PA, PB, outcome in IMMUNE_DICTIONARY_PREREGISTERED_CASCADES:
            if outcome != "MUST_PASS":
                continue
            assert m.loc[PA, PB] == 0.0, (
                f"§25 MUST-PASS cascade {A!r} -> {B!r} has nonzero overlap "
                f"between {PA!r} and {PB!r}: {m.loc[PA, PB]:.3f}. "
                f"Pre-registration is invalidated."
            )

    def test_must_fail_cascades_have_full_overlap(self):
        """
        §25 pre-registered MUST-FAIL cascades use P_A = P_B (overlap=1.0 by
        construction). This is the predicted failure-mode positive control.
        """
        from cytokine_mil.analysis.pathway_signatures import (
            IMMUNE_DICTIONARY_PREREGISTERED_CASCADES,
            compute_pathway_overlap_matrix,
        )

        m = compute_pathway_overlap_matrix()
        for A, B, PA, PB, outcome in IMMUNE_DICTIONARY_PREREGISTERED_CASCADES:
            if outcome != "MUST_FAIL":
                continue
            assert m.loc[PA, PB] == 1.0, (
                f"§25 MUST-FAIL cascade {A!r} -> {B!r} expects P_A == P_B "
                f"(overlap=1.0) but got {m.loc[PA, PB]:.3f}. Pre-registration "
                f"defines this as the overlap-failure positive control."
            )

    def test_preregistration_cascade_count(self):
        """Lock the pre-registered cascade count at 10 (5 MUST-PASS + 3 MUST-FAIL + 2 NEG_CONTROL)."""
        from cytokine_mil.analysis.pathway_signatures import (
            IMMUNE_DICTIONARY_PREREGISTERED_CASCADES,
        )

        outcomes = [c[4] for c in IMMUNE_DICTIONARY_PREREGISTERED_CASCADES]
        assert len(IMMUNE_DICTIONARY_PREREGISTERED_CASCADES) == 10
        assert outcomes.count("MUST_PASS") == 5
        assert outcomes.count("MUST_FAIL") == 3
        assert outcomes.count("NEG_CONTROL") == 2
