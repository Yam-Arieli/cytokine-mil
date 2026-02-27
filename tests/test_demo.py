"""
End-to-end pytest suite on simulated demo data.

Run with:
    pip install -e ".[dev]"
    pytest tests/test_demo.py -v

Tests:
  - Label encoder roundtrip and PBS index
  - Dataset loading and item shapes
  - Model forward pass shapes (all four components)
  - Encoder freeze / unfreeze
  - Attention weights sum to 1
  - Stage 1 encoder pre-training runs without error
  - Stage 2 MIL training runs and returns dynamics dict
  - Learnability ranking produces correct output
  - Instance confidence grouping by cell type
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.make_demo_data import (
    CELL_TYPES,
    CYTOKINES,
    N_CELLS_PER_TYPE,
    N_GENES,
    N_PSEUDO_TUBES,
    DONORS,
    make_demo_data,
)

# n_classes is always PBS_INDEX + 1 = 91 so PBS label (90) is valid
N_CLASSES = 91


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    """Create demo data once for the whole test session."""
    base = tmp_path_factory.mktemp("demo")
    make_demo_data(str(base), seed=0)
    return str(base)


@pytest.fixture(scope="session")
def manifest_path(demo_dir):
    return str(Path(demo_dir) / "manifest.json")


@pytest.fixture(scope="session")
def label_encoder(manifest_path):
    from cytokine_mil.data.label_encoder import CytokineLabel
    with open(manifest_path) as f:
        manifest = json.load(f)
    return CytokineLabel().fit(manifest)


@pytest.fixture(scope="session")
def dataset(manifest_path, label_encoder):
    from cytokine_mil.data.dataset import PseudoTubeDataset
    return PseudoTubeDataset(manifest_path, label_encoder)


@pytest.fixture(scope="session")
def cell_dataset(manifest_path):
    from cytokine_mil.data.dataset import CellDataset
    return CellDataset(manifest_path)


@pytest.fixture(scope="session")
def encoder(cell_dataset):
    from cytokine_mil.models.instance_encoder import InstanceEncoder
    return InstanceEncoder(
        input_dim=N_GENES, embed_dim=128, n_cell_types=cell_dataset.n_cell_types()
    )


@pytest.fixture(scope="session")
def mil_model(encoder):
    """Shape-test model (not used for training tests to avoid state pollution)."""
    from cytokine_mil.models.attention import AttentionModule
    from cytokine_mil.models.bag_classifier import BagClassifier
    from cytokine_mil.models.cytokine_abmil import CytokineABMIL
    attention = AttentionModule(embed_dim=128, attention_hidden_dim=64)
    classifier = BagClassifier(embed_dim=128, n_classes=N_CLASSES)
    return CytokineABMIL(encoder, attention, classifier, encoder_frozen=True)


def _make_fresh_mil_model(n_genes: int = N_GENES) -> "CytokineABMIL":
    """Create a fully independent MIL model for training tests."""
    from cytokine_mil.models.instance_encoder import InstanceEncoder
    from cytokine_mil.models.attention import AttentionModule
    from cytokine_mil.models.bag_classifier import BagClassifier
    from cytokine_mil.models.cytokine_abmil import CytokineABMIL
    enc = InstanceEncoder(input_dim=n_genes, embed_dim=128, n_cell_types=len(CELL_TYPES))
    attn = AttentionModule(embed_dim=128, attention_hidden_dim=64)
    clf = BagClassifier(embed_dim=128, n_classes=N_CLASSES)
    return CytokineABMIL(enc, attn, clf, encoder_frozen=True)


# ---------------------------------------------------------------------------
# Label encoder tests
# ---------------------------------------------------------------------------

class TestLabelEncoder:
    def test_pbs_index(self, label_encoder):
        assert label_encoder.encode("PBS") == 90

    def test_roundtrip(self, label_encoder):
        for cyt in CYTOKINES:
            idx = label_encoder.encode(cyt)
            assert label_encoder.decode(idx) == cyt

    def test_non_pbs_indices_below_90(self, label_encoder):
        for cyt in CYTOKINES:
            assert label_encoder.encode(cyt) < 90

    def test_save_load(self, label_encoder, tmp_path):
        from cytokine_mil.data.label_encoder import CytokineLabel
        path = str(tmp_path / "label_encoder.json")
        label_encoder.save(path)
        loaded = CytokineLabel.load(path)
        for cyt in CYTOKINES + ["PBS"]:
            assert loaded.encode(cyt) == label_encoder.encode(cyt)

    def test_n_classes_is_91(self, label_encoder):
        # n_classes() returns PBS_INDEX + 1 = 91 for any dataset size
        assert label_encoder.n_classes() == 91


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestPseudoTubeDataset:
    def test_len(self, dataset):
        expected = len(DONORS) * (len(CYTOKINES) + 1) * N_PSEUDO_TUBES
        assert len(dataset) == expected

    def test_item_types(self, dataset):
        X, label, donor, cyt_name = dataset[0]
        assert isinstance(X, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(donor, str)
        assert isinstance(cyt_name, str)

    def test_item_shape(self, dataset):
        X, label, donor, cyt_name = dataset[0]
        n_cells = len(CELL_TYPES) * N_CELLS_PER_TYPE
        assert X.shape == (n_cells, N_GENES), f"Expected ({n_cells}, {N_GENES}), got {X.shape}"

    def test_label_in_range(self, dataset, label_encoder):
        X, label, donor, cyt_name = dataset[0]
        assert 0 <= label < label_encoder.n_classes()


class TestCellDataset:
    def test_n_cell_types(self, cell_dataset):
        assert cell_dataset.n_cell_types() == len(CELL_TYPES)

    def test_item_shape(self, cell_dataset):
        x, label = cell_dataset[0]
        assert x.shape == (N_GENES,)
        assert isinstance(label, int)

    def test_label_in_range(self, cell_dataset):
        _, label = cell_dataset[0]
        assert 0 <= label < cell_dataset.n_cell_types()

    def test_total_cells(self, cell_dataset):
        expected = (
            len(DONORS) * (len(CYTOKINES) + 1) * N_PSEUDO_TUBES
            * len(CELL_TYPES) * N_CELLS_PER_TYPE
        )
        assert len(cell_dataset) == expected


# ---------------------------------------------------------------------------
# Model forward pass tests
# ---------------------------------------------------------------------------

class TestInstanceEncoder:
    def test_output_shape(self, encoder):
        x = torch.randn(100, N_GENES)
        h = encoder(x)
        assert h.shape == (100, 128)

    def test_cell_type_head_shape(self, encoder, cell_dataset):
        x = torch.randn(50, N_GENES)
        h = encoder(x)
        logits = encoder.cell_type_head(h)
        assert logits.shape == (50, cell_dataset.n_cell_types())

    def test_output_dtype(self, encoder):
        x = torch.randn(10, N_GENES)
        h = encoder(x)
        assert h.dtype == torch.float32


class TestAttentionModule:
    def test_output_shape(self):
        from cytokine_mil.models.attention import AttentionModule
        attn = AttentionModule(embed_dim=128, attention_hidden_dim=64)
        H = torch.randn(100, 128)
        a = attn(H)
        assert a.shape == (100,)

    def test_weights_sum_to_one(self):
        from cytokine_mil.models.attention import AttentionModule
        attn = AttentionModule(embed_dim=128, attention_hidden_dim=64)
        H = torch.randn(100, 128)
        a = attn(H)
        assert abs(a.sum().item() - 1.0) < 1e-5

    def test_weights_non_negative(self):
        from cytokine_mil.models.attention import AttentionModule
        attn = AttentionModule(embed_dim=128, attention_hidden_dim=64)
        a = attn(torch.randn(50, 128))
        assert (a >= 0).all()


class TestBagClassifier:
    def test_output_shape(self):
        from cytokine_mil.models.bag_classifier import BagClassifier
        clf = BagClassifier(embed_dim=128, n_classes=N_CLASSES)
        z = torch.randn(128)
        y = clf(z)
        assert y.shape == (N_CLASSES,)


class TestCytokineABMIL:
    def test_forward_shapes(self, mil_model):
        X = torch.randn(100, N_GENES)
        y_hat, a, H = mil_model(X)
        assert y_hat.shape == (N_CLASSES,)
        assert a.shape == (100,)
        assert H.shape == (100, 128)

    def test_attention_sums_to_one(self, mil_model):
        X = torch.randn(100, N_GENES)
        _, a, _ = mil_model(X)
        assert abs(a.sum().item() - 1.0) < 1e-5

    def test_encoder_frozen(self, mil_model):
        for param in mil_model.encoder.parameters():
            assert not param.requires_grad

    def test_unfreeze_encoder(self, encoder):
        from cytokine_mil.models.attention import AttentionModule
        from cytokine_mil.models.bag_classifier import BagClassifier
        from cytokine_mil.models.cytokine_abmil import CytokineABMIL
        attn = AttentionModule()
        clf = BagClassifier(n_classes=N_CLASSES)
        model = CytokineABMIL(encoder, attn, clf, encoder_frozen=True)
        model.unfreeze_encoder()
        for param in model.encoder.parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# Training tests (fresh models to avoid state leakage between tests)
# ---------------------------------------------------------------------------

class TestTrainEncoder:
    def test_stage1_runs(self, encoder, cell_dataset):
        from torch.utils.data import DataLoader
        from cytokine_mil.training.train_encoder import train_encoder
        loader = DataLoader(cell_dataset, batch_size=64, shuffle=True)
        trained = train_encoder(
            encoder, loader, n_epochs=1, lr=0.01, momentum=0.9, verbose=False
        )
        assert trained is encoder  # in-place, same object


class TestTrainMIL:
    def test_stage2_runs_and_returns_dynamics(self, dataset):
        from cytokine_mil.training.train_mil import train_mil
        model = _make_fresh_mil_model()
        dynamics = train_mil(
            model, dataset, n_epochs=2, lr=0.01, momentum=0.9,
            log_every_n_epochs=1, verbose=False, seed=42,
        )
        assert "logged_epochs" in dynamics
        assert "records" in dynamics
        assert len(dynamics["records"]) == len(dataset)

    def test_dynamics_contains_trajectory(self, dataset):
        from cytokine_mil.training.train_mil import train_mil
        model = _make_fresh_mil_model()
        dynamics = train_mil(
            model, dataset, n_epochs=2, log_every_n_epochs=2, verbose=False, seed=0,
        )
        rec = dynamics["records"][0]
        assert "p_correct_trajectory" in rec
        assert "entropy_trajectory" in rec
        assert "instance_confidence_trajectory" in rec
        assert len(rec["p_correct_trajectory"]) == 1  # logged once at epoch 2
        assert len(rec["entropy_trajectory"]) == 1
        assert rec["instance_confidence_trajectory"] is not None
        assert rec["instance_confidence_trajectory"].shape[1] == 1  # n_logged_epochs=1
        assert "confusion_entropy_trajectory" in dynamics

    def test_p_correct_in_valid_range(self, dataset):
        from cytokine_mil.training.train_mil import train_mil
        model = _make_fresh_mil_model()
        dynamics = train_mil(model, dataset, n_epochs=2, verbose=False, seed=1)
        for rec in dynamics["records"]:
            for p in rec["p_correct_trajectory"]:
                assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

class TestDynamicsAnalysis:
    @pytest.fixture(scope="class")
    def dynamics(self, dataset):
        from cytokine_mil.training.train_mil import train_mil
        model = _make_fresh_mil_model()
        return train_mil(model, dataset, n_epochs=2, verbose=False, seed=42)

    def test_learnability_ranking(self, dynamics):
        from cytokine_mil.analysis.dynamics import (
            aggregate_to_donor_level,
            rank_cytokines_by_learnability,
        )
        donor_traj = aggregate_to_donor_level(dynamics["records"])
        result = rank_cytokines_by_learnability(donor_traj, exclude=["PBS"])
        assert "ranking" in result
        assert "metric_description" in result
        ranking = result["ranking"]
        assert len(ranking) == len(CYTOKINES)
        names = [r[0] for r in ranking]
        assert set(names) == set(CYTOKINES)

    def test_aggregate_to_donor_level_shape(self, dynamics):
        from cytokine_mil.analysis.dynamics import aggregate_to_donor_level
        donor_traj = aggregate_to_donor_level(dynamics["records"])
        for cytokine, donors in donor_traj.items():
            for donor, traj in donors.items():
                assert isinstance(traj, np.ndarray)
                assert traj.ndim == 1

    def test_instance_confidence_grouping(self, dynamics):
        from cytokine_mil.analysis.dynamics import group_confidence_by_cell_type
        rec = next(
            r for r in dynamics["records"]
            if r["instance_confidence_trajectory"] is not None
        )
        # instance_confidence_trajectory is (n_cells, n_logged_epochs);
        # mean across epochs to get a (n_cells,) summary for grouping.
        conf = rec["instance_confidence_trajectory"].mean(axis=1)
        n_cells = len(CELL_TYPES) * N_CELLS_PER_TYPE
        fake_labels = np.array(CELL_TYPES * N_CELLS_PER_TYPE)
        grouped = group_confidence_by_cell_type(conf, fake_labels)
        assert set(grouped.keys()) == set(CELL_TYPES)
        for ct, vals in grouped.items():
            assert len(vals) == N_CELLS_PER_TYPE

    def test_instance_confidence_trajectory_shape(self, dynamics):
        rec = next(
            r for r in dynamics["records"]
            if r["instance_confidence_trajectory"] is not None
        )
        traj = rec["instance_confidence_trajectory"]
        n_cells = len(CELL_TYPES) * N_CELLS_PER_TYPE
        n_logged = len(dynamics["logged_epochs"])
        assert traj.ndim == 2, f"Expected 2-D (n_cells, n_logged_epochs), got shape {traj.shape}"
        assert traj.shape == (n_cells, n_logged), (
            f"Expected ({n_cells}, {n_logged}), got {traj.shape}"
        )

    def test_confusion_entropy_trajectory_present(self, dynamics):
        assert "confusion_entropy_trajectory" in dynamics
        cet = dynamics["confusion_entropy_trajectory"]
        # Should have one entry per cytokine (including PBS)
        assert len(cet) > 0
        for cyt, traj in cet.items():
            assert isinstance(cyt, str)
            assert isinstance(traj, np.ndarray)
            assert traj.ndim == 1
            assert len(traj) == len(dynamics["logged_epochs"])

    def test_confusion_entropy_values_non_negative(self, dynamics):
        for cyt, traj in dynamics["confusion_entropy_trajectory"].items():
            assert (traj >= 0).all(), f"Negative confusion entropy for {cyt}"

    def test_entropy_non_negative(self):
        from cytokine_mil.analysis.dynamics import compute_entropy
        a = torch.softmax(torch.randn(100), dim=0)
        assert compute_entropy(a) >= 0.0

    def test_instance_confidence_formula(self):
        from cytokine_mil.analysis.dynamics import compute_instance_confidence
        a = torch.tensor([0.1, 0.4, 0.5])
        C = compute_instance_confidence(a, 0.8)
        expected = torch.tensor([0.08, 0.32, 0.40])
        assert torch.allclose(C, expected, atol=1e-5)


class TestExperimentSetup:
    """Minimal tests for cytokine_mil.experiment_setup."""

    def test_build_stage1_manifest_one_per_cytokine(self, manifest_path):
        from cytokine_mil.experiment_setup import build_stage1_manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        stage1 = build_stage1_manifest(manifest)
        # One entry per cytokine (including PBS)
        cytokines_in = {e["cytokine"] for e in stage1}
        all_cytokines = {e["cytokine"] for e in manifest}
        assert cytokines_in == all_cytokines
        assert len(stage1) == len(all_cytokines)

    def test_build_stage1_manifest_saves_json(self, manifest_path, tmp_path):
        from cytokine_mil.experiment_setup import build_stage1_manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        out = str(tmp_path / "stage1.json")
        build_stage1_manifest(manifest, save_path=out)
        with open(out) as f:
            saved = json.load(f)
        assert isinstance(saved, list)
        assert len(saved) == len({e["cytokine"] for e in manifest})

    def test_filter_manifest_keeps_subset_and_pbs(self, manifest_path):
        from cytokine_mil.experiment_setup import filter_manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        subset = ["IL-2", "IL-4"]
        filtered = filter_manifest(manifest, subset)
        cytokines_in = {e["cytokine"] for e in filtered}
        assert cytokines_in == {"IL-2", "IL-4", "PBS"}

    def test_filter_manifest_exclude_pbs(self, manifest_path):
        from cytokine_mil.experiment_setup import filter_manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        filtered = filter_manifest(manifest, ["IL-2"], include_pbs=False)
        assert all(e["cytokine"] == "IL-2" for e in filtered)

    def test_make_binary_manifest_two_classes(self, manifest_path):
        from cytokine_mil.experiment_setup import make_binary_manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        filtered, label_enc = make_binary_manifest(manifest, "IL-2")
        assert {e["cytokine"] for e in filtered} == {"IL-2", "PBS"}
        assert label_enc.encode("IL-2") == 0
        assert label_enc.encode("PBS") == 1
        assert label_enc.n_classes() == 2

    def test_build_encoder_and_mil_model(self):
        from cytokine_mil.experiment_setup import build_encoder, build_mil_model
        enc = build_encoder(n_input_genes=N_GENES, n_cell_types=len(CELL_TYPES))
        model = build_mil_model(enc, n_classes=N_CLASSES)
        X = torch.randn(50, N_GENES)
        y_hat, a, H = model(X)
        assert y_hat.shape == (N_CLASSES,)
        assert a.shape == (50,)

    def test_build_mil_model_binary(self):
        from cytokine_mil.experiment_setup import build_encoder, build_mil_model
        enc = build_encoder(n_input_genes=N_GENES, n_cell_types=len(CELL_TYPES))
        model = build_mil_model(enc, n_classes=2)
        y_hat, _, _ = model(torch.randn(50, N_GENES))
        assert y_hat.shape == (2,)


class TestDonorValidationSplit:
    """Tests for split_manifest_by_donor and train_mil val_dataset integration."""

    def test_split_counts(self, manifest_path):
        from cytokine_mil.experiment_setup import split_manifest_by_donor
        with open(manifest_path) as f:
            manifest = json.load(f)
        train_m, val_m = split_manifest_by_donor(manifest, val_donors=["Donor3"])
        assert all(e["donor"] != "Donor3" for e in train_m)
        assert all(e["donor"] == "Donor3" for e in val_m)
        assert len(train_m) + len(val_m) == len(manifest)

    def test_split_preserves_cytokines(self, manifest_path):
        from cytokine_mil.experiment_setup import split_manifest_by_donor
        with open(manifest_path) as f:
            manifest = json.load(f)
        train_m, val_m = split_manifest_by_donor(manifest, val_donors=["Donor3"])
        assert {e["cytokine"] for e in train_m} == {e["cytokine"] for e in manifest}
        assert {e["cytokine"] for e in val_m} == {e["cytokine"] for e in manifest}

    def test_train_mil_with_val_dataset_returns_val_records(
        self, demo_dir, manifest_path, label_encoder
    ):
        from cytokine_mil.data.dataset import PseudoTubeDataset
        from cytokine_mil.experiment_setup import split_manifest_by_donor
        from cytokine_mil.training.train_mil import train_mil
        with open(manifest_path) as f:
            manifest = json.load(f)
        train_m, val_m = split_manifest_by_donor(manifest, val_donors=["Donor3"])

        import tempfile, json as _json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tf:
            _json.dump(train_m, tf)
            train_path = tf.name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tf:
            _json.dump(val_m, tf)
            val_path = tf.name

        train_dataset = PseudoTubeDataset(train_path, label_encoder)
        val_dataset = PseudoTubeDataset(val_path, label_encoder)
        model = _make_fresh_mil_model()
        dynamics = train_mil(
            model, train_dataset, n_epochs=2, log_every_n_epochs=1,
            verbose=False, seed=42, val_dataset=val_dataset,
        )
        assert "val_records" in dynamics
        assert "val_confusion_entropy_trajectory" in dynamics
        assert len(dynamics["val_records"]) == len(val_m)
        for rec in dynamics["val_records"]:
            assert "p_correct_trajectory" in rec
            assert "entropy_trajectory" in rec
            assert len(rec["p_correct_trajectory"]) == len(dynamics["logged_epochs"])

    def test_train_mil_without_val_dataset_returns_empty(self, dataset):
        from cytokine_mil.training.train_mil import train_mil
        model = _make_fresh_mil_model()
        dynamics = train_mil(
            model, dataset, n_epochs=1, log_every_n_epochs=1,
            verbose=False, seed=42,
        )
        assert dynamics["val_records"] == []
        assert dynamics["val_confusion_entropy_trajectory"] == {}


class TestBinaryLabel:
    def test_encode_decode(self):
        from cytokine_mil.data.label_encoder import BinaryLabel
        enc = BinaryLabel(positive="IL-2", negative="PBS")
        assert enc.encode("IL-2") == 0
        assert enc.encode("PBS") == 1
        assert enc.decode(0) == "IL-2"
        assert enc.decode(1) == "PBS"

    def test_n_classes(self):
        from cytokine_mil.data.label_encoder import BinaryLabel
        enc = BinaryLabel("IL-4")
        assert enc.n_classes() == 2

    def test_cytokines_property(self):
        from cytokine_mil.data.label_encoder import BinaryLabel
        enc = BinaryLabel("TNF", "PBS")
        assert enc.cytokines == ["TNF", "PBS"]


class TestValidation:
    def test_fdr_correction_shape(self):
        from cytokine_mil.analysis.validation import apply_fdr_correction
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        rejected, corrected = apply_fdr_correction(pvals)
        assert rejected.shape == pvals.shape
        assert corrected.shape == pvals.shape

    def test_fdr_rejects_small_pvalues(self):
        from cytokine_mil.analysis.validation import apply_fdr_correction
        pvals = np.array([1e-10, 1e-8, 0.9, 0.95])
        rejected, _ = apply_fdr_correction(pvals)
        assert rejected[0] and rejected[1]
        assert not rejected[2] and not rejected[3]

    def test_seed_stability_structure(self, dataset):
        from cytokine_mil.training.train_mil import train_mil
        from cytokine_mil.analysis.validation import check_seed_stability
        model_a = _make_fresh_mil_model()
        model_b = _make_fresh_mil_model()
        dyn_a = train_mil(model_a, dataset, n_epochs=2, verbose=False, seed=0)
        dyn_b = train_mil(model_b, dataset, n_epochs=2, verbose=False, seed=1)
        result = check_seed_stability([dyn_a, dyn_b], exclude=["PBS"])
        assert "rankings" in result
        assert "spearman_matrix" in result
        assert result["spearman_matrix"].shape == (2, 2)
        assert "stable" in result
