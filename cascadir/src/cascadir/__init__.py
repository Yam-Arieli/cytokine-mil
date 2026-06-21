"""cascadir — cascade-direction inference from a single-cell snapshot.

Estimate **who is upstream** in a signaling cascade from one snapshot of
stimulus-labeled single-cell expression, using the antisymmetric cross-engagement
statistic ``cross_asym(a, b) = s(a, S_b) - s(b, S_a)``.

Quick start::

    import cascadir as cd

    est = cd.CascadeDirection(
        condition_col="cytokine", donor_col="donor", celltype_col="cell_type",
        control_label="PBS", device="cpu",
    ).fit(adata, assume="raw")           # raw counts -> auto preprocess
    print(est.direction_table())          # all pairs, sorted by |cross_asym|

What it answers: **direction, not existence.** Given a pair, who is upstream — not
whether the pair is coupled at all (a non-coupled pair can still score large). See
the README for the full method and honest caveats.

Composable steps (run any stage yourself, on your own device):
    validate_anndata -> preprocess -> build_pseudotubes -> train_encoder ->
    train_all_binary -> derive_signatures -> direction_call / direction_table
"""

from __future__ import annotations

__version__ = "0.1.0"

# Config (validated defaults)
from cascadir.config import (
    CrossAsymConfig,
    PreprocessConfig,
    TrainConfig,
    TubeConfig,
)

# Analysis / benchmark
from cascadir.analysis import score_directions

# Coupling (Path A)
from cascadir.coupling import (
    build_axis_table,
    build_embedding_cache,
    compute_directional_bias_per_donor,
    compute_pbs_centroids_per_cell_type,
    discover_axes,
    test_directional_significance,
)

# Coupling (signature-space) — the second coupling path
from cascadir.signature_coupling import (
    cross_engagement_matrix,
    signature_coupling,
)

# Cross_asym core
from cascadir.cross_asym import (
    aggregate_direction,
    classify_call,
    direction_call,
    direction_table,
    directional_asymmetry_test,
    random_gene_set_null,
)

# Exceptions
from cascadir.exceptions import (
    CascadirError,
    DataValidationError,
    InsufficientDataError,
    NotFittedError,
    NotPreprocessedError,
    SignatureError,
)

# Models
from cascadir.models import AbMil, AttentionModule, BagClassifier, InstanceEncoder

# Orchestrator
from cascadir.pipeline import CascadeDirection

# Preprocessing
from cascadir.preprocess import (
    is_lognormalized,
    is_raw_counts,
    normalize_log1p,
    preprocess,
    select_hvgs,
)

# Pseudo-tubes
from cascadir.pseudotubes import InMemoryTubeDataset, build_pseudotubes

# Signatures
from cascadir.signatures import (
    derive_signature,
    derive_signatures,
    integrated_gradients,
)

# Recurrent IG (opt-in signature trajectories — see cascadir.dynamics)
from cascadir.dynamics import (
    coupling_trajectory,
    derive_signature_trajectory,
    signature_trajectory_collector,
)

# Training
from cascadir.train import (
    resolve_device,
    train_all_binary,
    train_binary_mil,
    train_encoder,
    train_multiclass_mil,
)

# Types
from cascadir.types import (
    AxisResult,
    BenchmarkResult,
    BinaryLabel,
    DirectionCall,
    MultiLabel,
    PseudoTube,
    PseudoTubeSet,
    Signature,
    SignatureCheckpoint,
    SignatureTrajectory,
    ValidationReport,
)

# Validation
from cascadir.validate import validate_anndata

__all__ = [
    "__version__",
    # orchestrator
    "CascadeDirection",
    # configs
    "PreprocessConfig",
    "TubeConfig",
    "TrainConfig",
    "CrossAsymConfig",
    # validate / preprocess
    "validate_anndata",
    "preprocess",
    "normalize_log1p",
    "select_hvgs",
    "is_raw_counts",
    "is_lognormalized",
    # pseudo-tubes
    "build_pseudotubes",
    "InMemoryTubeDataset",
    # models
    "AbMil",
    "AttentionModule",
    "BagClassifier",
    "InstanceEncoder",
    # training
    "train_encoder",
    "train_binary_mil",
    "train_all_binary",
    "train_multiclass_mil",
    "resolve_device",
    # signatures
    "integrated_gradients",
    "derive_signature",
    "derive_signatures",
    # recurrent IG (opt-in)
    "derive_signature_trajectory",
    "signature_trajectory_collector",
    "coupling_trajectory",
    "SignatureCheckpoint",
    "SignatureTrajectory",
    # cross_asym
    "directional_asymmetry_test",
    "aggregate_direction",
    "classify_call",
    "random_gene_set_null",
    "direction_call",
    "direction_table",
    # coupling (Path A)
    "discover_axes",
    "build_embedding_cache",
    "build_axis_table",
    "compute_pbs_centroids_per_cell_type",
    "compute_directional_bias_per_donor",
    "test_directional_significance",
    # coupling (signature-space — second path)
    "cross_engagement_matrix",
    "signature_coupling",
    # analysis / benchmark
    "score_directions",
    # types
    "PseudoTube",
    "PseudoTubeSet",
    "Signature",
    "DirectionCall",
    "AxisResult",
    "BenchmarkResult",
    "ValidationReport",
    "BinaryLabel",
    "MultiLabel",
    # exceptions
    "CascadirError",
    "DataValidationError",
    "NotPreprocessedError",
    "InsufficientDataError",
    "SignatureError",
    "NotFittedError",
]
