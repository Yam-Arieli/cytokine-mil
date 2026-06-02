"""Configuration dataclasses with the method's scientifically-locked defaults.

Every default here is the value validated in the source study (Oesinghaus 88% /
Sheu 86% / Immune Dictionary 83%). They are exposed so a user can *deliberately*
override them, but the defaults reproduce the published method exactly. Changing
them changes the science — see the docstrings for what each one controls.

All configs are frozen (immutable) dataclasses: build a new one to change a value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    """Normalization + highly-variable-gene (HVG) selection.

    Attributes:
        n_hvgs: Number of highly-variable genes to keep (4000, matching the study).
            HVG selection is what lets the binary models learn condition-specific
            programs instead of housekeeping noise.
        target_sum: Per-cell total-count normalization target (1e4 = "counts per
            10k"). Applied before log1p.
        flavor: scanpy HVG flavor for raw-count input ("seurat_v3"). For
            already-log-normalized input with no counts layer, cascadir falls back
            to "seurat" (documented in preprocess()).
        batch_key: Optional ``obs`` column to compute HVGs within-batch (e.g. the
            donor column for multi-batch data). ``None`` = pool all cells.
    """

    n_hvgs: int = 4000
    target_sum: float = 1e4
    flavor: str = "seurat_v3"
    batch_key: str | None = None


@dataclass(frozen=True)
class TubeConfig:
    """Pseudo-tube (bag) construction.

    A pseudo-tube is a bag of cells drawn from one (condition, donor), stratified
    by cell type so that abundance differences do not drive the signal.

    Attributes:
        n_per_cell_type: Cells sampled per cell type per tube (30).
        min_cells: A cell type needs >= this many cells in a (condition, donor) to
            be included in a tube; a tube needs >= this many cells to be kept (10).
            Also the per-(condition, cell_type) floor enforced by cross_asym.
        n_tubes: Tubes built per (condition, donor) (10).
        seed: RNG seed for reproducible sampling.
    """

    n_per_cell_type: int = 30
    min_cells: int = 10
    n_tubes: int = 10
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """AB-MIL training (Stage-1 encoder + per-condition binary models).

    Attributes:
        embed_dim: Cell-embedding dimension (128).
        hidden_dims: InstanceEncoder hidden widths (512, 256).
        attention_hidden_dim: Attention bottleneck width (64).
        encoder_epochs: Stage-1 cell-type pre-training epochs (50).
        binary_epochs: Stage-2 per-condition binary MIL epochs (250).
        encoder_lr: SGD lr for Stage-1 (0.01).
        binary_lr: SGD lr for Stage-2 with a frozen encoder (3e-5).
        momentum: SGD momentum (0.9).
        encoder_frozen: Freeze the encoder during binary MIL training (True) so the
            discovered signatures come from the bag-level attention/classifier, not
            from re-fitting the cell encoder per condition.
    """

    embed_dim: int = 128
    hidden_dims: tuple[int, int] = (512, 256)
    attention_hidden_dim: int = 64
    encoder_epochs: int = 50
    binary_epochs: int = 250
    encoder_lr: float = 0.01
    binary_lr: float = 3e-5
    momentum: float = 0.9
    encoder_frozen: bool = True


@dataclass(frozen=True)
class CrossAsymConfig:
    """Signature derivation, the cross_asym statistic, calls, and the null.

    Attributes:
        top_n: Genes per discovered signature S_X, ranked by Integrated Gradients
            (50). This is the size of the gene set whose cross-engagement is scored.
        n_ig_steps: Integrated-Gradients interpolation steps, midpoint rule (20).
        min_cells: Per-(condition, cell_type) cell floor for the directional test (10).
        magnitude_threshold: Minimum |median cross_asym| for a non-AMBIGUOUS call
            (0.01). Below this there is no detectable directional signal.
        strong_consensus: Minimum sign-consensus across cell types for a STRONG call
            (0.75).
        weak_consensus: Minimum sign-consensus for a WEAK call (0.50).
        n_null_perms: Random-gene-set null permutations per pair (100). Set 0 to skip.
        null_seed: RNG seed for the null.
    """

    top_n: int = 50
    n_ig_steps: int = 20
    min_cells: int = 10
    magnitude_threshold: float = 0.01
    strong_consensus: float = 0.75
    weak_consensus: float = 0.50
    n_null_perms: int = 100
    null_seed: int = 42
