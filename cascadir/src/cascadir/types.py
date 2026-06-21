"""Typed data containers that define cascadir's input/output contracts.

These frozen dataclasses are the precise "formats" that flow between steps:
a :class:`PseudoTubeSet` comes out of pseudo-tube construction, a
:class:`Signature` per condition comes out of the bridge, and a
:class:`DirectionCall` is the final answer for one pair. Holding state in typed
containers (rather than loose tuples/dicts) is what makes the pipeline
self-documenting and lets us validate shapes at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid a hard pandas import at module load for type hints only
    import pandas as pd


# ---------------------------------------------------------------------------
# Label encoder (binary: condition-vs-control)
# ---------------------------------------------------------------------------


class BinaryLabel:
    """Two-class label encoder for one-condition-vs-control models.

    ``positive -> 0``, ``negative(control) -> 1``. The positive class is index 0 so
    that Integrated Gradients attributes ``logit[0]`` (the "this is the stimulus"
    signal). Mirrors the source project's contract exactly.
    """

    def __init__(self, positive: str, negative: str = "PBS") -> None:
        if positive == negative:
            raise ValueError(
                f"positive and negative labels must differ (both were {positive!r})."
            )
        self.positive = positive
        self.negative = negative
        self._label_to_idx: dict[str, int] = {positive: 0, negative: 1}
        self._idx_to_label: dict[int, str] = {0: positive, 1: negative}

    def encode(self, label: str) -> int:
        return self._label_to_idx[label]

    def decode(self, idx: int) -> str:
        return self._idx_to_label[idx]

    def n_classes(self) -> int:
        return 2

    @property
    def cytokines(self) -> list[str]:  # name kept for trainer compatibility
        return [self.positive, self.negative]


class MultiLabel:
    """Multiclass label encoder over all conditions (including the control).

    Conditions are sorted and assigned indices ``0..K-1``. Used by the multiclass
    model and the latent-geometry (Path A) coupling analysis. Exposes
    ``_idx_to_label`` and ``cytokines`` so it plugs into the vendored geometry code.
    """

    def __init__(self, conditions) -> None:
        names = sorted({str(c) for c in conditions})
        if not names:
            raise ValueError("MultiLabel requires at least one condition.")
        self._label_to_idx: dict[str, int] = {c: i for i, c in enumerate(names)}
        self._idx_to_label: dict[int, str] = {i: c for i, c in enumerate(names)}

    def encode(self, label: str) -> int:
        return self._label_to_idx[label]

    def decode(self, idx: int) -> str:
        return self._idx_to_label[idx]

    def n_classes(self) -> int:
        return len(self._label_to_idx)

    @property
    def cytokines(self) -> list[str]:
        return [self._idx_to_label[i] for i in range(len(self._idx_to_label))]


# ---------------------------------------------------------------------------
# Pseudo-tubes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PseudoTube:
    """One bag of cells from a single (condition, donor), stratified by cell type.

    Attributes:
        X: ``(n_cells, n_genes)`` float32 expression (log-normalized, HVG-subset).
        condition: The stimulus label this tube was built from.
        donor: The biological replicate this tube was built from.
        cell_types: Per-cell cell-type label, length ``n_cells`` (aligned to rows of X).
        cell_types_included: The distinct cell types present in this tube.
        tube_idx: Index of this tube within its (condition, donor) group.
    """

    X: np.ndarray
    condition: str
    donor: str
    cell_types: tuple[str, ...]
    cell_types_included: tuple[str, ...]
    tube_idx: int

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(f"PseudoTube.X must be 2-D, got shape {self.X.shape}.")
        if len(self.cell_types) != self.X.shape[0]:
            raise ValueError(
                f"len(cell_types)={len(self.cell_types)} != n_cells={self.X.shape[0]} "
                f"for tube (condition={self.condition!r}, donor={self.donor!r})."
            )

    @property
    def n_cells(self) -> int:
        return self.X.shape[0]


@dataclass
class PseudoTubeSet:
    """A collection of :class:`PseudoTube` plus the shared gene order and labels.

    This is the in-memory replacement for the source project's on-disk
    manifest+h5ad folder — no files are written.
    """

    tubes: list[PseudoTube]
    gene_names: tuple[str, ...]
    control_label: str

    def __post_init__(self) -> None:
        if not self.tubes:
            raise ValueError("PseudoTubeSet is empty (no tubes were built).")
        g = len(self.gene_names)
        for t in self.tubes:
            if t.X.shape[1] != g:
                raise ValueError(
                    f"tube (condition={t.condition!r}, donor={t.donor!r}) has "
                    f"{t.X.shape[1]} genes but gene_names has {g}."
                )

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(sorted({t.condition for t in self.tubes}))

    @property
    def stimulus_conditions(self) -> tuple[str, ...]:
        """All conditions except the control."""
        return tuple(c for c in self.conditions if c != self.control_label)

    @property
    def donors(self) -> tuple[str, ...]:
        return tuple(sorted({t.donor for t in self.tubes}))

    @property
    def cell_types(self) -> tuple[str, ...]:
        out: set[str] = set()
        for t in self.tubes:
            out.update(t.cell_types_included)
        return tuple(sorted(out))

    def gene_index(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.gene_names)}

    def filter(
        self,
        *,
        conditions: list[str] | None = None,
        donors: list[str] | None = None,
    ) -> "PseudoTubeSet":
        """Return a new set keeping only the given conditions and/or donors."""
        cond_set = set(conditions) if conditions is not None else None
        donor_set = set(donors) if donors is not None else None
        kept = [
            t
            for t in self.tubes
            if (cond_set is None or t.condition in cond_set)
            and (donor_set is None or t.donor in donor_set)
        ]
        return PseudoTubeSet(
            tubes=kept, gene_names=self.gene_names, control_label=self.control_label
        )

    def cells_by_pair(
        self,
        conditions: list[str] | None = None,
        donors: list[str] | None = None,
    ) -> dict[tuple[str, str], np.ndarray]:
        """Group all cells into ``{(condition, cell_type): (n_cells, n_genes)}``.

        This is the exact input shape the directional test consumes. Cells are
        pooled across tubes and donors (optionally restricted via ``donors``). The
        control condition is always included if present so PBS-normalization works.

        Args:
            conditions: If given, keep only these conditions (the control is added
                automatically). ``None`` = all conditions.
            donors: If given, pool only cells from these donors. ``None`` = all donors.
        """
        keep_cond: set[str] | None
        if conditions is not None:
            keep_cond = set(conditions) | {self.control_label}
        else:
            keep_cond = None
        donor_set = set(donors) if donors is not None else None

        buckets: dict[tuple[str, str], list[np.ndarray]] = {}
        for t in self.tubes:
            if keep_cond is not None and t.condition not in keep_cond:
                continue
            if donor_set is not None and t.donor not in donor_set:
                continue
            ct_arr = np.asarray(t.cell_types)
            for ct in t.cell_types_included:
                rows = np.where(ct_arr == ct)[0]
                if rows.size == 0:
                    continue
                buckets.setdefault((t.condition, ct), []).append(t.X[rows])
        return {
            key: np.concatenate(chunks, axis=0) for key, chunks in buckets.items()
        }


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signature:
    """A discovered per-condition gene set S_X (top-``top_n`` by Integrated Gradients).

    Attributes:
        condition: The stimulus this signature was discovered for.
        genes: The ``top_n`` gene names, most-attributed first.
        ig_scores: Mean IG attribution per gene, aligned to ``genes``.
        top_n: The requested signature size (``len(genes)`` may be smaller if the
            panel had fewer genes).
    """

    condition: str
    genes: tuple[str, ...]
    ig_scores: tuple[float, ...]
    top_n: int

    def __post_init__(self) -> None:
        if len(self.genes) != len(self.ig_scores):
            raise ValueError(
                f"Signature for {self.condition!r}: genes ({len(self.genes)}) and "
                f"ig_scores ({len(self.ig_scores)}) length mismatch."
            )
        if not self.genes:
            raise ValueError(f"Signature for {self.condition!r} is empty.")


# ---------------------------------------------------------------------------
# Recurrent-IG signature trajectories (opt-in; see cascadir.dynamics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignatureCheckpoint:
    """One per-epoch snapshot of a condition's IG gene ranking.

    Attributes:
        epoch: Training epoch at which IG was captured.
        genes: Genes ranked by mean IG (most-attributed first). Full ranking unless a
            ``checkpoint_ig_top_n`` was set.
        ig_scores: Mean IG attribution per gene, aligned to ``genes``.
    """

    epoch: int
    genes: tuple[str, ...]
    ig_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.genes) != len(self.ig_scores):
            raise ValueError(
                f"SignatureCheckpoint epoch={self.epoch}: genes ({len(self.genes)}) "
                f"and ig_scores ({len(self.ig_scores)}) length mismatch."
            )


@dataclass(frozen=True)
class SignatureTrajectory:
    """A condition's IG signature captured across training (recurrent IG).

    The static ``Signature`` is the final checkpoint's top-``top_n``; the trajectory
    keeps every captured epoch so recruitment order, rank trajectories, and the
    per-epoch coupling panel can be reconstructed (see :mod:`cascadir.dynamics`).

    Attributes:
        condition: The stimulus this trajectory was captured for.
        checkpoints: One :class:`SignatureCheckpoint` per captured epoch, in order.
        total_epochs: The full training length (the last checkpoint's epoch may equal
            this when ``total_epochs % checkpoint_every == 0``).
    """

    condition: str
    checkpoints: tuple[SignatureCheckpoint, ...]
    total_epochs: int

    @property
    def epochs(self) -> tuple[int, ...]:
        return tuple(c.epoch for c in self.checkpoints)

    def final(self) -> SignatureCheckpoint | None:
        """The last captured checkpoint (``None`` if no checkpoints were taken)."""
        return self.checkpoints[-1] if self.checkpoints else None

    def signature_at(self, epoch: int, *, top_n: int | None = None) -> Signature:
        """The :class:`Signature` (top-``top_n``) at a captured ``epoch``."""
        for c in self.checkpoints:
            if c.epoch == epoch:
                k = len(c.genes) if top_n is None else min(top_n, len(c.genes))
                return Signature(
                    condition=self.condition,
                    genes=c.genes[:k],
                    ig_scores=c.ig_scores[:k],
                    top_n=top_n if top_n is not None else len(c.genes),
                )
        raise KeyError(f"epoch {epoch} not in trajectory for {self.condition!r}.")


# ---------------------------------------------------------------------------
# Direction call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionCall:
    """The cascade-direction answer for one alphabetically-ordered pair (a, b).

    ``cross_asym = s(a, S_b) - s(b, S_a)`` (PBS-normalized) is *antisymmetric*: its
    sign encodes direction. Positive median => ``a`` is upstream (``a_to_b``).

    Attributes:
        condition_a, condition_b: The pair, with ``condition_a < condition_b``.
        cross_asym_median: Median of cross_asym across cell types (the call statistic).
        directional_score_median: Median of the SYMMETRIC §24 directional_score across
            cell types. Reference only — it is symmetric in (a, b) for self-signatures,
            so its sign does NOT encode direction (the internal control that scored
            ~chance vs cross_asym's ~85% in the source study).
        sign_consensus: Fraction of cell types whose cross_asym matches the median sign.
        n_pos, n_neg: Counts of positive / negative cross_asym across cell types.
        classification: "STRONG" | "WEAK" | "AMBIGUOUS".
        direction: "a_to_b" | "b_to_a" | "ambiguous".
        upstream: The upstream condition name, or ``None`` if ambiguous.
        null_p: Two-sided empirical p vs the random-gene-set null, or ``None`` if the
            null was not run.
        per_cell_type: The full per-cell-type DataFrame (audit trail).
    """

    condition_a: str
    condition_b: str
    cross_asym_median: float
    directional_score_median: float
    sign_consensus: float
    n_pos: int
    n_neg: int
    classification: str
    direction: str
    upstream: str | None
    null_p: float | None
    per_cell_type: "pd.DataFrame" = field(repr=False)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """Result of :func:`cascadir.validate.validate_anndata` in non-strict mode.

    Attributes:
        ok: True iff there are no blocking issues.
        n_cells, n_genes, n_donors, n_cell_types, n_conditions: Dataset shape.
        control_label: The control label that was checked for.
        control_present: Whether the control label exists in the condition column.
        x_state: "raw_counts" | "lognormalized" | "ambiguous".
        issues: Blocking problems (non-empty => ``ok`` is False).
        warnings: Non-blocking advisories.
        per_condition_cells: ``{condition: n_cells}``.
    """

    ok: bool
    n_cells: int
    n_genes: int
    n_donors: int
    n_cell_types: int
    n_conditions: int
    control_label: str
    control_present: bool
    x_state: str
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    per_condition_cells: dict[str, int]

    def summary(self) -> str:
        """A human-readable multi-line summary."""
        lines = [
            f"cascadir ValidationReport: {'OK' if self.ok else 'NOT OK'}",
            f"  cells={self.n_cells}  genes={self.n_genes}  donors={self.n_donors}"
            f"  cell_types={self.n_cell_types}  conditions={self.n_conditions}",
            f"  control={self.control_label!r} present={self.control_present}"
            f"  X state={self.x_state}",
        ]
        if self.issues:
            lines.append("  ISSUES (blocking):")
            lines += [f"    - {m}" for m in self.issues]
        if self.warnings:
            lines.append("  warnings:")
            lines += [f"    - {m}" for m in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path A (coupling) + analysis results
# ---------------------------------------------------------------------------


@dataclass
class AxisResult:
    """Output of Path A coupling discovery (latent-geometry axes).

    Attributes:
        axes: Tidy DataFrame, one row per unordered pair, columns include
            ``axis_a``, ``axis_b``, ``coupled`` (bool, at ``alpha``),
            ``axis_strength`` (max Wilcoxon W across cell types — the ranking score),
            ``coupling_call`` (a_to_b/b_to_a/shared/none), ``dominant_direction``,
            ``p_fwd``/``p_rev`` (min Bonferroni p per ordered direction),
            ``relay_T`` (candidate relay cell type). Sorted by descending strength.
        significance: The raw ``test_directional_significance`` dict (full audit trail).
        n_donors: Number of donors that contributed to the per-donor Wilcoxon.
        alpha: Significance threshold used for ``coupled``.
        direction_mode: 'global' or 'cell_type'.
        underpowered: True if ``n_donors`` is below the advisory floor (the donor-level
            Wilcoxon cannot reach ``alpha`` with too few donors — rank by
            ``axis_strength`` instead of trusting ``coupled``).
    """

    axes: "pd.DataFrame" = field(repr=False)
    significance: dict = field(repr=False)
    n_donors: int
    alpha: float
    direction_mode: str
    underpowered: bool

    def summary(self) -> str:
        n_axes = int(self.axes["coupled"].sum()) if len(self.axes) else 0
        lines = [
            f"cascadir AxisResult: {n_axes} coupled axes at alpha={self.alpha} "
            f"(of {len(self.axes)} pairs), n_donors={self.n_donors}, "
            f"direction_mode={self.direction_mode}",
        ]
        if self.underpowered:
            lines.append(
                "  WARNING: too few donors for the donor-level Wilcoxon to reach "
                f"alpha={self.alpha}; trust the axis_strength RANKING, not 'coupled'."
            )
        return "\n".join(lines)


@dataclass
class BenchmarkResult:
    """Output of scoring direction calls against known-direction labels.

    Attributes:
        n_labeled: Number of labeled pairs provided.
        n_found: Labeled pairs that had a computed direction call.
        n_scored: Found pairs that were non-AMBIGUOUS (the denominator for accuracy).
        cross_accuracy: cross_asym signed accuracy among non-AMBIGUOUS calls.
        cross_accuracy_all: cross_asym signed accuracy over all found pairs.
        dirscore_accuracy: directional_score (symmetric control) signed accuracy among
            non-AMBIGUOUS calls — expected near chance, the headline contrast.
        n_null_pass: Non-AMBIGUOUS calls also passing the null (null_p < 0.05).
        classification_counts: {STRONG/WEAK/AMBIGUOUS: count} over found pairs.
        table: Per-pair DataFrame (expected vs called, correctness, classification, null_p).
    """

    n_labeled: int
    n_found: int
    n_scored: int
    cross_accuracy: float
    cross_accuracy_all: float
    dirscore_accuracy: float
    n_null_pass: int
    classification_counts: dict
    table: "pd.DataFrame" = field(repr=False)

    def summary(self) -> str:
        return "\n".join(
            [
                "cascadir BenchmarkResult:",
                f"  labeled={self.n_labeled}  found={self.n_found}  "
                f"scored(non-AMBIGUOUS)={self.n_scored}",
                f"  cross_asym accuracy (non-AMBIGUOUS): {self.cross_accuracy:.3f} "
                f"({self.n_scored} scored)",
                f"  cross_asym accuracy (all found):     {self.cross_accuracy_all:.3f}",
                f"  directional_score control accuracy:  {self.dirscore_accuracy:.3f} "
                "(symmetric -> expect ~chance)",
                f"  null-passing (p<0.05) non-AMBIGUOUS: {self.n_null_pass}",
                f"  classification: {self.classification_counts}",
            ]
        )
