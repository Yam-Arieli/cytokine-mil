"""``CascadeDirection`` — the one-call orchestrator.

Wraps the full method: validate -> preprocess -> build pseudo-tubes -> train the
Stage-1 encoder -> train per-condition binary models -> derive signatures, then
answers direction queries. Every composable step is also a public module-level
function, so you can run any stage yourself, on your own device. Nothing here
touches the cluster, SLURM, or the filesystem.

Example::

    from cascadir import CascadeDirection

    cd = CascadeDirection(
        condition_col="cytokine", donor_col="donor", celltype_col="cell_type",
        control_label="PBS", device="cpu",
    ).fit(adata, conditions=["IFNb", "IFNg", "IL6", "TNFa"], assume="raw")

    print(cd.direction("IFNb", "IFNg"))   # one pair
    print(cd.direction_table())            # all pairs, sorted by |cross_asym|
"""

from __future__ import annotations

import pandas as pd

from cascadir.analysis import score_directions
from cascadir.config import (
    CrossAsymConfig,
    PreprocessConfig,
    TrainConfig,
    TubeConfig,
)
from cascadir.coupling import discover_axes
from cascadir.cross_asym import direction_call, direction_table
from cascadir.exceptions import NotFittedError
from cascadir.preprocess import preprocess
from cascadir.pseudotubes import build_pseudotubes
from cascadir.signatures import derive_signatures
from cascadir.train import train_all_binary, train_encoder
from cascadir.types import (
    AxisResult,
    BenchmarkResult,
    DirectionCall,
    PseudoTubeSet,
    Signature,
    ValidationReport,
)
from cascadir.validate import validate_anndata


class CascadeDirection:
    """End-to-end cascade-direction estimator.

    Args:
        condition_col / donor_col / celltype_col: ``obs`` column names in the input
            AnnData.
        control_label: The resting/unstimulated label (PBS baseline).
        preprocess_config / tube_config / train_config / cross_asym_config: Frozen
            config dataclasses with the method's validated defaults.
        device: ``None`` (auto: cuda>mps>cpu) or any torch device spec ("cpu",
            "cuda", "mps", ...). This is how you choose where it runs.
        seed: Global seed for tube sampling and training.
    """

    def __init__(
        self,
        *,
        condition_col: str,
        donor_col: str,
        celltype_col: str,
        control_label: str = "PBS",
        preprocess_config: PreprocessConfig | None = None,
        tube_config: TubeConfig | None = None,
        train_config: TrainConfig | None = None,
        cross_asym_config: CrossAsymConfig | None = None,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.condition_col = condition_col
        self.donor_col = donor_col
        self.celltype_col = celltype_col
        self.control_label = control_label
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.tube_config = tube_config or TubeConfig()
        self.train_config = train_config or TrainConfig()
        self.cross_asym_config = cross_asym_config or CrossAsymConfig()
        self.device = device
        self.seed = seed

        # populated by fit()
        self.validation_report: ValidationReport | None = None
        self.tube_set: PseudoTubeSet | None = None
        self.encoder = None
        self.models: dict = {}
        self.signatures: dict[str, Signature] = {}
        self._cells_by_pair: dict | None = None
        self._fitted = False

    # -- fitting -------------------------------------------------------------

    def fit(
        self,
        adata,
        *,
        conditions: list[str] | None = None,
        assume: str = "auto",
        validate: bool = True,
    ) -> "CascadeDirection":
        """Run the full pipeline and store the fitted state.

        Args:
            adata: Raw-or-log-normalized cells x genes AnnData with the three named
                ``obs`` columns and a control condition.
            conditions: Stimuli to model (default: every non-control condition). The
                control is always included. Restricting this saves training time.
            assume: Passed to :func:`cascadir.preprocess.preprocess`
                ("auto" / "raw" / "lognorm").
            validate: Run strict validation first (recommended).

        Returns:
            ``self`` (fitted).
        """
        pc, tc, trc, cac = (
            self.preprocess_config,
            self.tube_config,
            self.train_config,
            self.cross_asym_config,
        )

        if validate:
            self.validation_report = validate_anndata(
                adata,
                condition_col=self.condition_col,
                donor_col=self.donor_col,
                celltype_col=self.celltype_col,
                control_label=self.control_label,
                min_cells=tc.min_cells,
                n_hvgs=pc.n_hvgs,
                strict=True,
            )

        proc = preprocess(
            adata,
            n_hvgs=pc.n_hvgs,
            target_sum=pc.target_sum,
            flavor=pc.flavor,
            batch_key=pc.batch_key,
            assume=assume,
            copy=True,
        )

        if conditions is not None:
            keep = set(conditions) | {self.control_label}
            mask = proc.obs[self.condition_col].astype(str).isin(keep).to_numpy()
            proc = proc[mask].copy()

        self.tube_set = build_pseudotubes(
            proc,
            condition_col=self.condition_col,
            donor_col=self.donor_col,
            celltype_col=self.celltype_col,
            control_label=self.control_label,
            n_per_cell_type=tc.n_per_cell_type,
            min_cells=tc.min_cells,
            n_tubes=tc.n_tubes,
            seed=tc.seed,
        )

        self.encoder = train_encoder(
            proc,
            celltype_col=self.celltype_col,
            embed_dim=trc.embed_dim,
            hidden_dims=trc.hidden_dims,
            epochs=trc.encoder_epochs,
            lr=trc.encoder_lr,
            momentum=trc.momentum,
            device=self.device,
            seed=self.seed,
        )

        target = (
            conditions
            if conditions is not None
            else list(self.tube_set.stimulus_conditions)
        )
        self.models = train_all_binary(
            self.tube_set,
            self.encoder,
            conditions=target,
            control_label=self.control_label,
            attention_hidden_dim=trc.attention_hidden_dim,
            epochs=trc.binary_epochs,
            lr=trc.binary_lr,
            momentum=trc.momentum,
            encoder_frozen=trc.encoder_frozen,
            device=self.device,
            seed=self.seed,
        )

        self.signatures = derive_signatures(
            self.models,
            self.tube_set,
            top_n=cac.top_n,
            n_steps=cac.n_ig_steps,
            device=self.device,
        )
        self._cells_by_pair = self.tube_set.cells_by_pair()
        self._fitted = True
        return self

    # -- querying ------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(
                "CascadeDirection is not fitted; call .fit(adata) first."
            )

    def direction(self, a: str, b: str) -> DirectionCall:
        """Return the :class:`DirectionCall` for one pair of conditions."""
        self._check_fitted()
        assert self.tube_set is not None and self._cells_by_pair is not None
        return direction_call(
            self._cells_by_pair,
            self.signatures,
            self.tube_set.gene_names,
            a,
            b,
            control_label=self.control_label,
            config=self.cross_asym_config,
        )

    def direction_table(
        self, pairs: list[tuple[str, str]] | None = None
    ) -> pd.DataFrame:
        """Return a tidy table of direction calls.

        Args:
            pairs: Pairs to score (default: all unordered pairs among the modeled
                conditions).
        """
        self._check_fitted()
        assert self.tube_set is not None and self._cells_by_pair is not None
        if pairs is None:
            conds = sorted(self.signatures.keys())
            pairs = [
                (conds[i], conds[j])
                for i in range(len(conds))
                for j in range(i + 1, len(conds))
            ]
        return direction_table(
            self._cells_by_pair,
            self.signatures,
            self.tube_set.gene_names,
            pairs,
            control_label=self.control_label,
            config=self.cross_asym_config,
        )

    # -- Path A (coupling) + analysis ---------------------------------------

    def discover_axes(
        self,
        *,
        train_donors: list[str] | None = None,
        direction_mode: str = "global",
        alpha: float = 0.05,
    ) -> AxisResult:
        """Path A: discover coupling axes (*which* pairs are linked) from the fitted
        encoder + pseudo-tubes via latent-space geometry.

        Direction-agnostic existence test, complementary to ``direction`` (which
        assigns direction). See :func:`cascadir.coupling.discover_axes` and its power
        caveat (needs several donors).
        """
        self._check_fitted()
        assert self.tube_set is not None and self.encoder is not None
        return discover_axes(
            self.tube_set,
            self.encoder,
            train_donors=train_donors,
            direction_mode=direction_mode,
            alpha=alpha,
            device=self.device,
        )

    def benchmark(
        self,
        labels: list[tuple[str, str]],
        *,
        null_alpha: float = 0.05,
    ) -> BenchmarkResult:
        """Analysis: score direction calls against known ``(upstream, downstream)``
        labels — cross_asym accuracy, the symmetric directional_score control, the
        classification breakdown, and the null-pass count.

        Only the labeled pairs are scored (efficient). See
        :func:`cascadir.analysis.score_directions`.
        """
        self._check_fitted()
        pairs = sorted({tuple(sorted(p)) for p in labels})
        table = self.direction_table(pairs=pairs)
        return score_directions(table, labels, null_alpha=null_alpha)
