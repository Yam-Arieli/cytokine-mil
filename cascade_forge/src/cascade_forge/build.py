"""``CascadeSimulator`` — author a cascade, forge a single-cell snapshot.

Ties together :mod:`.graph` (parse the cascade dict), :mod:`.dynamics` (propagate label
activation over pseudo-time), and :mod:`.expression` (plant weak programs on top of a
dominant cell-type signature) to emit an :class:`anndata.AnnData` per snapshot time that
satisfies the ``cascadir`` data contract, with the ground truth in ``adata.uns``.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from .dynamics import ActivationVector, propagate_all
from .expression import ExpressionConfig, GeneModel, build_gene_model, generate_tube
from .graph import CascadeGraph

Edge = Tuple[str, str]


@dataclass
class SimulationResult:
    """Output of :meth:`CascadeSimulator.simulate`.

    Attributes:
        adatas: ``{pseudo_time: AnnData}`` — one single-snapshot dataset per requested time.
        graph: the normalized cascade ground truth.
        model: the fixed gene model (layout, means, programs, responders).
        snapshot_times: the requested pseudo-times.
        config: the simulator configuration (JSON-friendly).
        activation: ``{t: {applied_label: {label: activation}}}`` at each snapshot.
    """

    adatas: Dict[float, ad.AnnData]
    graph: CascadeGraph
    model: GeneModel
    snapshot_times: List[float]
    config: dict
    activation: Dict[float, Dict[str, ActivationVector]] = field(default_factory=dict)

    @property
    def direct_edges(self) -> List[Edge]:
        """Authored ``(upstream, downstream)`` pairs — the primary benchmark oracle."""
        return list(self.graph.direct)

    @property
    def reachable_edges(self) -> List[Edge]:
        """Transitive closure of the cascade (all true directional pairs)."""
        return list(self.graph.reachable)

    @property
    def bidirectional_pairs(self) -> List[Tuple[str, str]]:
        """Mutually-reachable (feedback) pairs — exclude from a signed direction score."""
        return list(self.graph.bidirectional)

    @property
    def adata(self) -> ad.AnnData:
        """Convenience accessor when exactly one snapshot time was requested."""
        if len(self.adatas) != 1:
            raise ValueError(
                f".adata requires a single snapshot time; got {len(self.adatas)}. "
                "Use .adatas[t]."
            )
        return next(iter(self.adatas.values()))

    def save(self, out_dir: str) -> List[str]:
        """Write ``snapshot_t{t}.h5ad`` per time + ``ground_truth.json``. Returns paths."""
        os.makedirs(out_dir, exist_ok=True)
        paths: List[str] = []
        for t, adata in self.adatas.items():
            path = os.path.join(out_dir, f"snapshot_t{t:g}.h5ad")
            adata.write_h5ad(path)
            paths.append(path)
        gt = dict(self.graph.to_ground_truth())
        gt["config"] = self.config
        gt["snapshot_times"] = list(self.snapshot_times)
        with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
            json.dump(gt, f, indent=2)
        return paths


def _jsonify_activation(act: Dict[str, ActivationVector]) -> Dict[str, Dict[str, float]]:
    return {applied: {lab: float(v) for lab, v in vec.items()} for applied, vec in act.items()}


class CascadeSimulator:
    """Author a cascade dict, forge single-cell snapshots with known-direction ground truth.

    Args:
        cascades: ``{source: {downstream: (strength, pseudo_time_delta)}}`` (delta optional,
            defaults to 1.0; a bare number is a strength). Labels appearing only downstream
            exist with no outgoing cascade.
        n_cell_types: number of (pseudo) cell types.
        n_cells_per_tube: total cells per (donor, label) tube.
        n_donors: number of donors (biological replicates); cascadir needs >= 3.
        n_program_genes: signature genes per label (default 50, matching cascadir top_n).
        n_background_genes: number of uninformative background genes.
        effect_size: program bump at full activation (log1p space, << marker gap ~1.2).
        responder_mode: ``"all"`` (every cell type responds) or ``"receptor"`` (relays,
            with ``resp(src) subset resp(dst)`` enforced).
        responders: optional ``{label: [cell_type, ...]}`` override (receptor mode).
        output: ``"raw"`` (Poisson counts) or ``"lognorm"`` (log1p floats).
        shared_activation: fraction of ``effect_size`` every active label adds to a shared
            pool (the shared-activation confound); 0 = clean.
        condition_col / donor_col / celltype_col / control_label: obs column names + control
            label written into each AnnData (feed the same to ``cascadir.CascadeDirection``).
        dt_step: pseudo-time integration step.
        activation_cap: cap guarding against runaway feedback loops.
        expr_config: advanced :class:`ExpressionConfig` overrides.
        seed: RNG seed.
    """

    def __init__(
        self,
        cascades: Mapping[str, Mapping[str, object]],
        *,
        n_cell_types: int = 4,
        n_cells_per_tube: int = 300,
        n_donors: int = 6,
        n_program_genes: Optional[int] = None,
        n_background_genes: Optional[int] = None,
        effect_size: Optional[float] = None,
        responder_mode: str = "all",
        responders: Optional[Mapping[str, Sequence[str]]] = None,
        output: str = "raw",
        shared_activation: Optional[float] = None,
        condition_col: str = "condition",
        donor_col: str = "donor",
        celltype_col: str = "cell_type",
        control_label: str = "PBS",
        dt_step: float = 0.05,
        activation_cap: float = 10.0,
        expr_config: Optional[ExpressionConfig] = None,
        seed: int = 0,
    ) -> None:
        if output not in ("raw", "lognorm"):
            raise ValueError(f"output must be 'raw' or 'lognorm', got {output!r}")
        if n_cell_types < 1:
            raise ValueError("n_cell_types must be >= 1")
        if n_cells_per_tube < n_cell_types:
            raise ValueError("n_cells_per_tube must be >= n_cell_types")
        if n_donors < 3:
            warnings.warn(
                f"n_donors={n_donors} < 3; cascadir.validate_anndata requires >= 3 donors "
                "and the output will not pass validation.",
                RuntimeWarning, stacklevel=2,
            )
        if control_label in cascades or any(
            control_label in dn for dn in cascades.values()
        ):
            raise ValueError(
                f"control_label {control_label!r} must not be one of the cascade labels."
            )

        self.graph = CascadeGraph.from_dict(cascades)
        self.n_cell_types = int(n_cell_types)
        self.n_cells_per_tube = int(n_cells_per_tube)
        self.n_donors = int(n_donors)
        self.output = output
        self.responder_mode = responder_mode
        self.condition_col = condition_col
        self.donor_col = donor_col
        self.celltype_col = celltype_col
        self.control_label = control_label
        self.dt_step = float(dt_step)
        self.activation_cap = float(activation_cap)
        self.seed = int(seed)

        cfg = expr_config if expr_config is not None else ExpressionConfig()
        if n_program_genes is not None:
            cfg.n_program_genes = int(n_program_genes)
        if n_background_genes is not None:
            cfg.n_background_genes = int(n_background_genes)
        if effect_size is not None:
            cfg.effect_size = float(effect_size)
        if shared_activation is not None:
            cfg.shared_activation = float(shared_activation)
        self.expr_config = cfg

        self._rng = np.random.default_rng(self.seed)
        self.model: GeneModel = build_gene_model(
            self.graph.labels, self.graph.edges,
            n_cell_types=self.n_cell_types, n_donors=self.n_donors,
            responder_mode=responder_mode, responders=responders,
            cfg=cfg, rng=self._rng,
        )
        self.donor_names = [f"donor_{d}" for d in range(self.n_donors)]

    # ------------------------------------------------------------------
    def _config_dict(self) -> dict:
        cfg = self.expr_config
        return {
            "n_cell_types": self.n_cell_types,
            "n_cells_per_tube": self.n_cells_per_tube,
            "n_donors": self.n_donors,
            "output": self.output,
            "responder_mode": self.responder_mode,
            "control_label": self.control_label,
            "condition_col": self.condition_col,
            "donor_col": self.donor_col,
            "celltype_col": self.celltype_col,
            "n_program_genes": cfg.n_program_genes,
            "n_background_genes": cfg.n_background_genes,
            "n_genes": self.model.n_genes,
            "effect_size": cfg.effect_size,
            "shared_activation": cfg.shared_activation,
            "marker_high_mu": cfg.marker_high_mu,
            "dt_step": self.dt_step,
            "seed": self.seed,
        }

    def _build_adata(
        self, t: float, activation_all: Dict[str, ActivationVector]
    ) -> ad.AnnData:
        rows_X: List[np.ndarray] = []
        obs_cond: List[str] = []
        obs_donor: List[str] = []
        obs_ct: List[str] = []
        # Conditions: the control first, then every label (sorted for determinism).
        for d, donor in enumerate(self.donor_names):
            # Control tube (no program).
            Xc, cts = generate_tube(
                self.model, d, None,
                n_cells=self.n_cells_per_tube, output=self.output, rng=self._rng,
            )
            rows_X.append(Xc)
            obs_cond += [self.control_label] * len(cts)
            obs_donor += [donor] * len(cts)
            obs_ct += cts
            # One tube per applied label.
            for label in self.model.labels:
                Xl, cts = generate_tube(
                    self.model, d, activation_all[label],
                    n_cells=self.n_cells_per_tube, output=self.output, rng=self._rng,
                )
                rows_X.append(Xl)
                obs_cond += [label] * len(cts)
                obs_donor += [donor] * len(cts)
                obs_ct += cts

        X = np.concatenate(rows_X, axis=0)
        obs = pd.DataFrame(
            {
                self.condition_col: obs_cond,
                self.donor_col: obs_donor,
                self.celltype_col: obs_ct,
            },
            index=[f"cell_{i}" for i in range(X.shape[0])],
        )
        var = pd.DataFrame(index=list(self.model.gene_names))
        adata = ad.AnnData(X=X, obs=obs, var=var)
        gt = dict(self.graph.to_ground_truth())
        gt["snapshot_time"] = float(t)
        gt["config"] = self._config_dict()
        gt["program_genes_by_label"] = {
            lab: list(genes) for lab, genes in self.model.program_genes.items()
        }
        gt["responders_by_label"] = {
            lab: list(cts) for lab, cts in self.model.responders.items()
        }
        gt["activation_at_snapshot"] = _jsonify_activation(activation_all)
        adata.uns["cascade_forge"] = gt
        return adata

    # ------------------------------------------------------------------
    def simulate(self, snapshot_times: Sequence[float] = (1.0,)) -> SimulationResult:
        """Forge one AnnData per requested pseudo-time.

        Args:
            snapshot_times: pseudo-times at which to snapshot the system. cascadir works
                on a single snapshot, so each time yields its own dataset.

        Returns:
            A :class:`SimulationResult`.
        """
        times = [float(t) for t in snapshot_times]
        if not times:
            raise ValueError("snapshot_times must be non-empty")
        # activation[applied][t] -> vector
        act_by_applied = propagate_all(
            self.graph.labels, self.graph.edges, times,
            dt_step=self.dt_step, activation_cap=self.activation_cap,
        )
        adatas: Dict[float, ad.AnnData] = {}
        activation: Dict[float, Dict[str, ActivationVector]] = {}
        for t in times:
            activation_all = {applied: act_by_applied[applied][t] for applied in self.graph.labels}
            activation[t] = activation_all
            adatas[t] = self._build_adata(t, activation_all)
        return SimulationResult(
            adatas=adatas,
            graph=self.graph,
            model=self.model,
            snapshot_times=times,
            config=self._config_dict(),
            activation=activation,
        )


def simulate(
    cascades: Mapping[str, Mapping[str, object]],
    snapshot_times: Sequence[float] = (1.0,),
    **kwargs: object,
) -> SimulationResult:
    """One-call convenience: build a :class:`CascadeSimulator` and run it."""
    return CascadeSimulator(cascades, **kwargs).simulate(snapshot_times)  # type: ignore[arg-type]
