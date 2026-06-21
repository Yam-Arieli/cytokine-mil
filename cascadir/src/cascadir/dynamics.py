"""Recurrent Integrated Gradients — per-epoch signature trajectories (opt-in).

By default cascadir runs Integrated Gradients **once**, on the final binary model, and
reports each condition's top-``top_n`` genes as its signature ``S_X``. This module adds
the OPTION to capture IG **every ``checkpoint_every`` epochs** of binary-MIL (full-model,
Stage-2) training — turning each static ``S_X`` into a *trajectory* of gene rankings —
plus the matching per-epoch **degree-corrected cross-engagement panel**.

Nothing here changes the default path. It is only active when you ask for it:

* one condition  -> :func:`derive_signature_trajectory`
* inside the pipeline -> ``CascadeDirection.fit(ig_checkpoint_every=N)`` (which uses
  :func:`signature_trajectory_collector`), then ``.signature_trajectory_table()`` /
  ``.coupling_trajectory()``.

What the trajectory means (and does not). With a **frozen** Stage-1 encoder the
gene->feature map is fixed across epochs, so a gene's *recruitment order* reflects the
attention/classifier learning to weight encoder features — read-out learning, not the
representation drifting. Cross-condition timing is comparable only because the encoder is
shared. See ``hypotheses/recurrent_training_dynamics_IG.md`` and CLAUDE.md §31.

The per-epoch panel reuses the validated signature-coupling math
(:func:`cascadir.signature_coupling.cross_engagement_matrix` +
:func:`cascadir.signature_coupling._degree_center`) unchanged — only the signatures it is
fed change with epoch.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
import torch

from cascadir.exceptions import SignatureError
from cascadir.models import AbMil, InstanceEncoder
from cascadir.signature_coupling import (
    _coupling_matrix,
    _degree_center,
    _pair_rows,
    _signatures_to_idx,
    cross_engagement_matrix,
)
from cascadir.signatures import _control_baseline, integrated_gradients
from cascadir.train import resolve_device, train_binary_mil
from cascadir.types import (
    PseudoTubeSet,
    Signature,
    SignatureCheckpoint,
    SignatureTrajectory,
)

logger = logging.getLogger("cascadir")


# ---------------------------------------------------------------------------
# Per-checkpoint IG ranking (no parameter mutation — safe to call mid-training)
# ---------------------------------------------------------------------------


def _ig_ranking(
    model: AbMil,
    tube_set: PseudoTubeSet,
    condition: str,
    *,
    control_label: str,
    n_steps: int,
    top_n: int | None,
    device: str | torch.device | None,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Rank genes by mean IG for ``condition`` at the model's current state.

    Mirrors :func:`cascadir.signatures.derive_signature` but **does not** flip the
    model's ``requires_grad`` flags (so it is safe to call inside the training loop):
    :func:`integrated_gradients` only needs grad on the interpolated input, computed via
    ``torch.autograd.grad`` w.r.t. that input alone. Returns the full ranking when
    ``top_n is None``.
    """
    dev = resolve_device(device)
    ts = tube_set
    if control_label != tube_set.control_label:
        ts = PseudoTubeSet(
            tubes=tube_set.tubes,
            gene_names=tube_set.gene_names,
            control_label=control_label,
        )
    cond_tubes = [t for t in ts.tubes if t.condition == condition]
    if not cond_tubes:
        raise SignatureError(
            f"No tubes for condition {condition!r}; cannot capture IG trajectory."
        )

    gene_names = ts.gene_names
    g = len(gene_names)
    model = model.to(dev)
    was_training = model.training
    model.eval()
    baseline = _control_baseline(ts, dev)  # (G,)
    ig_accum = np.zeros(g, dtype=np.float64)
    n_used = 0
    for t in cond_tubes:
        X = torch.from_numpy(np.ascontiguousarray(t.X, dtype=np.float32)).to(dev)
        base = baseline.unsqueeze(0).expand_as(X).contiguous()
        ig = integrated_gradients(
            model, X, target_class=0, baseline=base, n_steps=n_steps
        )
        ig_accum += ig.mean(dim=0).detach().cpu().numpy()
        n_used += 1
    if was_training:
        model.train()

    ig_mean = ig_accum / max(n_used, 1)
    order = np.argsort(-ig_mean)
    k = g if top_n is None else min(top_n, g)
    genes = tuple(gene_names[i] for i in order[:k])
    scores = tuple(float(ig_mean[i]) for i in order[:k])
    return genes, scores


# ---------------------------------------------------------------------------
# Collector (multi-condition, single training pass — used by CascadeDirection.fit)
# ---------------------------------------------------------------------------


def signature_trajectory_collector(
    tube_set: PseudoTubeSet,
    *,
    control_label: str = "PBS",
    n_steps: int = 20,
    top_n: int | None = None,
    device: str | torch.device | None = None,
) -> tuple[dict[str, list[SignatureCheckpoint]], Callable[[str], Callable[[int, AbMil], None]]]:
    """Build a ``(trajectories, factory)`` pair for :func:`cascadir.train.train_all_binary`.

    ``factory(condition)`` returns a fresh ``(epoch, model) -> None`` callback that
    appends a :class:`SignatureCheckpoint` to ``trajectories[condition]``. Capturing
    happens during the **same** training pass (no retraining). Pass ``factory`` and a
    ``checkpoint_every`` to ``train_all_binary``; read ``trajectories`` after it returns.
    """
    trajectories: dict[str, list[SignatureCheckpoint]] = {}

    def factory(condition: str) -> Callable[[int, AbMil], None]:
        trajectories.setdefault(condition, [])

        def on_checkpoint(epoch: int, model: AbMil) -> None:
            genes, scores = _ig_ranking(
                model,
                tube_set,
                condition,
                control_label=control_label,
                n_steps=n_steps,
                top_n=top_n,
                device=device,
            )
            trajectories[condition].append(
                SignatureCheckpoint(epoch=epoch, genes=genes, ig_scores=scores)
            )

        return on_checkpoint

    return trajectories, factory


# ---------------------------------------------------------------------------
# Single-condition convenience
# ---------------------------------------------------------------------------


def derive_signature_trajectory(
    tube_set: PseudoTubeSet,
    condition: str,
    encoder: InstanceEncoder,
    *,
    control_label: str = "PBS",
    checkpoint_every: int = 10,
    n_steps: int = 20,
    top_n: int | None = None,
    device: str | torch.device | None = None,
    epochs: int = 250,
    **train_kwargs,
) -> SignatureTrajectory:
    """Train one binary model and capture its IG signature every ``checkpoint_every`` epochs.

    Args:
        tube_set / condition / encoder: As :func:`cascadir.train.train_binary_mil`.
        control_label: Negative class for the binary model and the IG baseline.
        checkpoint_every: Capture interval (epochs).
        n_steps: IG interpolation steps.
        top_n: Genes kept per checkpoint (``None`` = full ranking).
        device / epochs / **train_kwargs: Forwarded to ``train_binary_mil``.

    Returns:
        A :class:`SignatureTrajectory` with one checkpoint per captured epoch.
    """
    checkpoints: list[SignatureCheckpoint] = []

    def on_checkpoint(epoch: int, model: AbMil) -> None:
        genes, scores = _ig_ranking(
            model,
            tube_set,
            condition,
            control_label=control_label,
            n_steps=n_steps,
            top_n=top_n,
            device=device,
        )
        checkpoints.append(
            SignatureCheckpoint(epoch=epoch, genes=genes, ig_scores=scores)
        )

    train_binary_mil(
        tube_set,
        condition,
        encoder,
        control_label=control_label,
        epochs=epochs,
        device=device,
        checkpoint_every=checkpoint_every,
        on_checkpoint=on_checkpoint,
        **train_kwargs,
    )
    return SignatureTrajectory(
        condition=condition, checkpoints=tuple(checkpoints), total_epochs=epochs
    )


# ---------------------------------------------------------------------------
# Per-epoch degree-corrected coupling panel ("panel matrix correction")
# ---------------------------------------------------------------------------


def coupling_trajectory(
    signatures_by_epoch: dict[int, dict[str, Signature]],
    cells_by_pair: dict[tuple[str, str], np.ndarray],
    gene_names: tuple[str, ...],
    *,
    control_label: str = "PBS",
    min_cells: int = 10,
    degree_correct: bool = True,
) -> dict[int, pd.DataFrame]:
    """Per-epoch cross-engagement panel + degree correction, one DataFrame per epoch.

    For each epoch's signatures, builds ``M[a,b] = s(a, S_b) - s(PBS, S_b)`` via
    :func:`cascadir.signature_coupling.cross_engagement_matrix`, then reads off
    ``coupling = M+Mᵀ`` (degree-corrected when ``degree_correct`` and >= 3 conditions —
    the validated hub fix, symmetric so ``cross_asym`` is untouched) and
    ``cross_asym = M-Mᵀ`` (direction). Same math as the static
    :func:`cascadir.signature_coupling.signature_coupling`; only the per-epoch signatures
    vary.

    Args:
        signatures_by_epoch: ``{epoch: {condition: Signature}}``.
        cells_by_pair: ``{(condition, cell_type): (n_cells, n_genes)}`` (epoch-independent
            — the cells do not change; only which genes each signature points at does).
        gene_names: Gene order of the arrays.
        control_label / min_cells: As the static path.
        degree_correct: Apply the hub/degree correction to ``coupling`` (default True).

    Returns:
        ``{epoch: DataFrame}`` with columns ``condition_a, condition_b, m_ab, m_ba,
        coupling_raw, coupling, cross_asym, epoch`` (one row per unordered pair).
    """
    cols = [
        "condition_a", "condition_b", "m_ab", "m_ba",
        "coupling_raw", "coupling", "cross_asym", "epoch",
    ]
    out: dict[int, pd.DataFrame] = {}
    for epoch in sorted(signatures_by_epoch):
        sigs = signatures_by_epoch[epoch]
        sig_idx = _signatures_to_idx(sigs, gene_names)
        conditions, M = cross_engagement_matrix(
            cells_by_pair, sig_idx, control_label=control_label, min_cells=min_cells
        )
        if len(conditions) < 2:
            out[epoch] = pd.DataFrame(columns=cols)
            continue
        rows = _pair_rows(conditions, M)
        idx_of = {c: i for i, c in enumerate(conditions)}
        C_raw = _coupling_matrix(M)
        do_degree = degree_correct and len(conditions) >= 3
        C_used = _degree_center(C_raw) if do_degree else C_raw
        for r in rows:
            i, j = idx_of[r["condition_a"]], idx_of[r["condition_b"]]
            r["coupling_raw"] = float(C_raw[i, j])
            r["coupling"] = float(C_used[i, j])
            r["epoch"] = int(epoch)
        out[epoch] = pd.DataFrame(rows)[cols]
    return out
