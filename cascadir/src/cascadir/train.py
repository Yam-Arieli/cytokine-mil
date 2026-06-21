"""Training: Stage-1 cell-type encoder + per-condition binary AB-MIL models.

Two stages, faithful to the validated method:

* **Stage 1** pre-trains the :class:`InstanceEncoder` to classify cell type from a
  single cell. Its backbone becomes a fixed cell representation.
* **Stage 2** trains one **binary** AB-MIL per stimulus (stimulus-vs-control) on a
  shared, frozen encoder. The bag-level head of these models is what Integrated
  Gradients later attributes to discover each condition's signature.

Optimization is SGD with momentum over **mega-batches** (one tube per class, grads
accumulated, one step) — this keeps the gradient scale independent of class balance.
The research-only dynamics logging is intentionally dropped; nothing here writes to
disk. Every entry point takes a ``device`` so you choose where it runs.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from anndata import AnnData
from torch.utils.data import DataLoader, Dataset

from cascadir.exceptions import InsufficientDataError
from cascadir.models import AbMil, AttentionModule, BagClassifier, InstanceEncoder
from cascadir.pseudotubes import InMemoryTubeDataset
from cascadir.types import BinaryLabel, MultiLabel, PseudoTubeSet

logger = logging.getLogger("cascadir")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device spec. ``None`` -> cuda, else mps, else cpu."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage 1: encoder pre-training (cell-type supervision)
# ---------------------------------------------------------------------------


class _CellDataset(Dataset):
    """Per-cell dataset: yields (expression_row, cell_type_int). Sparse-aware, lazy."""

    def __init__(self, X, y: np.ndarray) -> None:
        self.X = X
        self.y = y.astype(np.int64)
        self._sparse = hasattr(X, "toarray")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        row = self.X[i]
        if self._sparse:
            row = row.toarray()
        row = np.asarray(row, dtype=np.float32).ravel()
        return torch.from_numpy(row), int(self.y[i])


def train_encoder(
    adata: AnnData,
    *,
    celltype_col: str,
    embed_dim: int = 128,
    hidden_dims: tuple[int, int] = (512, 256),
    epochs: int = 50,
    lr: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> InstanceEncoder:
    """Pre-train an :class:`InstanceEncoder` on cell-type labels (Stage 1).

    Trains on the **real cells** of the preprocessed AnnData (not on pseudo-tube
    draws), which is the correct Stage-1 signal. The returned encoder's backbone is
    used by the binary models; its cell-type head is discarded downstream.

    Pass the preprocessed AnnData of **unique** cells (the same object given to
    :func:`cascadir.pseudotubes.build_pseudotubes`). Do not pass an AnnData built by
    concatenating pseudo-tube contents — that re-introduces the cross-tube
    cell-duplication this design avoids. (``CascadeDirection.fit`` does this correctly.)

    Args:
        adata: Preprocessed (log-normalized, HVG-subset) AnnData.
        celltype_col: ``obs`` column with cell-type labels.
        embed_dim / hidden_dims: Encoder size.
        epochs / lr / momentum / batch_size: SGD schedule.
        device: Where to train (``None`` = auto).
        seed: Torch seed for reproducibility.
        verbose: Log per-epoch loss/accuracy.

    Returns:
        The trained :class:`InstanceEncoder` (with ``cell_type_head`` set).
    """
    torch.manual_seed(seed)
    dev = resolve_device(device)

    ct = adata.obs[celltype_col].astype(str).to_numpy()
    classes = sorted(set(ct.tolist()))
    if len(classes) < 1:
        raise InsufficientDataError("train_encoder: no cell-type labels found.")
    ct_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([ct_to_idx[c] for c in ct], dtype=np.int64)

    encoder = InstanceEncoder(
        input_dim=adata.n_vars,
        embed_dim=embed_dim,
        n_cell_types=len(classes),
        hidden_dims=hidden_dims,
    ).to(dev)

    loader = DataLoader(
        _CellDataset(adata.X, y), batch_size=batch_size, shuffle=True, num_workers=0
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=momentum)

    encoder.train()
    for epoch in range(1, epochs + 1):
        total, n_correct, n_total = 0.0, 0, 0
        for X, labels in loader:
            X = X.to(dev)
            labels = labels.to(dev)
            h = encoder(X)
            logits = encoder.cell_type_head(h)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item() * len(X)
            n_correct += int((logits.argmax(1) == labels).sum())
            n_total += len(X)
        if verbose:
            logger.info(
                "[Stage 1] epoch %d/%d loss=%.4f acc=%.4f",
                epoch,
                epochs,
                total / max(n_total, 1),
                n_correct / max(n_total, 1),
            )
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Mega-batch helpers (vendored from the validated trainer)
# ---------------------------------------------------------------------------


def _build_class_queues(entries: list[dict], label_encoder) -> dict[int, list[int]]:
    queues: dict[int, list[int]] = {}
    for idx, entry in enumerate(entries):
        cls = label_encoder.encode(entry["condition"])
        queues.setdefault(cls, []).append(idx)
    return queues


def _epoch_megabatches(
    queues: dict[int, list[int]], rng: np.random.Generator
) -> list[dict[int, int]]:
    shuffled = {c: rng.permutation(idxs).tolist() for c, idxs in queues.items()}
    n_steps = max(len(v) for v in shuffled.values())
    for c, idxs in shuffled.items():
        deficit = n_steps - len(idxs)
        if deficit > 0:
            shuffled[c] = idxs + rng.choice(idxs, size=deficit).tolist()
    return [{c: shuffled[c][s] for c in shuffled} for s in range(n_steps)]


def _train_one_megabatch(
    model: AbMil,
    optimizer: torch.optim.Optimizer,
    tubes_per_class: dict[int, tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    optimizer.zero_grad()
    total = 0.0
    n = len(tubes_per_class)
    for _cls, (X, label) in tubes_per_class.items():
        X = X.to(device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)
        y_hat, _a, _H = model(X)
        loss = criterion(y_hat.unsqueeze(0), label_t) / n
        loss.backward()
        total += loss.item()
    optimizer.step()
    return total


# ---------------------------------------------------------------------------
# Stage 2: per-condition binary AB-MIL
# ---------------------------------------------------------------------------


def train_binary_mil(
    tube_set: PseudoTubeSet,
    condition: str,
    encoder: InstanceEncoder,
    *,
    control_label: str = "PBS",
    attention_hidden_dim: int = 64,
    epochs: int = 250,
    lr: float = 3e-5,
    momentum: float = 0.9,
    encoder_frozen: bool = True,
    device: str | torch.device | None = None,
    seed: int = 42,
    verbose: bool = False,
    checkpoint_every: int | None = None,
    on_checkpoint: Callable[[int, AbMil], None] | None = None,
) -> AbMil:
    """Train one binary AB-MIL: ``condition`` vs ``control_label``.

    Uses the shared (frozen) Stage-1 encoder. One mega-batch = one ``condition`` tube
    + one control tube; grads accumulate over the two before a single SGD step.

    Args:
        tube_set: The full pseudo-tube set.
        condition: The stimulus to model (must have tubes).
        encoder: A trained :class:`InstanceEncoder` (shared across all binary models).
        control_label: The negative class.
        attention_hidden_dim / epochs / lr / momentum: Stage-2 schedule.
        encoder_frozen: Keep the encoder fixed (default; recommended).
        device / seed / verbose: Runtime controls.
        checkpoint_every: OPT-IN. If a positive int N, call ``on_checkpoint`` every N
            epochs (model temporarily in ``eval`` mode). ``None`` (default) =
            unchanged behavior. Used for recurrent IG (:mod:`cascadir.dynamics`).
        on_checkpoint: Callback ``(epoch, model) -> None`` invoked at each checkpoint.
            Must not mutate the model's parameters (it may read/attribute only).

    Returns:
        The trained :class:`AbMil` (in ``eval`` mode).

    Raises:
        InsufficientDataError: if either the condition or the control has no tubes.
    """
    torch.manual_seed(seed)
    dev = resolve_device(device)

    sub = tube_set.filter(conditions=[condition, control_label])
    present = set(t.condition for t in sub.tubes)
    if condition not in present or control_label not in present:
        raise InsufficientDataError(
            f"train_binary_mil: need tubes for both {condition!r} and the control "
            f"{control_label!r}; found {sorted(present)}."
        )

    label_encoder = BinaryLabel(positive=condition, negative=control_label)
    dataset = InMemoryTubeDataset(sub, label_encoder)

    attention = AttentionModule(
        embed_dim=encoder.embed_dim, attention_hidden_dim=attention_hidden_dim
    )
    classifier = BagClassifier(embed_dim=encoder.embed_dim, n_classes=2)
    model = AbMil(encoder, attention, classifier, encoder_frozen=encoder_frozen).to(dev)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    queues = _build_class_queues(dataset.get_entries(), label_encoder)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        megabatches = _epoch_megabatches(queues, rng)
        for mb in megabatches:
            tubes = {cls: dataset[idx][:2] for cls, idx in mb.items()}
            epoch_loss += _train_one_megabatch(
                model, optimizer, tubes, criterion, dev
            )
        if (
            checkpoint_every
            and on_checkpoint is not None
            and epoch % checkpoint_every == 0
        ):
            model.eval()
            on_checkpoint(epoch, model)
            model.train()
        if verbose and (epoch % 50 == 0 or epoch == 1):
            logger.info(
                "[Stage 2 %s] epoch %d/%d loss=%.5f",
                condition,
                epoch,
                epochs,
                epoch_loss / max(len(megabatches), 1),
            )
    model.eval()
    return model


def train_all_binary(
    tube_set: PseudoTubeSet,
    encoder: InstanceEncoder,
    *,
    conditions: list[str] | None = None,
    control_label: str | None = None,
    device: str | torch.device | None = None,
    checkpoint_every: int | None = None,
    on_checkpoint_factory: Callable[[str], Callable[[int, AbMil], None]] | None = None,
    **kwargs,
) -> dict[str, AbMil]:
    """Train one binary AB-MIL per stimulus condition.

    Args:
        tube_set: The full pseudo-tube set.
        encoder: Shared trained encoder.
        conditions: Stimuli to model (default: all non-control conditions).
        control_label: Override the set's control label if needed.
        device: Where to train.
        checkpoint_every: OPT-IN recurrent-IG interval, forwarded to each
            :func:`train_binary_mil`. ``None`` (default) = unchanged behavior.
        on_checkpoint_factory: ``condition -> (epoch, model) -> None``. Builds a fresh
            per-condition checkpoint callback so each model's trajectory is captured
            separately (see :mod:`cascadir.dynamics`). ``None`` = no checkpointing.
        **kwargs: Forwarded to :func:`train_binary_mil`.

    Returns:
        ``{condition: AbMil}``.
    """
    ctrl = control_label or tube_set.control_label
    conds = conditions if conditions is not None else list(tube_set.stimulus_conditions)
    models: dict[str, AbMil] = {}
    for cond in conds:
        logger.info("train_all_binary: training %s vs %s", cond, ctrl)
        cb = on_checkpoint_factory(cond) if on_checkpoint_factory is not None else None
        models[cond] = train_binary_mil(
            tube_set,
            cond,
            encoder,
            control_label=ctrl,
            device=device,
            checkpoint_every=checkpoint_every,
            on_checkpoint=cb,
            **kwargs,
        )
    return models


# ---------------------------------------------------------------------------
# Multiclass AB-MIL (Path A — feeds the latent-geometry coupling discovery)
# ---------------------------------------------------------------------------


def train_multiclass_mil(
    tube_set: PseudoTubeSet,
    encoder: InstanceEncoder,
    *,
    attention_hidden_dim: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    momentum: float = 0.9,
    encoder_frozen: bool = True,
    device: str | torch.device | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[AbMil, MultiLabel]:
    """Train a multiclass AB-MIL over **all** conditions (including the control).

    This is the Path A model. With ``encoder_frozen=True`` (default) the cell
    embeddings used for coupling discovery equal the Stage-1 encoder's — which is the
    regime the validated 121-axis result used. Set ``encoder_frozen=False`` to
    fine-tune the encoder (sharper but slower geometry).

    Args:
        tube_set: The full pseudo-tube set.
        encoder: The shared Stage-1 encoder.
        attention_hidden_dim / epochs / lr / momentum: Stage-2 schedule.
        encoder_frozen: Freeze the encoder during multiclass training.
        device / seed / verbose: Runtime controls.

    Returns:
        ``(model, multilabel)`` — the trained model and its :class:`MultiLabel` encoder.
    """
    torch.manual_seed(seed)
    dev = resolve_device(device)
    multilabel = MultiLabel(tube_set.conditions)
    dataset = InMemoryTubeDataset(tube_set, multilabel)

    attention = AttentionModule(
        embed_dim=encoder.embed_dim, attention_hidden_dim=attention_hidden_dim
    )
    classifier = BagClassifier(
        embed_dim=encoder.embed_dim, n_classes=multilabel.n_classes()
    )
    model = AbMil(encoder, attention, classifier, encoder_frozen=encoder_frozen).to(dev)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    queues = _build_class_queues(dataset.get_entries(), multilabel)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        megabatches = _epoch_megabatches(queues, rng)
        for mb in megabatches:
            tubes = {cls: dataset[idx][:2] for cls, idx in mb.items()}
            epoch_loss += _train_one_megabatch(model, optimizer, tubes, criterion, dev)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            logger.info(
                "[multiclass] epoch %d/%d loss=%.4f",
                epoch,
                epochs,
                epoch_loss / max(len(megabatches), 1),
            )
    model.eval()
    return model, multilabel
