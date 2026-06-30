"""
Stage 2/3: train the full CytokineABMIL model with mega-batch gradient accumulation.

Returns a dynamics dict containing per-tube learning trajectories that are
consumed by cytokine_mil.analysis.dynamics for cascade inference.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.models.cytokine_abmil import CytokineABMIL
from cytokine_mil.models.cytokine_abmil_v2 import CytokineABMIL_V2
from cytokine_mil.training.trainer import (
    build_cytokine_queues,
    generate_epoch_megabatches,
)


def train_mil(
    model,
    dataset: PseudoTubeDataset,
    n_epochs: int,
    lr: float = 0.01,
    momentum: float = 0.9,
    encoder_lr_factor: float = 1.0,
    lr_scheduler: Optional[str] = None,
    lr_warmup_epochs: int = 0,
    log_every_n_epochs: int = 1,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True,
    val_dataset: Optional[PseudoTubeDataset] = None,
    kl_lambda: float = 0.0,
    aux_loss_weight: float = 0.0,
    checkpoint_dir: Optional[str] = None,
    checkpoint_epochs: Optional[List[int]] = None,
    cell_type_obs: Optional[Dict[int, List[str]]] = None,
    pbs_ct_means: Optional[Dict[str, np.ndarray]] = None,
    centroid_log_every_n_epochs: int = 10,
    attn_entropy_lambda: float = 0.0,
    exclude_cell_types: Optional[set] = None,
) -> Dict:
    """
    Train the CytokineABMIL model and record dynamics trajectories.

    Uses mega-batch gradient accumulation: one tube per cytokine per step,
    gradients accumulated across all cytokines, optimizer stepped once.
    SGD with momentum is used to produce smooth, monotonic learning curves
    suitable for dynamics analysis.

    Args:
        model: CytokineABMIL (encoder may be frozen for Stage 2).
        dataset: PseudoTubeDataset.
        n_epochs: Training epochs.
        lr: SGD learning rate.
        momentum: SGD momentum.
        lr_scheduler: Optional scheduler type ('cosine' or None).
        lr_warmup_epochs: Linear warmup epochs (0 to disable).
        log_every_n_epochs: Frequency of dynamics snapshots.
        device: Target device.
        seed: RNG seed for reproducible mega-batch sampling.
        verbose: Print per-epoch loss.
        val_dataset: Optional held-out donor dataset for validation logging.
        kl_lambda: Weight for KL(a_CA || a_SA) regularization term (v2 only).
        aux_loss_weight: Weight for auxiliary SA and CA classification losses (v2 only).
        cell_type_obs: Optional dict mapping tube-index (same ordering as
            dataset.get_entries()) to a list of cell-type strings (one per
            cell). When provided together with pbs_ct_means, centroid
            snapshots are computed in PBS-RC space at every
            centroid_log_every_n_epochs step.
        pbs_ct_means: Optional dict mapping cell-type name to the per-cell-
            type PBS mean embedding (np.ndarray, shape (embed_dim,)).
            Used to compute h̃_i = h_i - µ_{PBS,τ(i)} before centroid
            accumulation.
        centroid_log_every_n_epochs: Centroid snapshots are added at every
            Nth dynamics-log step. Must be >= log_every_n_epochs. Default 10.
    Returns:
        dynamics: dict with keys:
            'logged_epochs': list of epoch indices where dynamics were recorded.
            'records': list of per-tube dicts, each containing:
                'p_correct_trajectory':          list[float], shape (n_logged_epochs,)
                'entropy_trajectory':            list[float], shape (n_logged_epochs,)
                'instance_confidence_trajectory': ndarray, shape (n_cells, n_logged_epochs)
            'confusion_entropy_trajectory': dict mapping cytokine_name ->
                ndarray of shape (n_logged_epochs,).
                H_confusion(C,t) = -sum_{k!=C} q_k(t) log q_k(t),
                where q_k is the renormalized off-diagonal mean softmax score
                across all pseudo-tubes of cytokine C at epoch t.
            'val_records': list of per-tube dicts for held-out val donors, same
                structure as 'records'. Empty list if val_dataset is None.
            'val_confusion_entropy_trajectory': dict mapping cytokine_name ->
                ndarray of shape (n_logged_epochs,), same computation as
                confusion_entropy_trajectory but on val donors.
                Empty dict if val_dataset is None.
            'loss_components': dict with per-epoch loss component lists:
                'total': total loss per epoch (all models).
                'main': combined-head classification loss (v2 only, else []).
                'sa_aux': SA auxiliary classification loss (v2 only, else []).
                'ca_aux': CA auxiliary classification loss (v2 only, else []).
                'kl': KL(a_CA || a_SA) loss (v2 only, else []).
            'centroid_trajectory': list of centroid snapshots, one per centroid
                log step. Each snapshot is a dict mapping
                (cytokine, cell_type, donor) -> np.ndarray of shape (embed_dim,)
                representing the mean PBS-RC embedding of that cell-type
                subpopulation. Empty list if cell_type_obs or pbs_ct_means is None.
            'centroid_logged_epochs': list of epoch indices at which centroid
                snapshots were recorded. Empty list if not logging centroids.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, lr, momentum, encoder_lr_factor)
    scheduler = _build_scheduler(optimizer, n_epochs, lr_scheduler, lr_warmup_epochs)
    rng = np.random.default_rng(seed)

    # Checkpoint setup
    import pathlib
    _ckpt_dir = pathlib.Path(checkpoint_dir) if checkpoint_dir is not None else None
    _ckpt_epochs = set(checkpoint_epochs) if checkpoint_epochs is not None else set()
    if _ckpt_dir is not None:
        _ckpt_dir.mkdir(parents=True, exist_ok=True)

    entries = dataset.get_entries()
    queues = build_cytokine_queues(entries, dataset.label_encoder)

    logged_epochs: List[int] = []
    tube_trajectories: Dict[int, Dict] = _init_tube_trajectories(entries)
    cytokine_confusion_epochs: Dict[str, List[float]] = defaultdict(list)

    val_entries = val_dataset.get_entries() if val_dataset is not None else []
    val_tube_trajectories: Dict[int, Dict] = _init_tube_trajectories(val_entries)
    val_cytokine_confusion_epochs: Dict[str, List[float]] = defaultdict(list)

    loss_components: Dict[str, List[float]] = {
        "total": [], "main": [], "sa_aux": [], "ca_aux": [], "kl": [], "attn_kl": []
    }
    exclude_cell_types = set(exclude_cell_types) if exclude_cell_types else None

    _do_centroids = cell_type_obs is not None and pbs_ct_means is not None
    centroid_trajectory: List[Dict] = []
    attn_centroid_trajectory: List[Dict] = []
    centroid_logged_epochs: List[int] = []

    for epoch in range(1, n_epochs + 1):
        # Apply LR warmup BEFORE training each epoch so epoch 1 starts
        # at a small LR (scale = 1/warmup_epochs), not the full LR.
        if lr_warmup_epochs > 0 and epoch <= lr_warmup_epochs:
            _apply_warmup(optimizer, lr, epoch, lr_warmup_epochs)

        epoch_loss_dict = _train_epoch(
            model, dataset, queues, optimizer, criterion, device, rng,
            kl_lambda=kl_lambda, aux_loss_weight=aux_loss_weight,
            cell_type_obs=cell_type_obs, exclude_cell_types=exclude_cell_types,
            attn_entropy_lambda=attn_entropy_lambda,
        )
        epoch_loss = epoch_loss_dict["total"]
        loss_components["total"].append(epoch_loss)
        for key in ("main", "sa_aux", "ca_aux", "kl", "attn_kl"):
            if key in epoch_loss_dict:
                loss_components[key].append(epoch_loss_dict[key])

        if lr_warmup_epochs > 0 and epoch <= lr_warmup_epochs:
            pass  # warmup already applied above
        elif scheduler is not None:
            scheduler.step()

        if epoch % log_every_n_epochs == 0 or epoch == n_epochs:
            snap_centroids_now = _do_centroids and (
                epoch % centroid_log_every_n_epochs == 0 or epoch == n_epochs
            )
            centroid_snap = _log_dynamics(
                model, dataset, entries, tube_trajectories,
                cytokine_confusion_epochs, dataset.label_encoder, device,
                cell_type_obs=cell_type_obs,
                pbs_ct_means=pbs_ct_means if snap_centroids_now else None,
                exclude_cell_types=exclude_cell_types,
            )
            if val_dataset is not None:
                _log_dynamics(
                    model, val_dataset, val_entries, val_tube_trajectories,
                    val_cytokine_confusion_epochs, val_dataset.label_encoder, device,
                    exclude_cell_types=exclude_cell_types,
                )
            logged_epochs.append(epoch)
            if centroid_snap is not None:
                centroid_trajectory.append(centroid_snap["encoder"])
                attn_centroid_trajectory.append(centroid_snap["attn_weighted"])
                centroid_logged_epochs.append(epoch)

        # Save checkpoint if requested
        if _ckpt_dir is not None and epoch in _ckpt_epochs:
            ckpt_path = _ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)

        if verbose:
            print(f"[Stage 2/3] Epoch {epoch:3d}/{n_epochs} | loss={epoch_loss:.4f}"
                  + (f" | main={epoch_loss_dict['main']:.4f}"
                     f" sa={epoch_loss_dict['sa_aux']:.4f}"
                     f" ca={epoch_loss_dict['ca_aux']:.4f}"
                     f" kl={epoch_loss_dict['kl']:.4f}"
                     if "main" in epoch_loss_dict else ""))

    records = _build_records(entries, tube_trajectories)
    confusion_traj = {
        cyt: np.array(epochs) for cyt, epochs in cytokine_confusion_epochs.items()
    }
    val_records = _build_records(val_entries, val_tube_trajectories)
    val_confusion_traj = {
        cyt: np.array(epochs) for cyt, epochs in val_cytokine_confusion_epochs.items()
    }
    return {
        "logged_epochs": logged_epochs,
        "records": records,
        "confusion_entropy_trajectory": confusion_traj,
        "val_records": val_records,
        "val_confusion_entropy_trajectory": val_confusion_traj,
        "loss_components": loss_components,
        "centroid_trajectory": centroid_trajectory,
        "attn_centroid_trajectory": attn_centroid_trajectory,
        "centroid_logged_epochs": centroid_logged_epochs,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_optimizer(
    model, lr: float, momentum: float, encoder_lr_factor: float = 1.0
) -> torch.optim.SGD:
    encoder_frozen = getattr(model, "encoder_frozen", True)
    if encoder_lr_factor != 1.0 and not encoder_frozen:
        # Differential LR: encoder gets lr * encoder_lr_factor, MIL head gets lr.
        encoder_param_ids = {id(p) for p in model.encoder.parameters()}
        enc_params   = [p for p in model.parameters()
                        if p.requires_grad and id(p) in encoder_param_ids]
        head_params  = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in encoder_param_ids]
        param_groups = [
            {"params": enc_params,  "lr": lr * encoder_lr_factor, "base_lr": lr * encoder_lr_factor},
            {"params": head_params, "lr": lr,                      "base_lr": lr},
        ]
        return torch.optim.SGD(param_groups, momentum=momentum)
    return torch.optim.SGD(
        [{"params": p, "lr": lr, "base_lr": lr}
         for p in filter(lambda p: p.requires_grad, model.parameters())],
        momentum=momentum,
    )


def _build_scheduler(optimizer, n_epochs, scheduler_type, warmup_epochs):
    if scheduler_type == "cosine":
        effective_epochs = max(n_epochs - warmup_epochs, 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_epochs)
    return None


def _apply_warmup(optimizer, base_lr: float, epoch: int, warmup_epochs: int) -> None:
    scale = epoch / warmup_epochs
    for pg in optimizer.param_groups:
        pg["lr"] = pg.get("base_lr", base_lr) * scale


def _train_epoch(
    model,
    dataset: PseudoTubeDataset,
    queues: Dict[int, List[int]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    rng: np.random.Generator,
    kl_lambda: float = 0.0,
    aux_loss_weight: float = 0.0,
    cell_type_obs: Optional[Dict[int, List[str]]] = None,
    exclude_cell_types: Optional[set] = None,
    attn_entropy_lambda: float = 0.0,
) -> Dict[str, float]:
    """
    Run one epoch of mega-batch training.

    Returns a dict with loss components:
        'total': mean total loss over megabatches.
    For v2 models additionally:
        'main': classification loss on combined representation.
        'sa_aux': auxiliary SA classification loss.
        'ca_aux': auxiliary CA classification loss.
        'kl': KL(a_CA || a_SA) regularization loss.
    """
    model.train()
    megabatches = generate_epoch_megabatches(queues, rng)
    total_loss = 0.0
    total_main = 0.0
    total_sa = 0.0
    total_ca = 0.0
    total_kl = 0.0
    total_attn_kl = 0.0
    is_v2 = isinstance(model, CytokineABMIL_V2)
    n_mb = max(len(megabatches), 1)

    for mb_indices in tqdm(megabatches, leave=False):
        optimizer.zero_grad()
        mb_loss = 0.0
        mb_main = 0.0
        mb_sa = 0.0
        mb_ca = 0.0
        mb_kl = 0.0
        mb_attn_kl = 0.0
        n = len(mb_indices)

        for _cyt_idx, ds_idx in mb_indices.items():
            X, label, _donor, _cyt_name = dataset[ds_idx]
            if exclude_cell_types and cell_type_obs is not None:
                X = _apply_exclude_mask(X, cell_type_obs.get(ds_idx), exclude_cell_types)
            X = X.to(device)
            label_t = torch.tensor([label], dtype=torch.long, device=device)

            if is_v2:
                y_hat, a_SA, a_CA, _H, y_hat_sa, y_hat_ca = model.forward_with_aux(X)
                loss_main = criterion(y_hat.unsqueeze(0), label_t) / n
                loss_sa = criterion(y_hat_sa.unsqueeze(0), label_t) / n
                loss_ca = criterion(y_hat_ca.unsqueeze(0), label_t) / n
                # KL(a_CA || a_SA): penalize CA for deviating from SA.
                # F.kl_div(log(Q), P) = sum(P * log(P/Q)) = KL(P||Q)
                # Here P=a_CA, Q=a_SA -> input=log(a_SA), target=a_CA
                loss_kl = F.kl_div(
                    (a_SA + 1e-8).log(), a_CA, reduction="batchmean"
                ) / n
                loss = (
                    loss_main
                    + aux_loss_weight * loss_sa
                    + aux_loss_weight * loss_ca
                    + kl_lambda * loss_kl
                )
                mb_main += loss_main.item()
                mb_sa += loss_sa.item()
                mb_ca += loss_ca.item()
                mb_kl += loss_kl.item()
            else:
                y_hat, a, _H = model(X)
                loss = criterion(y_hat.unsqueeze(0), label_t) / n
                if attn_entropy_lambda > 0:
                    peak = _attn_peakedness(a)
                    loss = loss + attn_entropy_lambda * peak / n
                    mb_attn_kl += float(peak.detach().item())
            # Guard against NaN loss (e.g. from corrupted input or early
            # gradient explosion) — skip backward for this tube rather than
            # propagating NaN into the weight tensors.
            if torch.isnan(loss):
                continue
            loss.backward()
            mb_loss += loss.item()

        # Gradient clipping: essential when the encoder is unfrozen (Stage 3),
        # prevents first-step gradient explosion from setting weights to NaN.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += mb_loss
        total_main += mb_main
        total_sa += mb_sa
        total_ca += mb_ca
        total_kl += mb_kl
        total_attn_kl += mb_attn_kl

    result: Dict[str, float] = {"total": total_loss / n_mb}
    if is_v2:
        result["main"] = total_main / n_mb
        result["sa_aux"] = total_sa / n_mb
        result["ca_aux"] = total_ca / n_mb
        result["kl"] = total_kl / n_mb
    else:
        result["attn_kl"] = total_attn_kl / n_mb
    return result


def _init_tube_trajectories(entries: List[dict]) -> Dict[int, Dict]:
    """Create an empty trajectory dict keyed by dataset index."""
    return {
        i: {
            "p_correct": [],
            "entropy": [],
            "entropy_ca": [],  # v2-only: CA attention entropy
            "instance_confidence_epochs": [],
            # v2-only: per-epoch SA and CA confidence vectors; empty unless v2 model.
            "instance_confidence_sa_epochs": [],
            "instance_confidence_ca_epochs": [],
            "softmax_epochs": [],
        }
        for i in range(len(entries))
    }


@torch.no_grad()
def _log_dynamics(
    model,
    dataset: PseudoTubeDataset,
    entries: List[dict],
    tube_trajectories: Dict[int, Dict],
    cytokine_confusion_epochs: Dict[str, List[float]],
    label_encoder,
    device: torch.device,
    cell_type_obs: Optional[Dict[int, List[str]]] = None,
    pbs_ct_means: Optional[Dict[str, np.ndarray]] = None,
    exclude_cell_types: Optional[set] = None,
) -> Optional[Dict]:
    """
    Evaluate all tubes and append one snapshot to each tube's trajectory.

    Per tube (inside loop):
      - p_correct_trajectory: softmax probability of the correct class
      - entropy_trajectory: H = -sum(a_i * log(a_i)) over SA attention weights
      - instance_confidence_trajectory: C_i = a_i * p_correct per cell
      - softmax_epochs (internal): full K-dim softmax, used for confusion entropy

    For CytokineABMIL_V2 additionally:
      - instance_confidence_sa_epochs: C_SA_i = a_SA_i * p_correct per cell
      - instance_confidence_ca_epochs: C_CA_i = a_CA_i * p_correct per cell

    Per cytokine (after loop):
      - confusion_entropy_trajectory: H_confusion computed from mean off-diagonal
        softmax across all tubes of that cytokine.

    When cell_type_obs and pbs_ct_means are provided, also accumulates per-
    (cytokine, cell_type, donor) mean embeddings in PBS-RC space and returns
    them as a centroid snapshot dict. Returns None otherwise.

    Runs in eval mode with no_grad to avoid memory accumulation.
    """
    model.eval()
    is_v2 = isinstance(model, CytokineABMIL_V2)
    do_centroids = cell_type_obs is not None and pbs_ct_means is not None
    ct_sums: Dict = {}
    ct_counts: Dict = {}
    at_sums: Dict = {}   # attention-weighted: sum of a_i * h_tilde_i per (cyto, ct, donor)
    at_counts: Dict = {}

    for idx, entry in enumerate(entries):
        X, label, _donor, _cyt_name = dataset[idx]
        # Cell-type exclusion (data hygiene): mask X (and the cell-type labels used
        # for centroids) so eval matches what the model trained on.
        ct_labels_eff = cell_type_obs.get(idx) if cell_type_obs is not None else None
        if (exclude_cell_types and ct_labels_eff is not None
                and len(ct_labels_eff) == X.shape[0]):
            keep = [str(ct) not in exclude_cell_types for ct in ct_labels_eff]
            if any(keep) and not all(keep):
                X = X[torch.tensor(keep, dtype=torch.bool)]
                ct_labels_eff = [c for c, k in zip(ct_labels_eff, keep) if k]
        X = X.to(device)

        if is_v2:
            y_hat, a_SA, a_CA, H = model(X)
            probs = F.softmax(y_hat, dim=0)
            p_correct = probs[label].item()
            # SA entropy and CA entropy tracked independently.
            entropy = _compute_entropy(a_SA)
            entropy_ca = _compute_entropy(a_CA)
            instance_conf = (a_SA * p_correct).cpu().numpy()
            instance_conf_sa = instance_conf  # same as above; explicit for clarity
            instance_conf_ca = (a_CA * p_correct).cpu().numpy()
            a_for_centroids = a_SA  # use SA attention for centroid weighting
        else:
            y_hat, a, H = model(X)
            probs = F.softmax(y_hat, dim=0)
            p_correct = probs[label].item()
            entropy = _compute_entropy(a)
            instance_conf = (a * p_correct).cpu().numpy()
            a_for_centroids = a

        traj = tube_trajectories[idx]
        traj["p_correct"].append(p_correct)
        traj["entropy"].append(entropy)
        traj["instance_confidence_epochs"].append(instance_conf)
        traj["softmax_epochs"].append(probs.cpu().numpy())

        if is_v2:
            traj["entropy_ca"].append(entropy_ca)
            traj["instance_confidence_sa_epochs"].append(instance_conf_sa)
            traj["instance_confidence_ca_epochs"].append(instance_conf_ca)

        if do_centroids:
            ct_labels = ct_labels_eff or []
            if ct_labels:
                H_np = H.cpu().numpy().astype(np.float64)       # (N_cells, embed_dim)
                a_np = a_for_centroids.cpu().numpy().astype(np.float64)  # (N_cells,)
                cytokine = entry["cytokine"]
                donor = entry["donor"]
                for i, ct in enumerate(ct_labels):
                    ct_str = str(ct)
                    pbs_mean = pbs_ct_means.get(ct_str)
                    h_tilde = H_np[i] - pbs_mean if pbs_mean is not None else H_np[i]
                    key = (cytokine, ct_str, donor)
                    if key not in ct_sums:
                        ct_sums[key] = np.zeros(H_np.shape[1], dtype=np.float64)
                        ct_counts[key] = 0
                        at_sums[key] = np.zeros(H_np.shape[1], dtype=np.float64)
                        at_counts[key] = 0
                    ct_sums[key] += h_tilde
                    ct_counts[key] += 1
                    # Attention-weighted: a_i * h_tilde_i (PBS-RC space, attention-scaled)
                    at_sums[key] += a_np[i] * h_tilde
                    at_counts[key] += 1

    _compute_confusion_entropy_snapshot(
        entries, tube_trajectories, label_encoder, cytokine_confusion_epochs
    )

    if do_centroids:
        encoder_snap = {
            key: ct_sums[key] / ct_counts[key]
            for key in ct_sums if ct_counts[key] > 0
        }
        attn_snap = {
            key: at_sums[key] / at_counts[key]
            for key in at_sums if at_counts[key] > 0
        }
        return {"encoder": encoder_snap, "attn_weighted": attn_snap}
    return None


def _compute_confusion_entropy_snapshot(
    entries: List[dict],
    tube_trajectories: Dict[int, Dict],
    label_encoder,
    cytokine_confusion_epochs: Dict[str, List[float]],
) -> None:
    """
    Compute one confusion-entropy snapshot per cytokine from the latest softmax snapshot.

    Steps per cytokine C:
      1. Collect latest softmax ŷ_b ∈ R^K for each tube b of C.
      2. Average across tubes: ȳ_C ∈ R^K.
      3. Remove the true class k=C and renormalize: q_k = ȳ_{C,k} / sum_{j≠C} ȳ_{C,j}.
      4. H_confusion(C) = -sum_{k≠C} q_k log(q_k).

    Appended in-place to cytokine_confusion_epochs[cytokine_name].
    """
    cytokine_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        cytokine_to_indices[entry["cytokine"]].append(idx)

    for cytokine, indices in cytokine_to_indices.items():
        true_label = label_encoder.encode(cytokine)
        softmaxes = np.stack(
            [tube_trajectories[idx]["softmax_epochs"][-1] for idx in indices]
        )  # (n_tubes, K)
        mean_softmax = softmaxes.mean(axis=0)  # (K,)

        K = len(mean_softmax)
        off_diag = np.concatenate(
            [mean_softmax[:true_label], mean_softmax[true_label + 1:]]
        )
        off_sum = float(off_diag.sum())
        if off_sum < 1e-10:
            cytokine_confusion_epochs[cytokine].append(0.0)
            continue
        q = off_diag / off_sum
        q_safe = np.clip(q, 1e-10, None)
        entropy = -float((q_safe * np.log(q_safe)).sum())
        cytokine_confusion_epochs[cytokine].append(entropy)


def _compute_entropy(a: torch.Tensor) -> float:
    """Shannon entropy of attention weights (nats). Clipped for stability."""
    a_safe = a.clamp(min=1e-10)
    return float(-(a_safe * a_safe.log()).sum())


def _attn_peakedness(a: torch.Tensor) -> torch.Tensor:
    """
    Normalized attention peakedness: 1 - H(a)/log(N), differentiable in a.

    0 = perfectly uniform attention; ->1 = fully concentrated on one cell.
    Minimizing it (the entropy penalty) flattens attention and damps the
    late-training collapse onto rare discriminative cell types. Normalizing by
    log(N) makes the penalty tube-size-invariant (N varies per pseudo-tube).
    """
    n = a.numel()
    if n <= 1:
        return a.sum() * 0.0
    a_safe = a.clamp(min=1e-10)
    ent = -(a_safe * a_safe.log()).sum()
    return 1.0 - ent / math.log(n)


def _apply_exclude_mask(X: torch.Tensor, ct_labels, exclude_cell_types) -> torch.Tensor:
    """Drop rows of X whose cell type is in exclude_cell_types (data hygiene).
    No-op if labels are missing/misaligned or the mask would drop all/no cells."""
    if not ct_labels or len(ct_labels) != X.shape[0]:
        return X
    keep = [str(ct) not in exclude_cell_types for ct in ct_labels]
    if all(keep) or not any(keep):
        return X
    keep_t = torch.tensor(keep, dtype=torch.bool)
    return X[keep_t]


def _build_records(
    entries: List[dict], tube_trajectories: Dict[int, Dict]
) -> List[Dict]:
    """Combine manifest metadata with trajectory data into flat records."""
    records = []
    for idx, entry in enumerate(entries):
        traj = tube_trajectories[idx]
        ic_epochs = traj["instance_confidence_epochs"]
        if ic_epochs:
            # Stack to (n_logged_epochs, n_cells), transpose to (n_cells, n_logged_epochs)
            ic_trajectory = np.stack(ic_epochs, axis=0).T
        else:
            ic_trajectory = None

        # v2-only SA/CA trajectories — present only when the model is CytokineABMIL_V2.
        ic_sa_epochs = traj.get("instance_confidence_sa_epochs", [])
        ic_ca_epochs = traj.get("instance_confidence_ca_epochs", [])
        ic_sa_trajectory = np.stack(ic_sa_epochs, axis=0).T if ic_sa_epochs else None
        ic_ca_trajectory = np.stack(ic_ca_epochs, axis=0).T if ic_ca_epochs else None

        # Full softmax trajectory: shape (K, n_logged_epochs).
        # Used by confusion_dynamics.py to build the K×K×T confusion trajectory tensor.
        sm_epochs = traj["softmax_epochs"]
        softmax_trajectory = np.stack(sm_epochs, axis=0).T if sm_epochs else None

        records.append(
            {
                "cytokine": entry["cytokine"],
                "donor": entry["donor"],
                "tube_idx": entry["tube_idx"],
                "tube_path": entry["path"],
                "n_cells": entry["n_cells"],
                "p_correct_trajectory": traj["p_correct"],
                "entropy_trajectory": traj["entropy"],
                # v2 model: CA attention entropy trajectory. None for v1 model.
                "entropy_trajectory_ca": traj["entropy_ca"] if traj["entropy_ca"] else None,
                # Full trajectory: shape (n_cells, n_logged_epochs).
                # C_i(t) = a_i(t) * P(t)(Y_correct).
                # Do not collapse — aggregation happens in analysis layer.
                "instance_confidence_trajectory": ic_trajectory,
                # v2 model: SA and CA layer trajectories. None for v1 model.
                "confidence_trajectory_sa": ic_sa_trajectory,
                "confidence_trajectory_ca": ic_ca_trajectory,
                # Full softmax output per tube per logged epoch.
                # Shape: (K, n_logged_epochs). K = n_classes (e.g. 91).
                # Consumed by analysis/confusion_dynamics.py to build the
                # K×K×T confusion trajectory tensor C(A,B,t).
                "softmax_trajectory": softmax_trajectory,
            }
        )
    return records
