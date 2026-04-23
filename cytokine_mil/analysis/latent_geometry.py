"""
Latent Space Cytokine Geometry Analysis.

After MIL training the encoder maps each cell to h_i ∈ R^128. We hypothesize
that cells in cytokine-A tubes that are responding to endogenously produced
cytokine B (a cascade secondary signal) will have their embedding displaced
toward B's centroid in the learned representation space.

Cascade relationships are detected as per-cell-type directional bias of a
cell-type subpopulation's mean embedding toward another cytokine's centroid,
within a given cytokine's tubes — without using attention weights.

Experiments implemented here (per CLAUDE.md Section 20):
    Experiment 0 — Does cytokine geometry exist at the cell level? (GO/NO-GO)
    Experiment 1 — Per-cell-type directional bias toward other cytokine centroids
    Experiment 2 — Asymmetry test for cascade direction from bias scores

Functions:
    compute_cytokine_centroids
    compute_alignment_scores
    compute_directional_bias
    compute_asymmetry_matrix
    build_latent_cascade_graph
"""

# TODO: Experiment 3 — implement if Experiment 0 alignment ≈ null.
# Concept: small MLP trained post-hoc on frozen encoder output, supervised by
# bag-level softmax p_bag(C | tube) via KL divergence. Transfers bag-level
# cytokine geometry to cell level without requiring cell-level cytokine labels.
# See CLAUDE.md Section 20.5 for full spec.

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cytokine_mil.data.dataset import PseudoTubeDataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_cytokine_centroids(
    model: nn.Module,
    dataset: PseudoTubeDataset,
    label_encoder,
    device: str = "cpu",
    decoder: Optional[nn.Module] = None,
    encoder_space: bool = False,
) -> dict:
    """
    Compute mean cell embedding per cytokine across all tubes in dataset.

    The caller is responsible for passing a training-split dataset
    (training donors D1, D4–D12 only).

    For each cytokine C:
        μ_C = mean over all cells across all C-tubes in dataset.

    Cell-type labels are never used here — only the embeddings.
    Uses model.eval() and torch.no_grad() throughout.

    Args:
        model: CytokineABMIL or CytokineABMIL_V2 (eval mode, on device).
        dataset: PseudoTubeDataset (training donors only — caller responsibility).
        label_encoder: CytokineLabel or BinaryLabel with .encode() and .cytokines.
        device: torch device string.
        decoder: Optional AuxDecoder.
        encoder_space: If True, always compute centroids in h_i (encoder) space.
            When decoder is also provided, cell contributions are weighted by
            decoder_softmax(h_i)[C] — "how cytokine-C-like is this cell?".
            If False (default), use g_i = decoder.embed(h_i) when decoder is
            provided, otherwise h_i (original behaviour).

    Returns:
        dict with keys:
            'centroids': {cytokine_name -> np.ndarray of shape (embed_dim,)}
            'metric_description': str
    """
    torch_device = torch.device(device)
    model.eval()
    model.to(torch_device)
    if decoder is not None:
        decoder.eval()
        decoder.to(torch_device)

    embed_dim: Optional[int] = None
    embedding_sums: Dict[str, np.ndarray] = {}
    cell_counts: Dict[str, float] = defaultdict(float)

    with torch.no_grad():
        for ds_idx in range(len(dataset)):
            X, _label, _donor, cytokine_name = dataset[ds_idx]
            X = X.to(torch_device)

            if encoder_space:
                H = _extract_h_embeddings(model, X)          # (N, 128) always
                H_np = H.cpu().numpy().astype(np.float64)
                if embed_dim is None:
                    embed_dim = H_np.shape[1]
                    embedding_sums = defaultdict(
                        lambda: np.zeros(embed_dim, dtype=np.float64)
                    )
                if decoder is not None:
                    # Weight each cell by how cytokine-C-like it looks.
                    cyt_idx = label_encoder.encode(cytokine_name)
                    logits = decoder(H)                       # (N, K)
                    w = F.softmax(logits, dim=1)[:, cyt_idx] # (N,)
                    w_np = w.cpu().numpy().astype(np.float64)
                    embedding_sums[cytokine_name] += (w_np[:, None] * H_np).sum(axis=0)
                    cell_counts[cytokine_name] += float(w_np.sum())
                else:
                    embedding_sums[cytokine_name] += H_np.sum(axis=0)
                    cell_counts[cytokine_name] += float(H_np.shape[0])
            else:
                emb = _extract_embeddings(model, X, decoder=decoder)  # (N, D)
                if embed_dim is None:
                    embed_dim = emb.shape[1]
                    embedding_sums = defaultdict(
                        lambda: np.zeros(embed_dim, dtype=np.float64)
                    )
                emb_np = emb.cpu().numpy().astype(np.float64)
                embedding_sums[cytokine_name] += emb_np.sum(axis=0)
                cell_counts[cytokine_name] += float(emb_np.shape[0])

    centroids = {
        cyt: embedding_sums[cyt] / cell_counts[cyt]
        for cyt in embedding_sums
        if cell_counts[cyt] > 0
    }

    if encoder_space:
        if decoder is not None:
            space = "h_i (encoder), decoder-softmax weighted"
        else:
            space = "h_i (encoder), unweighted"
    else:
        space = "g_i (AuxDecoder, decoder-injected cytokine space)" \
                if decoder is not None else "h_i (encoder output)"

    return {
        "centroids": centroids,
        "metric_description": (
            f"μ_C = weighted mean over all cells in all C-tubes of {space}, "
            "training donors only"
        ),
    }


def compute_alignment_scores(
    model: nn.Module,
    dataset: PseudoTubeDataset,
    label_encoder,
    centroids: dict,
    n_permutations: int = 1000,
    device: str = "cpu",
    decoder: Optional[nn.Module] = None,
) -> dict:
    """
    Compute cytokine alignment scores with permutation-based null distribution.

    For each tube of cytokine A, for each cell i:
        f_i = softmax( -||h_i - μ_C||_2  for all C )
        per-tube score = mean_i f_i[A]

    When decoder is provided, h_i is replaced by g_i = decoder.embed(h_i).

    Cytokine alignment score for A = mean over all A-tubes of per-tube score.

    Null distribution: permute cytokine labels assigned to tubes 1000 times,
    recompute per-tube alignment as if the tube belonged to the permuted cytokine.

    Args:
        model: CytokineABMIL or CytokineABMIL_V2 (eval mode).
        dataset: PseudoTubeDataset (training donors only — caller responsibility).
        label_encoder: CytokineLabel or BinaryLabel.
        centroids: output of compute_cytokine_centroids()['centroids'].
        n_permutations: number of label permutations for null distribution.
        device: torch device string.
        decoder: Optional AuxDecoder. If provided, g_i = decoder.embed(h_i) is used.

    Returns:
        dict with keys:
            'alignment_scores': {cytokine_name -> float}  (mean over tubes)
            'null_mean': float
            'null_std': float
            'p_values': {cytokine_name -> float}  (fraction of null >= observed)
            'metric_description': str
    """
    torch_device = torch.device(device)
    model.eval()
    model.to(torch_device)
    if decoder is not None:
        decoder.eval()
        decoder.to(torch_device)

    centroid_names = list(centroids.keys())
    centroid_matrix = np.stack(
        [centroids[c] for c in centroid_names], axis=0
    ).astype(np.float32)
    centroid_tensor = torch.tensor(centroid_matrix, device=torch_device)

    # Per-tube: (cytokine_name, alignment_score)
    tube_scores: List[Tuple[str, float]] = []

    with torch.no_grad():
        for ds_idx in range(len(dataset)):
            X, _label, _donor, cytokine_name = dataset[ds_idx]
            if cytokine_name not in centroids:
                continue
            X = X.to(torch_device)
            H = _extract_embeddings(model, X, decoder=decoder)
            score = _per_tube_alignment(H, centroid_tensor, centroid_names, cytokine_name)
            tube_scores.append((cytokine_name, score))

    # Mean per cytokine across tubes.
    cyt_tube_scores: Dict[str, List[float]] = defaultdict(list)
    for cyt, score in tube_scores:
        cyt_tube_scores[cyt].append(score)

    alignment_scores = {
        cyt: float(np.mean(vals))
        for cyt, vals in cyt_tube_scores.items()
    }

    # Null distribution: permute tube labels, recompute alignment.
    null_scores = _compute_alignment_null(
        tube_scores, n_permutations=n_permutations, rng_seed=0
    )
    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores))

    # p-value: fraction of null mean >= observed per cytokine.
    p_values = {
        cyt: float(np.mean(np.array(null_scores) >= score))
        for cyt, score in alignment_scores.items()
    }

    space = "g_i (AuxDecoder)" if decoder is not None else "h_i"
    return {
        "alignment_scores": alignment_scores,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_values": p_values,
        "metric_description": (
            f"mean over cells in A-tubes of softmax(-||{space} - μ_C||_2 for all C)[A], "
            "aggregated to donor level (training donors only)"
        ),
    }


def compute_directional_bias(
    model: nn.Module,
    dataset: PseudoTubeDataset,
    label_encoder,
    centroids: dict,
    cell_type_obs: dict,
    n_permutations: int = 1000,
    device: str = "cpu",
    decoder: Optional[nn.Module] = None,
    encoder_space: bool = False,
) -> dict:
    """
    Compute per-cell-type directional bias of embeddings toward other cytokine centroids.

    For every (A, B, T) triple:
        bias(A, B, T) = (μ_{A,T} - μ_A) · (μ_B - μ_A) / ||μ_B - μ_A||_2

    Where:
        μ_{A,T} = mean embedding of cells of type T within A-tubes
        μ_A     = mean embedding of all cells within A-tubes (= centroids[A])
        (μ_B - μ_A) / ||...||_2 = unit direction from A centroid to B centroid

    When encoder_space=False and decoder is provided, h_i is replaced by
    g_i = decoder.embed(h_i).

    When encoder_space=True, h_i is always used. If decoder is also provided,
    each cell's contribution to μ_{A,T} is weighted by decoder_softmax(h_i)[A]
    — "how cytokine-A-like is this cell?" — matching the centroid weighting from
    compute_cytokine_centroids(encoder_space=True). Cell-type permutations for
    the null distribution permute only ct_labels; weights stay fixed per cell.

    Null: permute cell-type labels within each tube 1000 times, recompute bias.
    z-score: z(A, B, T) = (bias - mean(null)) / std(null).
    BH-FDR correction across all (A, B, T) triples.

    Cell-type labels are passed as cell_type_obs by the caller — never used during
    training or accessed from inside the model.

    Args:
        model: CytokineABMIL or CytokineABMIL_V2 (eval mode).
        dataset: PseudoTubeDataset (training donors only — caller responsibility).
        label_encoder: CytokineLabel or BinaryLabel.
        centroids: output of compute_cytokine_centroids()['centroids'].
        cell_type_obs: {tube_index (int) -> list of cell type strings, length N}.
        n_permutations: number of within-tube cell-type permutations for null.
        device: torch device string.
        decoder: Optional AuxDecoder.
        encoder_space: If True, compute bias in h_i space with optional decoder
            weighting (see above). If False, use g_i when decoder is provided.

    Returns:
        dict with keys:
            'bias': {(cyt_a, cyt_b, cell_type) -> float}
            'z_scores': {(cyt_a, cyt_b, cell_type) -> float}
            'q_values': {(cyt_a, cyt_b, cell_type) -> float}
            'metric_description': str
    """
    torch_device = torch.device(device)
    model.eval()
    model.to(torch_device)
    if decoder is not None:
        decoder.eval()
        decoder.to(torch_device)

    # tube_data: (cyt_name, H_np, ct_labels, cell_weights_or_None)
    # cell_weights (N,) = decoder_softmax[cyt_idx] per cell when encoder_space=True.
    tube_data: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []

    with torch.no_grad():
        for ds_idx in range(len(dataset)):
            X, _label, _donor, cytokine_name = dataset[ds_idx]
            if cytokine_name not in centroids:
                continue
            ct_labels = cell_type_obs.get(ds_idx)
            if ct_labels is None:
                continue
            X = X.to(torch_device)

            if encoder_space:
                H = _extract_h_embeddings(model, X)          # (N, 128) always
                H_np = H.cpu().numpy().astype(np.float64)
                if decoder is not None:
                    cyt_idx = label_encoder.encode(cytokine_name)
                    logits = decoder(H)                       # (N, K)
                    w = F.softmax(logits, dim=1)[:, cyt_idx] # (N,)
                    w_np = w.cpu().numpy().astype(np.float64)
                else:
                    w_np = None
            else:
                H = _extract_embeddings(model, X, decoder=decoder)
                H_np = H.cpu().numpy().astype(np.float64)
                w_np = None

            tube_data.append((cytokine_name, H_np, np.array(ct_labels), w_np))

    # Compute observed bias for all (A, B, T) triples.
    bias_obs = _compute_all_biases(tube_data, centroids)

    # Null: permute cell-type labels within tubes; weights stay fixed per cell.
    bias_null = _compute_bias_null(tube_data, centroids, n_permutations=n_permutations)

    # z-scores.
    z_scores = {}
    for key, obs in bias_obs.items():
        null_vals = bias_null.get(key, [])
        if len(null_vals) < 2:
            z_scores[key] = 0.0
        else:
            null_arr = np.array(null_vals)
            null_m = float(null_arr.mean())
            null_s = float(null_arr.std())
            z_scores[key] = (obs - null_m) / null_s if null_s > 0 else 0.0

    # Convert z-scores to two-tailed p-values and apply BH-FDR.
    q_values = _z_to_bh_qvalues(z_scores)

    if encoder_space:
        if decoder is not None:
            space = "h_i (encoder), decoder-softmax weighted"
        else:
            space = "h_i (encoder), unweighted"
    else:
        space = "g_i (AuxDecoder)" if decoder is not None else "h_i"

    return {
        "bias": dict(bias_obs),
        "z_scores": dict(z_scores),
        "q_values": q_values,
        "metric_description": (
            f"scalar projection of (μ_{{A,T}} - μ_A) [{space}] onto "
            "unit vector (μ_B - μ_A)/||μ_B - μ_A||_2"
        ),
    }


def compute_asymmetry_matrix(
    bias: dict,
    label_encoder,
) -> dict:
    """
    Compute the pair-level asymmetry matrix from per-cell-type directional bias.

    For each ordered pair (A, B):
        ASYM(A, B) = max_T [ bias(A,B,T) - bias(B,A,T) ]

    Positive ASYM(A, B) = evidence for cascade direction A→B.

    Args:
        bias: output of compute_directional_bias()['bias'].
              Keys are (cyt_a, cyt_b, cell_type) tuples.
        label_encoder: CytokineLabel or BinaryLabel with .cytokines.

    Returns:
        dict with keys:
            'asymmetry_matrix': np.ndarray of shape (K, K)
            'cytokine_names': list of length K, ordered to match matrix rows/cols
            'metric_description': str
    """
    cytokine_names = list(label_encoder.cytokines)
    K = len(cytokine_names)
    name_to_idx = {name: i for i, name in enumerate(cytokine_names)}

    # Collect all cell types seen in bias keys.
    cell_types = sorted({key[2] for key in bias.keys()})

    asym_matrix = np.zeros((K, K), dtype=np.float64)

    for a_name in cytokine_names:
        a_idx = name_to_idx[a_name]
        for b_name in cytokine_names:
            if a_name == b_name:
                continue
            b_idx = name_to_idx[b_name]
            max_asym = None
            for ct in cell_types:
                b_ab = bias.get((a_name, b_name, ct), 0.0)
                b_ba = bias.get((b_name, a_name, ct), 0.0)
                diff = b_ab - b_ba
                if max_asym is None or diff > max_asym:
                    max_asym = diff
            if max_asym is not None:
                asym_matrix[a_idx, b_idx] = max_asym

    return {
        "asymmetry_matrix": asym_matrix,
        "cytokine_names": cytokine_names,
        "metric_description": (
            "max_T [ bias(A,B,T) - bias(B,A,T) ]; positive = evidence for cascade A→B"
        ),
    }


def build_latent_cascade_graph(
    asymmetry_matrix: np.ndarray,
    z_scores: dict,
    label_encoder,
    fdr_alpha: float = 0.05,
) -> "nx.DiGraph":
    """
    Build a directed cytokine cascade graph from latent geometry asymmetry scores.

    Edge A→B is added when:
        max_T z(A, B, T) > FDR threshold (from q_values in z_scores)  AND
        ASYM(A, B) > 0

    Edge weight = max z-score over cell types for pair (A, B).

    Args:
        asymmetry_matrix: np.ndarray of shape (K, K) from compute_asymmetry_matrix().
        z_scores: output of compute_directional_bias()['z_scores'].
                  Keys are (cyt_a, cyt_b, cell_type) tuples.
        label_encoder: CytokineLabel or BinaryLabel with .cytokines.
        fdr_alpha: FDR significance threshold (default 0.05).

    Returns:
        nx.DiGraph with nodes = cytokine names, edges A→B with attributes:
            'asymmetry': ASYM(A, B) value
            'max_z': max z-score over cell types for pair (A, B)
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "networkx is required for build_latent_cascade_graph. "
            "Install with: pip install networkx"
        ) from exc

    cytokine_names = list(label_encoder.cytokines)
    K = len(cytokine_names)
    name_to_idx = {name: i for i, name in enumerate(cytokine_names)}

    # Compute FDR threshold from z_scores via BH correction.
    q_values = _z_to_bh_qvalues(z_scores)

    # Find max z-score per (A, B) pair and its FDR q-value.
    pair_max_z: Dict[Tuple[str, str], float] = {}
    pair_min_q: Dict[Tuple[str, str], float] = {}

    for (cyt_a, cyt_b, _ct), z in z_scores.items():
        pair = (cyt_a, cyt_b)
        if pair not in pair_max_z or z > pair_max_z[pair]:
            pair_max_z[pair] = z
            pair_min_q[pair] = q_values.get((cyt_a, cyt_b, _ct), 1.0)

    G = nx.DiGraph()
    for name in cytokine_names:
        G.add_node(name)

    for (cyt_a, cyt_b), max_z in pair_max_z.items():
        if cyt_a not in name_to_idx or cyt_b not in name_to_idx:
            continue
        a_idx = name_to_idx[cyt_a]
        b_idx = name_to_idx[cyt_b]
        asym = float(asymmetry_matrix[a_idx, b_idx])
        min_q = pair_min_q.get((cyt_a, cyt_b), 1.0)
        if asym > 0 and min_q <= fdr_alpha:
            G.add_edge(cyt_a, cyt_b, asymmetry=asym, max_z=float(max_z))

    return G


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_h_embeddings(
    model: nn.Module,
    X: torch.Tensor,
) -> torch.Tensor:
    """
    Extract h_i from model — always encoder output, ignores decoder.

    Handles both v1 (y_hat, a, H) and v2 (y_hat, a_SA, a_CA, H).
    X: (N, G) input tensor already on correct device.
    Returns H ∈ R^(N, 128).
    """
    outputs = model(X)
    if len(outputs) == 3:
        _, _, H = outputs
    elif len(outputs) == 4:
        _, _, _, H = outputs
    else:
        raise ValueError(
            f"Unexpected number of model outputs: {len(outputs)}. "
            "Expected 3 (v1) or 4 (v2)."
        )
    return H


def _extract_embeddings(
    model: nn.Module,
    X: torch.Tensor,
    decoder: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Run model forward pass and return cell embeddings.

    Handles both v1 (returns y_hat, a, H) and v2 (returns y_hat, a_SA, a_CA, H).
    X: (N, G) input tensor already on correct device.

    If decoder is None: returns H ∈ R^(N, 128) from the encoder.
    If decoder is provided: returns g_i = decoder.embed(H) ∈ R^(N, embed_dim).
      Use this for Experiment 3 (AuxDecoder) where cytokine geometry has been
      injected post-hoc into a lower-dimensional space.
    """
    H = _extract_h_embeddings(model, X)
    if decoder is not None:
        return decoder.embed(H)   # (N, embed_dim) — decoder-injected cytokine space
    return H                      # (N, 128)       — encoder space


def _per_tube_alignment(
    H: torch.Tensor,
    centroid_tensor: torch.Tensor,
    centroid_names: List[str],
    cytokine_name: str,
) -> float:
    """
    Compute alignment score for a single tube.

    f_i = softmax( -||h_i - μ_C||_2  for all C )
    score = mean_i f_i[A]

    H: (N, 128) on device.
    centroid_tensor: (C_count, 128) on same device.
    cytokine_name: true cytokine for this tube.
    Returns scalar float.
    """
    # Compute L2 distances from each cell to each centroid: (N, C_count)
    # ||h_i - μ_C||^2 = ||h_i||^2 + ||μ_C||^2 - 2 h_i μ_C^T
    dists = _l2_distances(H, centroid_tensor)  # (N, C_count)
    # Softmax over negative distances.
    affinities = F.softmax(-dists, dim=1)  # (N, C_count)

    if cytokine_name not in centroid_names:
        return 0.0
    cyt_idx = centroid_names.index(cytokine_name)
    score = float(affinities[:, cyt_idx].mean().item())
    return score


def _l2_distances(
    H: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise L2 distances between rows of H and centroids.

    H: (N, D)
    centroids: (C, D)
    Returns: (N, C) distance matrix.
    """
    # ||h - μ||^2 = ||h||^2 + ||μ||^2 - 2 h·μ^T
    h_sq = (H ** 2).sum(dim=1, keepdim=True)          # (N, 1)
    c_sq = (centroids ** 2).sum(dim=1, keepdim=True).T  # (1, C)
    cross = H @ centroids.T                            # (N, C)
    dist_sq = (h_sq + c_sq - 2 * cross).clamp(min=0.0)
    return dist_sq.sqrt()  # (N, C)


def _compute_alignment_null(
    tube_scores: List[Tuple[str, float]],
    n_permutations: int,
    rng_seed: int = 0,
) -> List[float]:
    """
    Build null distribution by permuting tube cytokine labels.

    For each permutation, randomly reassign cytokine names to tube scores and
    recompute the mean alignment score pooled across all (now mismatched) tubes.
    Returns list of n_permutations null mean scores.
    """
    rng = np.random.default_rng(rng_seed)
    scores_arr = np.array([s for _, s in tube_scores])
    null_means = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(scores_arr))
        null_means.append(float(scores_arr[perm].mean()))
    return null_means


def _compute_all_biases(
    tube_data: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    centroids: dict,
) -> Dict[Tuple[str, str, str], float]:
    """
    Compute observed bias(A, B, T) for all (A, B, T) triples.

    For each cytokine A:
        μ_{A,T} = (optionally weighted) mean embedding of cells of type T across
                  all A-tubes
        μ_A     = centroids[A]
        unit_AB = (μ_B - μ_A) / ||μ_B - μ_A||_2

        bias(A, B, T) = (μ_{A,T} - μ_A) · unit_AB

    tube_data: list of (cytokine_name, H (N,D), cell_types (N,), weights (N,) or None)
        When weights is None, uniform weighting is used.
    centroids: {cytokine_name -> np.ndarray (D,)}
    Returns: {(cyt_a, cyt_b, cell_type) -> float}
    """
    # Infer embedding dimension from first tube.
    if not tube_data:
        return {}
    embed_dim = tube_data[0][1].shape[1]

    # Aggregate per (cytokine, cell_type): sum of embeddings and count.
    cyt_ct_sum: Dict[Tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(embed_dim, dtype=np.float64)
    )
    cyt_ct_count: Dict[Tuple[str, str], float] = defaultdict(float)

    for cyt_name, H_np, ct_labels, cell_weights in tube_data:
        for ct in np.unique(ct_labels):
            mask = ct_labels == ct
            if cell_weights is not None:
                w = cell_weights[mask]                         # (n_ct,)
            else:
                w = np.ones(mask.sum(), dtype=np.float64)
            cyt_ct_sum[(cyt_name, ct)] += (w[:, None] * H_np[mask]).sum(axis=0)
            cyt_ct_count[(cyt_name, ct)] += float(w.sum())

    # Compute mean embeddings per (cytokine, cell_type).
    cyt_ct_mean: Dict[Tuple[str, str], np.ndarray] = {
        key: cyt_ct_sum[key] / cyt_ct_count[key]
        for key in cyt_ct_sum
        if cyt_ct_count[key] > 0
    }

    cytokines_in_data = {cyt for cyt, _ in cyt_ct_mean.keys()}
    cell_types_in_data = {ct for _, ct in cyt_ct_mean.keys()}

    bias: Dict[Tuple[str, str, str], float] = {}

    for cyt_a in cytokines_in_data:
        mu_a = centroids.get(cyt_a)
        if mu_a is None:
            continue
        for cyt_b in cytokines_in_data:
            if cyt_a == cyt_b:
                continue
            mu_b = centroids.get(cyt_b)
            if mu_b is None:
                continue
            direction = mu_b - mu_a
            norm = float(np.linalg.norm(direction))
            if norm < 1e-10:
                continue
            unit_ab = direction / norm
            for ct in cell_types_in_data:
                mu_at = cyt_ct_mean.get((cyt_a, ct))
                if mu_at is None:
                    continue
                displacement = mu_at - mu_a
                bias[(cyt_a, cyt_b, ct)] = float(np.dot(displacement, unit_ab))

    return bias


def _compute_bias_null(
    tube_data: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    centroids: dict,
    n_permutations: int,
    rng_seed: int = 0,
) -> Dict[Tuple[str, str, str], List[float]]:
    """
    Build per-(A,B,T) null distribution by permuting cell-type labels within tubes.

    For each permutation: shuffle cell-type assignments within each tube,
    recompute all bias(A, B, T) values, accumulate.
    Cell weights (if any) are tied to cells, not to cell-type labels — they are
    preserved as-is while only ct_labels are permuted.

    Returns: {(cyt_a, cyt_b, cell_type) -> list of n_permutations null bias values}
    """
    rng = np.random.default_rng(rng_seed)
    null: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    for _ in range(n_permutations):
        permuted_data = [
            (cyt, H_np, rng.permutation(ct_labels), cell_weights)
            for cyt, H_np, ct_labels, cell_weights in tube_data
        ]
        perm_bias = _compute_all_biases(permuted_data, centroids)
        for key, val in perm_bias.items():
            null[key].append(val)

    return dict(null)


def _z_to_bh_qvalues(
    z_scores: Dict[Tuple[str, str, str], float],
) -> Dict[Tuple[str, str, str], float]:
    """
    Convert z-scores to BH-FDR q-values.

    Two-tailed p-value from z-score via normal CDF approximation.
    Applies Benjamini-Hochberg correction across all (A, B, T) triples.

    Returns: {(cyt_a, cyt_b, cell_type) -> q_value}
    """
    from math import erfc, sqrt

    keys = list(z_scores.keys())
    if not keys:
        return {}

    # Two-tailed p-values using complementary error function.
    p_values = [
        float(erfc(abs(z_scores[k]) / sqrt(2)))
        for k in keys
    ]

    q_vals = _bh_correction(p_values)

    return {k: q for k, q in zip(keys, q_vals)}


def _bh_correction(p_values: List[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: list of p-values.
    Returns:
        list of q-values (adjusted p-values), same order as input.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort by p-value, track original indices.
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    ranks = np.arange(1, m + 1)
    bh_vals = sorted_p * m / ranks

    # Enforce monotonicity from the right.
    for i in range(m - 2, -1, -1):
        bh_vals[i] = min(bh_vals[i], bh_vals[i + 1])

    bh_vals = np.minimum(bh_vals, 1.0)

    # Return in original order.
    q_values = np.empty(m)
    q_values[order] = bh_vals
    return list(q_values)
