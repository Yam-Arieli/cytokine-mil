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

from cytokine_mil.analysis.pbs_rc import make_pbs_relative_fn
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


def compute_directional_bias_per_donor(
    cache: list,
    label_encoder,
    pbs_ct_means: dict,
    train_donors=None,
    direction_mode: str = "global",
) -> dict:
    """
    Per-donor directional bias in PBS-RC space. The refined readout that replaces
    the antisymmetric `bias - bias` subtraction.

    For each cytokine pair (A, B), each cell type T, each training donor d:

        b_fwd^{(d)}(A→B, T) = µ_{A,T}^{(d)} · û_{A→B}     (does T move toward B in A-tubes?)
        b_rev^{(d)}(B→A, T) = µ_{B,T}^{(d)} · û_{B→A}     (does T move toward A in B-tubes?)

    All embeddings are first transformed into PBS-RC space:
        h̃_i = h_i - µ_{PBS, τ(i)}.
    PBS centroids are computed on `train_donors` only and passed in via
    `pbs_ct_means` (use `pbs_rc.compute_pbs_centroids_per_cell_type`).

    The projection is (µ_{A,T}^{(d)} − µ_A) · û_{A→B}: the PBS-RC centroid of A
    is subtracted before projection so that the score reflects T's specific
    deviation from cytokine A's average cross-cell-type response, isolating the
    T-specific cascade component from A's direct generic effect on all cells.

    Args:
        cache: list of dicts with keys "H" (N, D) torch tensor, "label" int,
               "cell_types" list[str], "donor" str.
        label_encoder: object with `_idx_to_label` (int -> cytokine name) and
               `cytokines` (ordered list of names).
        pbs_ct_means: {cell_type -> µ_{PBS, T}} from
               pbs_rc.compute_pbs_centroids_per_cell_type.
        train_donors: iterable of donor names to include. If None, every donor
               in the cache is used.
        direction_mode: 'global' uses û_{A→B} = (µ_B - µ_A)/||...||;
               'cell_type' uses µ̂_{B,T} = µ_{B,T}/||µ_{B,T}|| as the direction
               for cell type T (T-specific direction).

    Returns:
        dict with keys:
            'b_per_donor':  {(A, T) -> {donor -> µ_{A,T}^{(d)}  in PBS-RC space}}
            'centroids':    {A -> µ_A in PBS-RC space (training-donor mean)}
            'centroids_AT': {(A, T) -> µ_{A,T} in PBS-RC space (training-donor mean)}
            'donors':       sorted list of donor names actually present
            'metric_description': str
    """
    if direction_mode not in ("global", "cell_type"):
        raise ValueError(
            f"direction_mode must be 'global' or 'cell_type', got {direction_mode!r}"
        )

    pbs_fn = make_pbs_relative_fn(pbs_ct_means)
    train_set = set(train_donors) if train_donors is not None else None

    # (A, T, donor) -> [sum vector, count]
    sums: Dict[Tuple[str, str, str], np.ndarray] = {}
    counts: Dict[Tuple[str, str, str], float] = defaultdict(float)
    embed_dim = None
    donors_seen = set()

    for entry in cache:
        donor = entry.get("donor")
        if train_set is not None and donor not in train_set:
            continue
        H_np = entry["H"].numpy().astype(np.float64)
        ct_labels = np.array(entry["cell_types"])
        cyt = label_encoder._idx_to_label[entry["label"]]
        if embed_dim is None:
            embed_dim = H_np.shape[1]
        H_pbs = pbs_fn(H_np, ct_labels)
        donors_seen.add(donor)
        for ct in np.unique(ct_labels):
            mask = ct_labels == ct
            key = (cyt, ct, donor)
            if key not in sums:
                sums[key] = np.zeros(embed_dim, dtype=np.float64)
            sums[key] += H_pbs[mask].sum(axis=0)
            counts[key] += float(mask.sum())

    # Per-donor µ_{A,T}^{(d)} in PBS-RC space.
    b_per_donor: Dict[Tuple[str, str], Dict[str, np.ndarray]] = defaultdict(dict)
    for (cyt, ct, donor), s in sums.items():
        c = counts[(cyt, ct, donor)]
        if c <= 0:
            continue
        b_per_donor[(cyt, ct)][donor] = s / c

    # Pooled-train centroids µ_A and µ_{A,T} (used for direction vectors).
    cyt_sums: Dict[str, np.ndarray] = {}
    cyt_counts: Dict[str, float] = defaultdict(float)
    at_sums: Dict[Tuple[str, str], np.ndarray] = {}
    at_counts: Dict[Tuple[str, str], float] = defaultdict(float)
    for (cyt, ct, _donor), s in sums.items():
        c = counts[(cyt, ct, _donor)]
        if cyt not in cyt_sums:
            cyt_sums[cyt] = np.zeros(embed_dim, dtype=np.float64)
        cyt_sums[cyt] += s
        cyt_counts[cyt] += c
        if (cyt, ct) not in at_sums:
            at_sums[(cyt, ct)] = np.zeros(embed_dim, dtype=np.float64)
        at_sums[(cyt, ct)] += s
        at_counts[(cyt, ct)] += c

    centroids = {
        cyt: cyt_sums[cyt] / cyt_counts[cyt]
        for cyt in cyt_sums if cyt_counts[cyt] > 0
    }
    centroids_AT = {
        key: at_sums[key] / at_counts[key]
        for key in at_sums if at_counts[key] > 0
    }

    description = (
        "µ_{{A,T}}^{{(d)}} · û_{{A→B}} per donor in PBS-RC space (direction_mode={mode})"
    ).format(mode=direction_mode)

    return {
        "b_per_donor": dict(b_per_donor),
        "centroids": centroids,
        "centroids_AT": centroids_AT,
        "donors": sorted(d for d in donors_seen if d is not None),
        "direction_mode": direction_mode,
        "metric_description": description,
    }


def test_directional_significance(
    bias_per_donor: dict,
    label_encoder,
    alpha: float = 0.05,
) -> dict:
    """
    Donor-level Wilcoxon tests on per-donor PBS-RC bias scores.

    For each ordered pair (A, B) and each cell type T:
        - Compute b_fwd^{(d)}(A→B, T) = µ_{A,T}^{(d)} · û_{A→B}
        - Compute b_rev^{(d)}(B→A, T) = µ_{B,T}^{(d)} · û_{B→A}
          where û_{A→B} = (µ_B - µ_A) / ||µ_B - µ_A|| (or µ̂_{B,T} when
          direction_mode='cell_type').
        - One-sided Wilcoxon signed-rank tests (alternative='greater'), independently.
        - Bonferroni-correct p-values by the number of cell types (relay search).
        - BH-FDR across the K(K-1) ordered pairs on the per-pair min p value.

    No subtraction of forward vs reverse anywhere. The two tests are stored
    side by side. Cascade direction is decided by their pattern of significance.

    Args:
        bias_per_donor: output of compute_directional_bias_per_donor().
        label_encoder: object with `.cytokines` ordered list of names.
        alpha: BH-FDR alpha for cascade calls.

    Returns:
        dict with keys:
            'p_fwd', 'p_rev':            {(A, B, T) -> p-value}
            'p_fwd_bonf', 'p_rev_bonf':  {(A, B, T) -> Bonferroni-adjusted p}
            'q_pair_fwd', 'q_pair_rev':  {(A, B) -> BH q-value of min Bonf p across T}
            'b_fwd', 'b_rev':            {(A, B, T) -> np.ndarray of per-donor scores}
            'relay_T':                   {(A, B) -> T* = argmin_T p_fwd_bonf}
            'cascade_call':              {(A, B) -> 'A->B' | 'B->A' | 'shared' | 'none'}
            'metric_description':        str
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise ImportError(
            "scipy>=1.10 is required for test_directional_significance. "
            "Install with: pip install 'scipy>=1.10'"
        ) from exc

    b_per_donor = bias_per_donor["b_per_donor"]
    centroids = bias_per_donor["centroids"]
    centroids_AT = bias_per_donor["centroids_AT"]
    donors = bias_per_donor["donors"]
    direction_mode = bias_per_donor.get("direction_mode", "global")

    cytokine_names = list(label_encoder.cytokines)
    cell_types = sorted({ct for (_cyt, ct) in b_per_donor.keys()})
    n_cell_types = max(len(cell_types), 1)

    # Pre-compute direction unit vectors.
    def _unit(v: np.ndarray) -> Optional[np.ndarray]:
        n = float(np.linalg.norm(v))
        return None if n < 1e-10 else v / n

    if direction_mode == "global":
        unit_dirs: Dict[Tuple[str, str], np.ndarray] = {}
        for a in cytokine_names:
            mu_a = centroids.get(a)
            if mu_a is None:
                continue
            for b in cytokine_names:
                if a == b:
                    continue
                mu_b = centroids.get(b)
                if mu_b is None:
                    continue
                u = _unit(mu_b - mu_a)
                if u is not None:
                    unit_dirs[(a, b)] = u
    else:  # cell_type
        unit_dirs_t: Dict[Tuple[str, str], np.ndarray] = {}
        for (cyt, ct), mu in centroids_AT.items():
            u = _unit(mu)
            if u is not None:
                unit_dirs_t[(cyt, ct)] = u

    b_fwd: Dict[Tuple[str, str, str], np.ndarray] = {}
    b_rev: Dict[Tuple[str, str, str], np.ndarray] = {}
    p_fwd: Dict[Tuple[str, str, str], float] = {}
    p_rev: Dict[Tuple[str, str, str], float] = {}

    for a in cytokine_names:
        for b in cytokine_names:
            if a == b:
                continue
            for ct in cell_types:
                # Forward: project µ_{A,T}^{(d)} onto direction toward B.
                if direction_mode == "global":
                    u_ab = unit_dirs.get((a, b))
                    u_ba = unit_dirs.get((b, a))
                else:
                    u_ab = unit_dirs_t.get((b, ct))   # direction = µ̂_{B,T}
                    u_ba = unit_dirs_t.get((a, ct))   # direction = µ̂_{A,T}
                if u_ab is None or u_ba is None:
                    continue

                # Subtract the pooled cytokine centroid before projection so that
                # the per-donor score reflects T's *specific* deviation from
                # cytokine A's average response across all cell types, not the
                # raw PBS-RC displacement (which is dominated by A's direct effect).
                mu_a = centroids.get(a)
                mu_b = centroids.get(b)
                fwd_per_donor = []
                for d in donors:
                    mu_at = b_per_donor.get((a, ct), {}).get(d)
                    if mu_at is None:
                        continue
                    vec = mu_at - mu_a if mu_a is not None else mu_at
                    fwd_per_donor.append(float(np.dot(vec, u_ab)))
                rev_per_donor = []
                for d in donors:
                    mu_bt = b_per_donor.get((b, ct), {}).get(d)
                    if mu_bt is None:
                        continue
                    vec = mu_bt - mu_b if mu_b is not None else mu_bt
                    rev_per_donor.append(float(np.dot(vec, u_ba)))

                if len(fwd_per_donor) >= 2:
                    arr = np.array(fwd_per_donor)
                    b_fwd[(a, b, ct)] = arr
                    p_fwd[(a, b, ct)] = _one_sided_wilcoxon_greater(arr, wilcoxon)
                if len(rev_per_donor) >= 2:
                    arr = np.array(rev_per_donor)
                    b_rev[(b, a, ct)] = arr
                    p_rev[(b, a, ct)] = _one_sided_wilcoxon_greater(arr, wilcoxon)

    # Bonferroni across cell types (relay search) — per ordered pair.
    p_fwd_bonf = {k: min(1.0, p * n_cell_types) for k, p in p_fwd.items()}
    p_rev_bonf = {k: min(1.0, p * n_cell_types) for k, p in p_rev.items()}

    # Per-pair min p across cell types.
    pair_min_fwd: Dict[Tuple[str, str], float] = defaultdict(lambda: 1.0)
    pair_min_rev: Dict[Tuple[str, str], float] = defaultdict(lambda: 1.0)
    pair_argmin_fwd: Dict[Tuple[str, str], Optional[str]] = defaultdict(lambda: None)
    for (a, b, ct), p in p_fwd_bonf.items():
        if p < pair_min_fwd[(a, b)]:
            pair_min_fwd[(a, b)] = p
            pair_argmin_fwd[(a, b)] = ct
    for (a, b, ct), p in p_rev_bonf.items():
        if p < pair_min_rev[(a, b)]:
            pair_min_rev[(a, b)] = p

    # BH-FDR across ordered pairs.
    pair_keys = sorted(pair_min_fwd.keys())
    if pair_keys:
        q_fwd_list = _bh_correction([pair_min_fwd[k] for k in pair_keys])
        q_pair_fwd = {k: q for k, q in zip(pair_keys, q_fwd_list)}
    else:
        q_pair_fwd = {}
    pair_keys_r = sorted(pair_min_rev.keys())
    if pair_keys_r:
        q_rev_list = _bh_correction([pair_min_rev[k] for k in pair_keys_r])
        q_pair_rev = {k: q for k, q in zip(pair_keys_r, q_rev_list)}
    else:
        q_pair_rev = {}

    # Cascade calls.
    cascade_call: Dict[Tuple[str, str], str] = {}
    relay_T: Dict[Tuple[str, str], Optional[str]] = {}
    for (a, b) in q_pair_fwd:
        fwd_sig = q_pair_fwd.get((a, b), 1.0) <= alpha
        rev_sig = q_pair_rev.get((b, a), 1.0) <= alpha
        if fwd_sig and not rev_sig:
            cascade_call[(a, b)] = "A->B"
        elif rev_sig and not fwd_sig:
            cascade_call[(a, b)] = "B->A"
        elif fwd_sig and rev_sig:
            cascade_call[(a, b)] = "shared"
        else:
            cascade_call[(a, b)] = "none"
        relay_T[(a, b)] = pair_argmin_fwd.get((a, b))

    return {
        "p_fwd": p_fwd,
        "p_rev": p_rev,
        "p_fwd_bonf": p_fwd_bonf,
        "p_rev_bonf": p_rev_bonf,
        "q_pair_fwd": q_pair_fwd,
        "q_pair_rev": q_pair_rev,
        "b_fwd": b_fwd,
        "b_rev": b_rev,
        "relay_T": relay_T,
        "cascade_call": cascade_call,
        "alpha": alpha,
        "n_cell_types": n_cell_types,
        "metric_description": (
            "Two independent one-sided Wilcoxon signed-rank tests on per-donor "
            "µ_{A,T}^{(d)} · û_{A→B} (forward) and µ_{B,T}^{(d)} · û_{B→A} (reverse) "
            f"in PBS-RC space; Bonferroni across {n_cell_types} cell types, "
            f"BH-FDR across ordered pairs at alpha={alpha}."
        ),
    }


def _one_sided_wilcoxon_greater(x: np.ndarray, wilcoxon_fn) -> float:
    """One-sided Wilcoxon signed-rank H1: median(x) > 0. Robust to all-zero / tied."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return 1.0
    if np.allclose(x, 0):
        return 1.0
    try:
        res = wilcoxon_fn(x, alternative="greater", zero_method="wilcox")
        return float(res.pvalue)
    except ValueError:
        return 1.0


def compute_asymmetry_matrix(
    bias: dict,
    label_encoder,
) -> dict:
    """
    DEPRECATED. The antisymmetric subtraction `bias(A,B,T) - bias(B,A,T)` injects
    a `µ_{B,T} · u_{A→B}` contamination term: a cell type that is a strong
    direct B-responder (no cascade) inflates the asymmetry score in the A→B
    direction. The score is also antisymmetric by construction, so it cannot
    distinguish a genuine cascade from an algebraic sign flip.

    Use `compute_directional_bias_per_donor` + `test_directional_significance`
    instead. Kept here only for backwards compatibility with the legacy pipeline
    (`scripts/run_experiment_geo.py --legacy-asymmetry`).

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


def compute_cytokine_centroids_from_cache(
    cache: list,
    label_encoder,
    h_transform_fn=None,
) -> dict:
    """
    Compute mean cell embedding per cytokine from a prebuilt in-memory cache.

    Avoids repeated h5ad disk I/O — uses the H tensors already cached by
    build_cache(). Intended for post-hoc geometry variants (PBS-relative,
    h_residual) where a cell-level transformation is applied before centroid
    computation.

    Args:
        cache: list of dicts with keys "H" (N, D) CPU tensor, "label" int,
               "cell_types" list[str].  Output of build_cache().
        label_encoder: CytokineLabel with ._idx_to_label dict.
        h_transform_fn: Optional callable (H_np: ndarray, ct_labels: ndarray)
               -> ndarray.  Applied to each tube's H before accumulation.
               Use make_pbs_relative_fn() or make_hresidual_fn() from the
               run_experiment_geo script.

    Returns:
        dict with keys:
            'centroids': {cytokine_name -> np.ndarray of shape (D,)}
            'metric_description': str
    """
    if not cache:
        return {"centroids": {}, "metric_description": "empty cache"}

    embed_dim = cache[0]["H"].shape[1]
    embedding_sums: Dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(embed_dim, dtype=np.float64)
    )
    cell_counts: Dict[str, float] = defaultdict(float)

    for entry in cache:
        H_np = entry["H"].numpy().astype(np.float64)          # (N, D)
        ct_labels = np.array(entry["cell_types"])              # (N,)
        cytokine = label_encoder._idx_to_label[entry["label"]]

        if h_transform_fn is not None:
            H_np = h_transform_fn(H_np, ct_labels)

        embedding_sums[cytokine] += H_np.sum(axis=0)
        cell_counts[cytokine] += float(H_np.shape[0])

    centroids = {
        cyt: embedding_sums[cyt] / cell_counts[cyt]
        for cyt in embedding_sums
        if cell_counts[cyt] > 0
    }

    transform_label = (
        h_transform_fn.__name__ if hasattr(h_transform_fn, "__name__") else "custom"
    ) if h_transform_fn is not None else "none"

    return {
        "centroids": centroids,
        "metric_description": (
            f"μ_C = mean h_i over all C-tubes in cache "
            f"(h_transform: {transform_label})"
        ),
    }


def build_tube_data_from_cache(
    cache: list,
    label_encoder,
    centroids: dict,
    h_transform_fn=None,
) -> list:
    """
    Build tube_data list for compute_directional_bias_from_arrays() from cache.

    Each entry is a 4-tuple:
        (cytokine_name, H_np (N,D), ct_labels (N,), cell_weights or None)

    Only tubes whose cytokine is present in centroids are included.

    Args:
        cache: output of build_cache().
        label_encoder: CytokineLabel with ._idx_to_label.
        centroids: dict from compute_cytokine_centroids_from_cache()['centroids'].
        h_transform_fn: same optional transform as in centroid computation.

    Returns:
        List of (cytokine_name, H_np, ct_labels, None) tuples.
    """
    tube_data = []
    for entry in cache:
        H_np = entry["H"].numpy().astype(np.float64)
        ct_labels = np.array(entry["cell_types"])
        cytokine = label_encoder._idx_to_label[entry["label"]]

        if cytokine not in centroids:
            continue

        if h_transform_fn is not None:
            H_np = h_transform_fn(H_np, ct_labels)

        tube_data.append((cytokine, H_np, ct_labels, None))  # None = uniform weights

    return tube_data


def compute_directional_bias_from_arrays(
    tube_data: list,
    centroids: dict,
    n_permutations: int = 1000,
    transform_label: str = "none",
) -> dict:
    """
    Compute directional bias, z-scores, and BH-FDR q-values from pre-built arrays.

    Accepts tube_data already built and transformed (no model or dataset needed).
    Intended for use with compute_cytokine_centroids_from_cache() and
    build_tube_data_from_cache().

    For every (A, B, T) triple:
        bias(A, B, T) = (μ_{A,T} - μ_A) · (μ_B - μ_A) / ||μ_B - μ_A||_2

    Null: permute cell-type labels within each tube n_permutations times.
    z-score: (obs - null_mean) / null_std.
    BH-FDR correction across all (A, B, T) triples.

    Args:
        tube_data: list of (cytokine_name, H_np (N,D), ct_labels (N,), weights or None).
                   Output of build_tube_data_from_cache().
        centroids: {cytokine_name -> np.ndarray (D,)}.
        n_permutations: number of cell-type permutations for null distribution.
        transform_label: string describing the h_transform applied (for metric_description).

    Returns:
        dict with keys:
            'bias': {(cyt_a, cyt_b, cell_type) -> float}
            'z_scores': {(cyt_a, cyt_b, cell_type) -> float}
            'q_values': {(cyt_a, cyt_b, cell_type) -> float}
            'metric_description': str
    """
    bias_obs = _compute_all_biases(tube_data, centroids)
    bias_null = _compute_bias_null(tube_data, centroids, n_permutations=n_permutations)

    z_scores: Dict[Tuple[str, str, str], float] = {}
    for key, obs in bias_obs.items():
        null_vals = bias_null.get(key, [])
        if len(null_vals) < 2:
            z_scores[key] = 0.0
        else:
            null_arr = np.array(null_vals)
            null_m = float(null_arr.mean())
            null_s = float(null_arr.std())
            z_scores[key] = (obs - null_m) / null_s if null_s > 0 else 0.0

    q_values = _z_to_bh_qvalues(z_scores)

    return {
        "bias": dict(bias_obs),
        "z_scores": dict(z_scores),
        "q_values": q_values,
        "metric_description": (
            f"scalar projection of (μ_{{A,T}} - μ_A) [h_i, transform={transform_label}] "
            "onto unit vector (μ_B - μ_A)/||μ_B - μ_A||_2"
        ),
    }


def build_latent_cascade_graph_from_calls(
    significance: dict,
    label_encoder,
) -> "nx.DiGraph":
    """
    Build a directed cytokine cascade graph from `test_directional_significance`
    output. This is the refined replacement for `build_latent_cascade_graph`.

    Edge A→B is added when `cascade_call[(A, B)] == 'A->B'`. Edge attributes
    record the per-pair forward q-value and the predicted relay cell type.

    Args:
        significance: output of test_directional_significance().
        label_encoder: object with `.cytokines`.

    Returns:
        nx.DiGraph with nodes = cytokine names, edges A→B with attributes:
            'q_pair_fwd': BH q-value for the (A, B) forward pair test
            'relay_T':    predicted relay cell type (argmin Bonferroni p_fwd)
            'call':       'A->B'
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "networkx is required for build_latent_cascade_graph_from_calls. "
            "Install with: pip install networkx"
        ) from exc

    G = nx.DiGraph()
    for name in label_encoder.cytokines:
        G.add_node(name)

    cascade_call = significance.get("cascade_call", {})
    q_pair_fwd = significance.get("q_pair_fwd", {})
    relay_T = significance.get("relay_T", {})

    for (a, b), call in cascade_call.items():
        if call == "A->B":
            G.add_edge(
                a, b,
                q_pair_fwd=float(q_pair_fwd.get((a, b), 1.0)),
                relay_T=relay_T.get((a, b)),
                call=call,
            )
    return G


def build_latent_cascade_graph(
    asymmetry_matrix: np.ndarray,
    z_scores: dict,
    label_encoder,
    fdr_alpha: float = 0.05,
) -> "nx.DiGraph":
    """
    DEPRECATED. Uses the legacy antisymmetric `compute_asymmetry_matrix` output.
    Use `build_latent_cascade_graph_from_calls(test_directional_significance(...),
    label_encoder)` instead.

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
