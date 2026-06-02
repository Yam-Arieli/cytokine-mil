"""Path A — coupling discovery via latent-space cytokine geometry.

This answers the *existence* question cross_asym does not: **which pairs of stimuli
are coupled at all** (direction-agnostic). After the encoder maps each cell to an
embedding, we subtract the per-cell-type control (PBS) centroid (PBS-RC), then for
each ordered pair (A, B) and cell type T test — across donors — whether A's cells of
type T are displaced toward B's centroid. A donor-level one-sided Wilcoxon signed-rank
test (Bonferroni-corrected across cell types) gives a coupling call and a candidate
relay cell type.

The statistics here are a faithful vendor of the validated §20.1 latent-geometry
pipeline (`pbs_rc.py` + `latent_geometry.py`). The only generalization is that the
control label is configurable (not hard-coded to "PBS").

**Power caveat (read this):** the test is across donors. With few donors the one-sided
Wilcoxon cannot reach small p (n=3 donors -> best one-sided p = 1/8 = 0.125 > 0.05), so
``coupled`` may be empty even when structure exists. With few donors, rank pairs by
``axis_strength`` (the Wilcoxon W) rather than trusting the ``coupled`` flag. The
validated 121-axis result used ~10 donors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from cascadir.exceptions import InsufficientDataError
from cascadir.train import resolve_device
from cascadir.types import AxisResult, MultiLabel, PseudoTubeSet

_ADVISORY_MIN_DONORS = 8  # below this the donor-level Wilcoxon is underpowered


# ===========================================================================
# PBS-RC primitives (vendored from pbs_rc.py; control label parameterized)
# ===========================================================================


def precompute_transform_means(
    cache: list,
    label_encoder,
    train_donors: Optional[Iterable[str]] = None,
    control_label: str = "PBS",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-cell-type embedding means: (global means, control means).

    Args:
        cache: list of dicts with keys "H" (N, D) torch tensor, "label" int,
            "cell_types" list[str], "donor" str.
        label_encoder: object with ``_idx_to_label`` (int -> condition name).
        train_donors: if given, only these donors contribute. None = all.
        control_label: the resting/control condition name.

    Returns:
        ``(global_ct_means, control_ct_means)`` keyed by cell type.
    """
    if not cache:
        return {}, {}
    embed_dim = cache[0]["H"].shape[1]
    ct_sum = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float64))
    ct_count: Dict[str, float] = defaultdict(float)
    pbs_sum = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float64))
    pbs_count: Dict[str, float] = defaultdict(float)
    train_set = set(train_donors) if train_donors is not None else None

    for entry in cache:
        if train_set is not None and entry.get("donor") not in train_set:
            continue
        H_np = entry["H"].numpy().astype(np.float64)
        ct_labels = entry["cell_types"]
        cytokine = label_encoder._idx_to_label[entry["label"]]
        for i, ct in enumerate(ct_labels):
            ct_sum[ct] += H_np[i]
            ct_count[ct] += 1.0
            if cytokine == control_label:
                pbs_sum[ct] += H_np[i]
                pbs_count[ct] += 1.0

    global_ct_means = {ct: ct_sum[ct] / ct_count[ct] for ct in ct_sum}
    pbs_ct_means = {
        ct: pbs_sum[ct] / pbs_count[ct] for ct in pbs_sum if pbs_count[ct] > 0
    }
    return global_ct_means, pbs_ct_means


def make_pbs_relative_fn(pbs_ct_means: Dict[str, np.ndarray]):
    """Return the transform ``h_i -> h_i - mu_{control, cell_type(i)}`` (cells of a
    cell type with no control representation are left unchanged)."""

    def fn(H_np: np.ndarray, ct_labels: np.ndarray) -> np.ndarray:
        result = H_np.copy()
        for i, ct in enumerate(ct_labels):
            if ct in pbs_ct_means:
                result[i] -= pbs_ct_means[ct]
        return result

    fn.__name__ = "pbs_relative"
    return fn


def compute_pbs_centroids_per_cell_type(
    cache: list,
    label_encoder,
    train_donors: Optional[Iterable[str]] = None,
    control_label: str = "PBS",
) -> Dict[str, np.ndarray]:
    """Per-cell-type control centroid dict (= ``precompute_transform_means(...)[1]``)."""
    _, pbs = precompute_transform_means(
        cache, label_encoder, train_donors=train_donors, control_label=control_label
    )
    return pbs


# ===========================================================================
# Directional bias per donor in PBS-RC space (vendored, §20.1)
# ===========================================================================


def compute_directional_bias_per_donor(
    cache: list,
    label_encoder,
    pbs_ct_means: dict,
    train_donors=None,
    direction_mode: str = "global",
) -> dict:
    """Per-donor mean embeddings ``mu_{A,T}^{(d)}`` in PBS-RC space (faithful §20.1).

    See :func:`test_directional_significance` for how these are turned into
    coupling calls. ``direction_mode`` is 'global' (direction = mu_B - mu_A) or
    'cell_type' (direction = mu_{B,T}).
    """
    if direction_mode not in ("global", "cell_type"):
        raise ValueError(
            f"direction_mode must be 'global' or 'cell_type', got {direction_mode!r}"
        )

    pbs_fn = make_pbs_relative_fn(pbs_ct_means)
    train_set = set(train_donors) if train_donors is not None else None

    sums: Dict[Tuple[str, str, str], np.ndarray] = {}
    counts: Dict[Tuple[str, str, str], float] = defaultdict(float)
    embed_dim = None
    donors_seen: set = set()

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

    b_per_donor: Dict[Tuple[str, str], Dict[str, np.ndarray]] = defaultdict(dict)
    for (cyt, ct, donor), s in sums.items():
        c = counts[(cyt, ct, donor)]
        if c <= 0:
            continue
        b_per_donor[(cyt, ct)][donor] = s / c

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
        cyt: cyt_sums[cyt] / cyt_counts[cyt] for cyt in cyt_sums if cyt_counts[cyt] > 0
    }
    centroids_AT = {
        key: at_sums[key] / at_counts[key] for key in at_sums if at_counts[key] > 0
    }

    return {
        "b_per_donor": dict(b_per_donor),
        "centroids": centroids,
        "centroids_AT": centroids_AT,
        "donors": sorted(d for d in donors_seen if d is not None),
        "direction_mode": direction_mode,
        "metric_description": (
            f"mu_{{A,T}}^{{(d)}} in PBS-RC space (direction_mode={direction_mode})"
        ),
    }


def _one_sided_wilcoxon_greater(x: np.ndarray, wilcoxon_fn) -> tuple:
    """One-sided Wilcoxon signed-rank H1: median(x) > 0. Returns (p_value, W)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return 1.0, 0.0
    if np.allclose(x, 0):
        return 1.0, 0.0
    try:
        res = wilcoxon_fn(x, alternative="greater", zero_method="wilcox")
        return float(res.pvalue), float(res.statistic)
    except ValueError:
        return 1.0, 0.0


def _bh_correction(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction (returns q-values in input order)."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    ranks = np.arange(1, m + 1)
    bh_vals = sorted_p * m / ranks
    for i in range(m - 2, -1, -1):
        bh_vals[i] = min(bh_vals[i], bh_vals[i + 1])
    bh_vals = np.minimum(bh_vals, 1.0)
    q_values = np.empty(m)
    q_values[order] = bh_vals
    return list(q_values)


def test_directional_significance(
    bias_per_donor: dict,
    label_encoder,
    alpha: float = 0.05,
) -> dict:
    """Donor-level Wilcoxon coupling tests on PBS-RC bias scores (faithful §20.1).

    For each ordered pair (A, B) and cell type T, two independent one-sided Wilcoxon
    signed-rank tests (forward A->B and reverse B->A) on the per-donor projections,
    Bonferroni-corrected across cell types. The cascade call uses the Bonferroni
    threshold only (BH across pairs is too conservative at small donor counts; the
    per-pair Wilcoxon W is the ranking score). Returns the full stat dict.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scipy>=1.10 is required for coupling discovery (Path A)."
        ) from exc

    b_per_donor = bias_per_donor["b_per_donor"]
    centroids = bias_per_donor["centroids"]
    centroids_AT = bias_per_donor["centroids_AT"]
    donors = bias_per_donor["donors"]
    direction_mode = bias_per_donor.get("direction_mode", "global")

    cytokine_names = list(label_encoder.cytokines)
    cell_types = sorted({ct for (_cyt, ct) in b_per_donor.keys()})
    n_cell_types = max(len(cell_types), 1)

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
    else:
        unit_dirs_t: Dict[Tuple[str, str], np.ndarray] = {}
        for (cyt, ct), mu in centroids_AT.items():
            u = _unit(mu)
            if u is not None:
                unit_dirs_t[(cyt, ct)] = u

    b_fwd: Dict[Tuple[str, str, str], np.ndarray] = {}
    b_rev: Dict[Tuple[str, str, str], np.ndarray] = {}
    p_fwd: Dict[Tuple[str, str, str], float] = {}
    p_rev: Dict[Tuple[str, str, str], float] = {}
    W_fwd: Dict[Tuple[str, str, str], float] = {}
    W_rev: Dict[Tuple[str, str, str], float] = {}

    for a in cytokine_names:
        for b in cytokine_names:
            if a == b:
                continue
            for ct in cell_types:
                if direction_mode == "global":
                    u_ab = unit_dirs.get((a, b))
                    u_ba = unit_dirs.get((b, a))
                else:
                    u_ab = unit_dirs_t.get((b, ct))
                    u_ba = unit_dirs_t.get((a, ct))
                if u_ab is None or u_ba is None:
                    continue
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
                    p, W = _one_sided_wilcoxon_greater(arr, wilcoxon)
                    p_fwd[(a, b, ct)] = p
                    W_fwd[(a, b, ct)] = W
                if len(rev_per_donor) >= 2:
                    arr = np.array(rev_per_donor)
                    b_rev[(b, a, ct)] = arr
                    p, W = _one_sided_wilcoxon_greater(arr, wilcoxon)
                    p_rev[(b, a, ct)] = p
                    W_rev[(b, a, ct)] = W

    p_fwd_bonf = {k: min(1.0, p * n_cell_types) for k, p in p_fwd.items()}
    p_rev_bonf = {k: min(1.0, p * n_cell_types) for k, p in p_rev.items()}

    pair_min_fwd: Dict[Tuple[str, str], float] = defaultdict(lambda: 1.0)
    pair_min_rev: Dict[Tuple[str, str], float] = defaultdict(lambda: 1.0)
    pair_argmin_fwd: Dict[Tuple[str, str], Optional[str]] = defaultdict(lambda: None)
    pair_max_W_fwd: Dict[Tuple[str, str], float] = defaultdict(float)
    pair_max_W_rev: Dict[Tuple[str, str], float] = defaultdict(float)
    for (a, b, ct), p in p_fwd_bonf.items():
        if p < pair_min_fwd[(a, b)]:
            pair_min_fwd[(a, b)] = p
            pair_argmin_fwd[(a, b)] = ct
        w = W_fwd.get((a, b, ct), 0.0)
        if w > pair_max_W_fwd[(a, b)]:
            pair_max_W_fwd[(a, b)] = w
    for (a, b, ct), p in p_rev_bonf.items():
        if p < pair_min_rev[(a, b)]:
            pair_min_rev[(a, b)] = p
        w = W_rev.get((a, b, ct), 0.0)
        if w > pair_max_W_rev[(a, b)]:
            pair_max_W_rev[(a, b)] = w

    pair_keys = sorted(pair_min_fwd.keys())
    q_pair_fwd = (
        {k: q for k, q in zip(pair_keys, _bh_correction([pair_min_fwd[k] for k in pair_keys]))}
        if pair_keys
        else {}
    )
    pair_keys_r = sorted(pair_min_rev.keys())
    q_pair_rev = (
        {k: q for k, q in zip(pair_keys_r, _bh_correction([pair_min_rev[k] for k in pair_keys_r]))}
        if pair_keys_r
        else {}
    )

    cascade_call: Dict[Tuple[str, str], str] = {}
    relay_T: Dict[Tuple[str, str], Optional[str]] = {}
    for (a, b) in pair_min_fwd:
        fwd_sig = pair_min_fwd[(a, b)] <= alpha
        rev_sig = pair_min_rev.get((b, a), 1.0) <= alpha
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
        "W_fwd": W_fwd,
        "W_rev": W_rev,
        "p_pair_fwd": dict(pair_min_fwd),
        "p_pair_rev": dict(pair_min_rev),
        "W_pair_fwd": dict(pair_max_W_fwd),
        "W_pair_rev": dict(pair_max_W_rev),
        "q_pair_fwd": q_pair_fwd,
        "q_pair_rev": q_pair_rev,
        "b_fwd": b_fwd,
        "b_rev": b_rev,
        "relay_T": relay_T,
        "cascade_call": cascade_call,
        "alpha": alpha,
        "n_cell_types": n_cell_types,
        "metric_description": (
            "Two one-sided Wilcoxon signed-rank tests on per-donor PBS-RC projections; "
            f"Bonferroni across {n_cell_types} cell types per pair; call at "
            f"Bonferroni p <= alpha={alpha}."
        ),
    }


# ===========================================================================
# cascadir glue: embedding cache + axis discovery
# ===========================================================================


def build_embedding_cache(
    encoder,
    tube_set: PseudoTubeSet,
    *,
    multilabel: MultiLabel,
    device: str | torch.device | None = None,
) -> list:
    """Run the encoder over every tube and collect per-tube embeddings.

    Args:
        encoder: an :class:`InstanceEncoder`, or any object with an ``.encoder``
            attribute (e.g. a trained multiclass :class:`AbMil`). The encoder output
            ``H`` is what the geometry uses.
        tube_set: the pseudo-tube set (all conditions, including the control).
        multilabel: a :class:`MultiLabel` over ``tube_set.conditions``.
        device: where to run the forward pass.

    Returns:
        list of dicts ``{"H": (N, D) CPU tensor, "label": int, "cell_types": list[str],
        "donor": str}`` — the cache format the geometry functions consume.
    """
    enc = getattr(encoder, "encoder", encoder)
    dev = resolve_device(device)
    enc = enc.to(dev).eval()
    cache: list = []
    with torch.no_grad():
        for t in tube_set.tubes:
            X = torch.from_numpy(np.ascontiguousarray(t.X, dtype=np.float32)).to(dev)
            H = enc(X).detach().cpu()
            cache.append(
                {
                    "H": H,
                    "label": multilabel.encode(t.condition),
                    "cell_types": list(t.cell_types),
                    "donor": t.donor,
                }
            )
    return cache


def build_axis_table(significance: dict, control_label: str) -> pd.DataFrame:
    """Collapse ``test_directional_significance`` into a tidy unordered-axis table.

    One row per unordered pair {A, B} (A < B), control excluded. Columns:
    ``axis_a``, ``axis_b``, ``coupled``, ``axis_strength`` (max Wilcoxon W),
    ``coupling_call`` (a_to_b/b_to_a/shared/none), ``dominant_direction``,
    ``p_fwd``, ``p_rev``, ``relay_T``. Sorted by descending strength.
    """
    cascade_call = significance.get("cascade_call", {})
    relay_T = significance.get("relay_T", {})
    p_pair_fwd = significance.get("p_pair_fwd", {})
    W_pair_fwd = significance.get("W_pair_fwd", {})

    conditions = sorted({c for pair in cascade_call for c in pair} | {
        c for pair in W_pair_fwd for c in pair
    })
    conditions = [c for c in conditions if c != control_label]
    call_map = {"A->B": "a_to_b", "B->A": "b_to_a", "shared": "shared", "none": "none"}

    rows: list[dict] = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            a, b = conditions[i], conditions[j]  # a < b by construction
            call = cascade_call.get((a, b), "none")
            w_fwd = float(W_pair_fwd.get((a, b), 0.0))
            w_rev = float(W_pair_fwd.get((b, a), 0.0))
            p_fwd = float(p_pair_fwd.get((a, b), 1.0))
            p_rev = float(p_pair_fwd.get((b, a), 1.0))
            if w_fwd > w_rev:
                dominant = "a_to_b"
                relay = relay_T.get((a, b))
            elif w_rev > w_fwd:
                dominant = "b_to_a"
                relay = relay_T.get((b, a))
            else:
                dominant = "tied"
                relay = relay_T.get((a, b))
            rows.append(
                {
                    "axis_a": a,
                    "axis_b": b,
                    "coupled": call != "none",
                    "axis_strength": max(w_fwd, w_rev),
                    "coupling_call": call_map.get(call, call),
                    "dominant_direction": dominant,
                    "p_fwd": p_fwd,
                    "p_rev": p_rev,
                    "relay_T": relay,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("axis_strength", ascending=False).reset_index(drop=True)
    return df


def discover_axes(
    tube_set: PseudoTubeSet,
    encoder,
    *,
    train_donors: Optional[List[str]] = None,
    direction_mode: str = "global",
    alpha: float = 0.05,
    device: str | torch.device | None = None,
) -> AxisResult:
    """Discover coupling axes (Path A) from a trained encoder + pseudo-tubes.

    Args:
        tube_set: the pseudo-tube set (all conditions incl control).
        encoder: a trained :class:`InstanceEncoder` (or an ``AbMil`` carrying one).
        train_donors: restrict the geometry to these donors (None = all donors).
        direction_mode: 'global' or 'cell_type' (see §20.1).
        alpha: Bonferroni threshold for the ``coupled`` call.
        device: where to run the encoder forward pass.

    Returns:
        An :class:`AxisResult` (axis table + raw significance + power flag).

    Raises:
        InsufficientDataError: if there are no control cells to build PBS-RC, or no
            donor pair is testable.
    """
    multilabel = MultiLabel(tube_set.conditions)
    cache = build_embedding_cache(encoder, tube_set, multilabel=multilabel, device=device)
    pbs_ct_means = compute_pbs_centroids_per_cell_type(
        cache, multilabel, train_donors=train_donors, control_label=tube_set.control_label
    )
    if not pbs_ct_means:
        raise InsufficientDataError(
            "discover_axes: no control cells found to build the PBS-RC baseline "
            f"(control_label={tube_set.control_label!r})."
        )
    bias = compute_directional_bias_per_donor(
        cache, multilabel, pbs_ct_means,
        train_donors=train_donors, direction_mode=direction_mode,
    )
    n_donors = len(bias["donors"])
    if n_donors < 2:
        raise InsufficientDataError(
            f"discover_axes: only {n_donors} donor(s) after filtering; the donor-level "
            "Wilcoxon needs >= 2 (and >= ~8 for real significance)."
        )
    significance = test_directional_significance(bias, multilabel, alpha=alpha)
    axes = build_axis_table(significance, tube_set.control_label)
    return AxisResult(
        axes=axes,
        significance=significance,
        n_donors=n_donors,
        alpha=alpha,
        direction_mode=direction_mode,
        underpowered=n_donors < _ADVISORY_MIN_DONORS,
    )
