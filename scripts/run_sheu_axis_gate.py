"""
Sheu 2024 phase 1 axis-discovery gate (CLAUDE.md §21).

For each of three trained seeds, this script:
  1. Loads model_stage2.pt + manifest_train + label_encoder.
  2. Builds a per-tube cache of (H embeddings, cell_types, donor, label).
  3. Computes PBS-relative-centroids per cell type from training donors only.
  4. Computes per-donor directional bias in PBS-RC space (§20.1 refined readout).
  5. Runs one-sided Wilcoxon + Bonferroni-over-T + BH-FDR over the K(K-1)
     ordered pairs to get cascade calls.

Across the 3 seeds it then:
  - Tests the 8 pre-registered axes (2 MUST, 3 SHOULD, 3 MUST-NOT) and writes
    a per-seed table.
  - Computes pair-wise Spearman ρ of axis ranking (signed-log-q) across seeds.
  - Applies the §21 verdict: 2/2 MUST + ≥2/3 SHOULD + 0/3 MUST-NOT.
  - Writes reports/sheu2024/AXIS_GATE_VERDICT.md.

Usage:
    python scripts/run_sheu_axis_gate.py \
        --seed_dirs results/sheu2024_full/seed_42 \
                    results/sheu2024_full/seed_123 \
                    results/sheu2024_full/seed_7 \
        --hvg_path /cs/.../Sheu2024_pseudotubes/hvg_list.json \
        --output_dir reports/sheu2024
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model
from cytokine_mil.analysis.pbs_rc import compute_pbs_centroids_per_cell_type
from cytokine_mil.analysis.latent_geometry import (
    compute_directional_bias_per_donor,
    test_directional_significance,
)


# Pre-registered axes — use GEO names that match the manifest (polyIC == "PIC").
PREREG_MUST = [("LPS", "TNF"), ("PIC", "IFNb")]
PREREG_SHOULD = [("LPS", "IFNb"), ("P3CSK", "CpG"), ("LPSlo", "P3CSK")]
PREREG_MUST_NOT = [("P3CSK", "IFNb"), ("CpG", "IFNb"), ("TNF", "IFNb")]

# Cosmetic alias for the report so readers see textbook names alongside GEO codes.
ALIAS = {"PIC": "polyIC", "P3CSK": "Pam3CSK4"}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs="+", required=True,
                   help="3 trained seed directories.")
    p.add_argument("--hvg_path", required=True)
    p.add_argument("--output_dir", default="reports/sheu2024")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="BH-FDR alpha for Wilcoxon (§21 gate uses 0.05).")
    p.add_argument("--min_seed_rho", type=float, default=0.7,
                   help="Spearman ρ across seeds required for axis ranking.")
    p.add_argument("--direction_mode", default="global",
                   choices=["global", "cell_type"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--relax_gate", action="store_true",
                   help="Add a relaxed-readout section to the verdict that "
                        "skips Bonferroni-over-T and BH-FDR (uses raw min-over-T "
                        "Wilcoxon p at alpha) and reports per-donor effect sizes "
                        "+ sign-agreement for the pre-registered positive axes. "
                        "Strict §21 gate is still reported alongside.")
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


class _DynamicLabelEncoder:
    """CytokineLabel-shaped wrapper with n_classes() inferred from the map.

    The package CytokineLabel.n_classes() is hardcoded to 91 (PBS_INDEX+1).
    For the slim Sheu encoder (n=8) that hardcode is wrong and causes the
    gate to allocate a 91-class head when it should build an 8-class one.
    """
    def __init__(self, label_to_idx):
        self._label_to_idx = dict(label_to_idx)
        self._idx_to_label = {v: k for k, v in self._label_to_idx.items()}
    def encode(self, c): return self._label_to_idx[c]
    def decode(self, i): return self._idx_to_label[i]
    def n_classes(self): return len(self._label_to_idx)
    @property
    def cytokines(self):
        return [self._idx_to_label[i] for i in sorted(self._idx_to_label)]


def _load_label_encoder(run_dir: Path):
    with open(run_dir / "label_encoder.json") as f:
        data = json.load(f)
    cytokines_list = data["cytokines"]
    label_to_idx = {cyt: i for i, cyt in enumerate(cytokines_list)}
    return _DynamicLabelEncoder(label_to_idx)


def _infer_n_cell_types(state_dict: dict) -> int:
    for key in ("encoder.cell_type_head.weight", "cell_type_head.weight"):
        if key in state_dict:
            return state_dict[key].shape[0]
    raise ValueError("Cannot infer n_cell_types from state dict.")


def _infer_dims_from_state(state_dict: dict):
    """Read embed_dim and attention_hidden_dim from the checkpoint shapes.

    `encoder.down2.fc2.weight` has shape (embed_dim, embed_dim).
    `attention.V.weight` has shape (attention_hidden_dim, embed_dim).
    """
    embed_dim = state_dict["encoder.down2.fc2.weight"].shape[0]
    attn_hidden = state_dict["attention.V.weight"].shape[0]
    return embed_dim, attn_hidden


def _load_model(run_dir: Path, label_enc, gene_names, device: str):
    state_dict = torch.load(run_dir / "model_stage2.pt", map_location="cpu")
    n_cell_types = _infer_n_cell_types(state_dict)
    embed_dim, attn_hidden = _infer_dims_from_state(state_dict)
    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=n_cell_types,
        embed_dim=embed_dim,
    )
    model = build_mil_model(
        encoder,
        embed_dim=embed_dim,
        attention_hidden_dim=attn_hidden,
        n_classes=label_enc.n_classes(),
        encoder_frozen=True,
    )
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()
    return model


def _build_cache(model, dataset: PseudoTubeDataset, device: str) -> list:
    """
    Forward each train tube; collect (H, label, cell_types, donor) per entry.
    H is the post-encoder embedding (shape (N, D)) BEFORE attention pooling.
    """
    cache = []
    dev = torch.device(device)
    for idx, entry in enumerate(dataset.entries):
        X_t, label_t, donor, cyt_name = dataset[idx]
        adata = sc.read_h5ad(entry["path"])
        cell_types = list(adata.obs["cell_type"].values) if "cell_type" in adata.obs.columns else None
        if cell_types is None:
            _log(f"  WARNING: tube {idx} ({entry['path']}) missing cell_type column.")
            continue
        with torch.no_grad():
            X = X_t.unsqueeze(0).to(dev)
            _, _, H = model(X)        # (1, N, D)
            H = H.squeeze(0).cpu()    # (N, D)
        cache.append({
            "H": H,
            "label": int(label_t),
            "cell_types": cell_types,
            "donor": donor,
        })
    return cache


def _run_seed(seed_dir: Path, gene_names, alpha: float,
              direction_mode: str, device: str) -> dict:
    _log(f"\n=== {seed_dir.name} ===")
    label_enc = _load_label_encoder(seed_dir)
    _log(f"  Label encoder: {label_enc.n_classes()} classes")
    model = _load_model(seed_dir, label_enc, gene_names, device)
    train_ds = PseudoTubeDataset(
        str(seed_dir / "manifest_train.json"),
        label_enc,
        gene_names=gene_names,
        preload=False,
    )
    _log(f"  Train tubes: {len(train_ds)}")
    cache = _build_cache(model, train_ds, device)
    train_donors = sorted({e["donor"] for e in cache})
    _log(f"  Train donors ({len(train_donors)}): {train_donors}")

    pbs_ct_means = compute_pbs_centroids_per_cell_type(
        cache, label_enc, train_donors=train_donors
    )
    _log(f"  PBS centroids: {len(pbs_ct_means)} cell types")

    bias = compute_directional_bias_per_donor(
        cache, label_enc, pbs_ct_means,
        train_donors=train_donors, direction_mode=direction_mode,
    )
    sig = test_directional_significance(bias, label_enc, alpha=alpha)
    _log(f"  Pairs with cascade_call != 'none': "
         f"{sum(1 for v in sig['cascade_call'].values() if v != 'none')}"
         f" / {len(sig['cascade_call'])}")
    return {
        "seed_dir": str(seed_dir),
        "label_encoder_cytokines": list(label_enc.cytokines),
        "train_donors": train_donors,
        "bias": bias,
        "sig": sig,
    }


# ---------------------------------------------------------------------------
# Axis-level aggregation
# ---------------------------------------------------------------------------


def _axis_min_q(sig, a, b):
    """Min(q_pair_fwd(A,B), q_pair_rev(A,B)) = q_pair_fwd(B,A). Direction-agnostic."""
    qf = sig["q_pair_fwd"].get((a, b), np.nan)
    qr = sig["q_pair_fwd"].get((b, a), np.nan)
    return float(np.nanmin([qf, qr]))


def _axis_call(sig, a, b):
    """'A->B', 'B->A', 'shared', 'none' — direction-agnostic axis is positive
    iff cascade_call(A,B) != 'none'."""
    return sig["cascade_call"].get((a, b), "none")


def _seed_axis_table(sig, axes):
    """List of dicts per axis with q and call for one seed."""
    out = []
    for a, b in axes:
        out.append({
            "axis": f"{a}—{b}",
            "min_q": _axis_min_q(sig, a, b),
            "call": _axis_call(sig, a, b),
            "relay_T": sig["relay_T"].get((a, b), None),
        })
    return out


def _ranking_vector(sig, cytokines):
    """Direction-agnostic axis ranking score: -log10(min_q) per unordered pair.
    Returns (pairs_in_order, vector). Pairs are canonicalised lexicographically."""
    pairs = []
    scores = []
    for i, a in enumerate(cytokines):
        for j, b in enumerate(cytokines):
            if a >= b:           # canonical unordered
                continue
            q = _axis_min_q(sig, a, b)
            if np.isnan(q):
                continue
            pairs.append((a, b))
            scores.append(-np.log10(max(q, 1e-300)))
    return pairs, np.array(scores)


def _cross_seed_rho(seed_results):
    """Pairwise Spearman ρ of axis ranking vectors across seeds."""
    cytokines = seed_results[0]["label_encoder_cytokines"]
    # Restrict to cytokines that are actually represented (non-PBS pre-reg active set)
    active = sorted({c for c in cytokines if c != "PBS" and
                     any((c, "PBS") in r["sig"]["q_pair_fwd"] or
                         (c, c) not in r["sig"]["q_pair_fwd"]
                         for r in seed_results)})
    # Fall back to a fixed Sheu-active set if introspection is brittle.
    if not active:
        active = ["LPS", "LPSlo", "P3CSK", "PIC", "TNF", "CpG", "IFNb"]

    # Use the intersection of pairs that have a q-value in all seeds.
    common_pairs = None
    seed_scores = []
    for r in seed_results:
        pairs, scores = _ranking_vector(r["sig"], active)
        pair_to_score = dict(zip(pairs, scores))
        if common_pairs is None:
            common_pairs = set(pairs)
        else:
            common_pairs &= set(pairs)
        seed_scores.append(pair_to_score)

    common_pairs = sorted(common_pairs)
    vectors = []
    for s in seed_scores:
        vectors.append(np.array([s[p] for p in common_pairs]))

    pair_rhos = []
    n = len(seed_results)
    for i in range(n):
        for j in range(i + 1, n):
            if len(vectors[i]) < 3:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(vectors[i], vectors[j])
            pair_rhos.append({
                "i": i, "j": j, "rho": float(rho), "p": float(p),
            })
    return {"common_pair_count": len(common_pairs), "pair_rhos": pair_rhos}


def _format_axis_row(name, axes_per_seed, kind):
    """Markdown row for one axis across all seeds."""
    qs = [a["min_q"] for a in axes_per_seed]
    calls = [a["call"] for a in axes_per_seed]
    relays = [a["relay_T"] for a in axes_per_seed]
    pass_flag = all((not np.isnan(q)) and q <= 0.05 for q in qs)
    if kind == "must_not":
        pass_flag = not any((c != "none") for c in calls)
    sig_str = " / ".join(f"{q:.3g}" for q in qs)
    call_str = " / ".join(calls)
    relay_str = " / ".join(str(r) for r in relays)
    pass_mark = "PASS" if pass_flag else "FAIL"
    return f"| {name} | {sig_str} | {call_str} | {relay_str} | {pass_mark} |", pass_flag


# ---------------------------------------------------------------------------
# Relaxed readout (raw min-over-T p, per-donor effect sizes, sign-agreement)
# ---------------------------------------------------------------------------


def _axis_raw_min_p_per_seed(sig, a, b):
    """Direction-agnostic raw min-over-T Wilcoxon p (no Bonferroni, no BH).

    p_axis = min over T of min(p_fwd[A,B,T], p_fwd[B,A,T])  (one-sided greater).
    """
    candidates = []
    for (ca, cb, ct), p in sig["p_fwd"].items():
        if {ca, cb} == {a, b}:
            candidates.append(float(p))
    if not candidates:
        return float("nan")
    return float(np.nanmin(candidates))


def _axis_effect_size(sig, a, b):
    """Aggregate per-donor effect size over the best-cell-type direction.

    Returns dict with:
      best_T : the T at which raw p is smallest (across both orientations)
      direction : 'A->B' or 'B->A'
      b_per_donor : np.array of per-donor scores at that (orientation, T)
      mean_b : mean across donors
      sign_agreement : fraction of donors with sign matching the mean's sign
    """
    best = None
    for (ca, cb, ct), p in sig["p_fwd"].items():
        if {ca, cb} != {a, b}:
            continue
        if best is None or p < best[0]:
            best = (p, ca, cb, ct)
    if best is None:
        return None
    p, ca, cb, ct = best
    b_vals = np.asarray(sig["b_fwd"].get((ca, cb, ct), []), dtype=float)
    if b_vals.size == 0:
        return None
    mean_b = float(np.mean(b_vals))
    sgn = np.sign(mean_b) if mean_b != 0 else 0.0
    if sgn == 0:
        agree = 0.5
    else:
        agree = float(np.mean(np.sign(b_vals) == sgn))
    return {
        "best_T": ct,
        "direction": f"{ca}->{cb}",
        "n_donors": int(b_vals.size),
        "b_per_donor": b_vals.tolist(),
        "mean_b": mean_b,
        "raw_p": float(p),
        "sign_agreement": agree,
    }


def _relaxed_axis_row(name, seed_results, axis_a, axis_b, alpha, kind):
    """Return (markdown row, pass_flag) for an axis under the relaxed gate.

    Relaxed positive criterion: raw min-over-T p ≤ alpha in every seed AND
    sign-agreement ≥ 2/3 of donors in every seed.
    Relaxed must-not criterion: no seed satisfies the positive criterion.
    """
    p_per_seed = [_axis_raw_min_p_per_seed(r["sig"], axis_a, axis_b)
                  for r in seed_results]
    eff_per_seed = [_axis_effect_size(r["sig"], axis_a, axis_b)
                    for r in seed_results]
    sign_per_seed = [(eff["sign_agreement"] if eff else float("nan"))
                     for eff in eff_per_seed]
    mean_b_per_seed = [(eff["mean_b"] if eff else float("nan"))
                       for eff in eff_per_seed]

    positive_each = [
        (not np.isnan(p)) and p <= alpha and
        (not np.isnan(sa)) and sa >= 2.0 / 3.0
        for p, sa in zip(p_per_seed, sign_per_seed)
    ]
    pass_flag = (all(positive_each) if kind != "must_not" else not any(positive_each))

    p_str = " / ".join(f"{p:.3g}" for p in p_per_seed)
    sa_str = " / ".join(f"{s:.2f}" for s in sign_per_seed)
    b_str = " / ".join(f"{b:+.4f}" for b in mean_b_per_seed)
    pass_mark = "PASS" if pass_flag else "FAIL"
    return (f"| {name} | {p_str} | {b_str} | {sa_str} | {pass_mark} |", pass_flag)


def _apply_verdict(must_pass, should_pass, must_not_pass):
    """§21 composite verdict."""
    n_must = sum(must_pass)
    n_should = sum(should_pass)
    n_must_not_violated = sum(1 for p in must_not_pass if not p)  # not-pass = called as cascade
    if n_must == 2 and n_should >= 2 and n_must_not_violated == 0:
        return "GREEN"
    if n_must >= 1 and n_must_not_violated <= 1:
        return "AMBER"
    return "RED"


def _label_pair(a, b):
    aa = f"{a} ({ALIAS[a]})" if a in ALIAS else a
    bb = f"{b} ({ALIAS[b]})" if b in ALIAS else b
    return f"{aa} — {bb}"


def _write_relaxed_section(seed_results, alpha):
    """Markdown lines for the relaxed-gate section."""
    lines = []
    lines.append("## Relaxed gate (raw min-over-T Wilcoxon p, no Bonferroni, no BH)")
    lines.append("")
    lines.append(f"_Pass criterion: raw p ≤ {alpha} in **every** seed AND per-donor "
                 f"sign-agreement ≥ 2/3 in every seed._")
    lines.append("")
    lines.append("| axis | raw min p per seed | mean per-donor b per seed | sign-agreement per seed | gate |")
    lines.append("|---|---|---|---|---|")
    must_pass = []
    for a, b in PREREG_MUST:
        row, ok = _relaxed_axis_row(_label_pair(a, b), seed_results, a, b, alpha, "must")
        lines.append(row)
        must_pass.append(ok)
    should_pass = []
    for a, b in PREREG_SHOULD:
        row, ok = _relaxed_axis_row(_label_pair(a, b), seed_results, a, b, alpha, "should")
        lines.append(row)
        should_pass.append(ok)
    must_not_pass = []
    for a, b in PREREG_MUST_NOT:
        row, ok = _relaxed_axis_row(_label_pair(a, b), seed_results, a, b, alpha, "must_not")
        lines.append(row)
        must_not_pass.append(ok)
    lines.append("")
    lines.append(f"- Relaxed MUST passed: **{sum(must_pass)} / 2**")
    lines.append(f"- Relaxed SHOULD passed: **{sum(should_pass)} / 3**")
    lines.append(f"- Relaxed MUST-NOT violations: **{sum(1 for p in must_not_pass if not p)} / 3**")
    lines.append("")
    return lines, must_pass, should_pass, must_not_pass


def _write_verdict(seed_results, rho_info, out_dir: Path, alpha: float,
                   min_seed_rho: float, direction_mode: str,
                   relax_gate: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "AXIS_GATE_VERDICT.md"

    must_tables = [_seed_axis_table(r["sig"], PREREG_MUST) for r in seed_results]
    should_tables = [_seed_axis_table(r["sig"], PREREG_SHOULD) for r in seed_results]
    must_not_tables = [_seed_axis_table(r["sig"], PREREG_MUST_NOT) for r in seed_results]

    lines = []
    lines.append("# Sheu 2024 phase 1 — axis-discovery gate verdict\n")
    lines.append(f"_Generated by `scripts/run_sheu_axis_gate.py`, CLAUDE.md §21._\n")
    lines.append(f"- BH-FDR alpha: **{alpha}**")
    lines.append(f"- Direction mode: **{direction_mode}**")
    lines.append(f"- Cross-seed Spearman ρ threshold: **{min_seed_rho}**")
    lines.append(f"- Seeds: {', '.join(Path(r['seed_dir']).name for r in seed_results)}")
    lines.append("")

    # MUST table
    lines.append("## MUST recover (2 of 2 required)")
    lines.append("")
    lines.append("| axis | min(q_fwd, q_rev) per seed | cascade_call per seed | relay_T per seed | gate |")
    lines.append("|---|---|---|---|---|")
    must_pass = []
    for k, (a, b) in enumerate(PREREG_MUST):
        per_seed = [t[k] for t in must_tables]
        row, ok = _format_axis_row(_label_pair(a, b), per_seed, "must")
        lines.append(row)
        must_pass.append(ok)
    lines.append("")

    # SHOULD table
    lines.append("## SHOULD recover (≥2 of 3 required)")
    lines.append("")
    lines.append("| axis | min(q_fwd, q_rev) per seed | cascade_call per seed | relay_T per seed | gate |")
    lines.append("|---|---|---|---|---|")
    should_pass = []
    for k, (a, b) in enumerate(PREREG_SHOULD):
        per_seed = [t[k] for t in should_tables]
        row, ok = _format_axis_row(_label_pair(a, b), per_seed, "should")
        lines.append(row)
        should_pass.append(ok)
    lines.append("")

    # MUST NOT table
    lines.append("## MUST NOT call (0 of 3 required as false positives)")
    lines.append("")
    lines.append("| axis | min(q_fwd, q_rev) per seed | cascade_call per seed | relay_T per seed | gate |")
    lines.append("|---|---|---|---|---|")
    must_not_pass = []
    for k, (a, b) in enumerate(PREREG_MUST_NOT):
        per_seed = [t[k] for t in must_not_tables]
        row, ok = _format_axis_row(_label_pair(a, b), per_seed, "must_not")
        lines.append(row)
        must_not_pass.append(ok)
    lines.append("")

    # Cross-seed stability
    lines.append("## Cross-seed stability (axis ranking Spearman ρ)")
    lines.append("")
    lines.append(f"Common axes across all seeds: **{rho_info['common_pair_count']}**\n")
    lines.append("| seed_i | seed_j | Spearman ρ | p |")
    lines.append("|---|---|---|---|")
    seed_names = [Path(r["seed_dir"]).name for r in seed_results]
    for pr in rho_info["pair_rhos"]:
        lines.append(f"| {seed_names[pr['i']]} | {seed_names[pr['j']]} | {pr['rho']:.3f} | {pr['p']:.3g} |")
    lines.append("")
    rho_pass = all((pr["rho"] >= min_seed_rho) for pr in rho_info["pair_rhos"]
                   if not np.isnan(pr["rho"]))
    lines.append(f"Spearman ρ ≥ {min_seed_rho} on every pair: **{'PASS' if rho_pass else 'FAIL'}**")
    lines.append("")

    # Optional relaxed section (raw min-over-T p, effect sizes, sign-agreement)
    relaxed_summary = None
    if relax_gate:
        rl_lines, rl_must, rl_should, rl_must_not = _write_relaxed_section(
            seed_results, alpha,
        )
        lines.extend(rl_lines)
        relaxed_summary = {
            "must_pass": rl_must,
            "should_pass": rl_should,
            "must_not_pass": rl_must_not,
        }

    # Verdict
    verdict = _apply_verdict(must_pass, should_pass, must_not_pass)
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- MUST passed: **{sum(must_pass)} / 2**")
    lines.append(f"- SHOULD passed: **{sum(should_pass)} / 3**")
    lines.append(f"- MUST-NOT violations: **{sum(1 for p in must_not_pass if not p)} / 3**")
    lines.append(f"- Axis-ranking Spearman ρ ≥ {min_seed_rho}: **{'PASS' if rho_pass else 'FAIL'}**")
    lines.append("")
    lines.append(f"### Composite: **{verdict}**")
    lines.append("")
    if verdict == "GREEN":
        lines.append("→ Proceed to phase 2 (time-axis extension, CLAUDE.md §21).")
    elif verdict == "AMBER":
        lines.append("→ Re-run with `direction_mode: cell_type` and `n_per_cell_type=50`; "
                     "if still amber, write partial result and defer the direction question.")
    else:
        lines.append("→ Cascade signal not recoverable from 3hr BMDM with this architecture. "
                     "Try 1hr/5hr time-point subsets before reconsidering phase 2.")
    lines.append("")
    lines.append("---\n")
    lines.append("**Note:** This phase-1 gate is a method-validation check; the Oesinghaus "
                 "axis-discovery result is independent of this verdict (CLAUDE.md §0).")

    md.write_text("\n".join(lines))
    _log(f"\nVerdict written to: {md}")

    # Also save the per-seed structured outputs.
    pkl = out_dir / "axis_gate_results.pkl"
    payload = {
        "seed_results": seed_results,
        "rho_info": rho_info,
        "verdict": verdict,
        "must_pass": must_pass,
        "should_pass": should_pass,
        "must_not_pass": must_not_pass,
    }
    if relaxed_summary is not None:
        payload["relaxed_summary"] = relaxed_summary
    with open(pkl, "wb") as f:
        pickle.dump(payload, f)
    _log(f"Detailed results saved to: {pkl}")
    return verdict


def main():
    args = _parse_args()
    out_dir = Path(args.output_dir)
    seed_dirs = [Path(d) for d in args.seed_dirs]
    _log(f"Seed dirs: {[d.name for d in seed_dirs]}")
    _log(f"HVG path : {args.hvg_path}")
    _log(f"Output   : {out_dir}")

    with open(args.hvg_path) as f:
        gene_names = json.load(f)
    _log(f"Gene names: {len(gene_names)}")

    seed_results = [
        _run_seed(d, gene_names, args.alpha, args.direction_mode, args.device)
        for d in seed_dirs
    ]
    rho_info = _cross_seed_rho(seed_results)
    verdict = _write_verdict(
        seed_results, rho_info, out_dir,
        alpha=args.alpha,
        min_seed_rho=args.min_seed_rho,
        direction_mode=args.direction_mode,
        relax_gate=args.relax_gate,
    )
    _log(f"\nFinal verdict: {verdict}")


if __name__ == "__main__":
    main()
