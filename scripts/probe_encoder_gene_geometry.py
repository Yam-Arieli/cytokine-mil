"""Does the frozen Stage-1 encoder bottleneck the gene space?

Every Oesinghaus-90 re-fit emits signatures far less cytokine-specific than the published
anchor (mean between-cytokine Jaccard 0.18-0.39 vs 0.065; chance 0.006), and four sweeps
have falsified every settings-level explanation (CLAUDE.md §38.3/§38.5). This probe asks a
different kind of question: what does the TRAINED ENCODER do to the gene space?

If the encoder maps many genes onto a few shared directions, every binary head sits on the
same impoverished representation, and Integrated Gradients must return overlapping genes
for every cytokine no matter what the head learned. IG attributions factor as

    dy/dx_g = (dy/dh) . (dh/dx_g)

so a low-rank {dh/dx_g} confines ALL heads to one low-dimensional gene subspace. That is a
complete mechanism for the collapse -- if it holds.

Three probes of the gene -> embedding map, same statistics on each:

  * one-hot   Z_oh[g]  = E(e_g) - E(0)     -- where a pure gene-g impulse lands.
                                              E(0) is subtracted because input_proj is
                                              LN(W1[:,g] + b1): without it every gene
                                              carries the same bias offset and the cosine
                                              matrix looks collapsed for a trivial reason.
                                              OFF-DISTRIBUTION by construction, and
                                              LayerNorm makes the map non-additive, so
                                              E(x) != sum_g x_g Z_oh[g] + E(0).
  * jacobian  Z_jac[g] = dE(x)/dx_g at real cells -- the local linearisation, and the one
                                              with the mechanistic chain to IG above.
  * w1        Z_w1[g]  = W1[:, g]          -- the literal 4000->512 squeeze, no modelling
                                              assumptions at all.

Note the architecture: the compression is at the INPUT (Linear(4000 -> 512), 7.8x), not at
the end -- down2 is 512->512 and compresses nothing. And all fits share this architecture,
so a bottleneck alone cannot explain the DIFFERENCES between them; the claim under test is
that training collapsed the effective rank, by different amounts in different fits.

No method math here: this computes SVDs, cosines and gradient norms on encoder outputs.
It does not compute engagement scores, PBS-normalised values, cross_asym, coupling or
direction, and it reads signatures only as an existing table, to count gene occurrences.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _encoder_geometry_fits as R  # noqa: E402
from cascadir.models import InstanceEncoder  # noqa: E402

PROBES = ("onehot", "jacobian", "w1")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# encoder loading -- one loader for both code paths
# ---------------------------------------------------------------------------

def infer_dims(state: dict) -> dict:
    """Read the architecture off the weights.

    `cascadir.models.InstanceEncoder` and `cytokine_mil.models.instance_encoder.
    InstanceEncoder` are structurally identical, so one loader serves both training paths.
    Dimensions come from the state_dict rather than a config, which is what makes the older
    bare `encoder_shared_stage1.pt` checkpoints (no `encoder_meta.json`) loadable at all.
    """
    try:
        w0 = state["input_proj.0.weight"]
        return {
            "input_dim": int(w0.shape[1]),
            "hidden_dims": (int(w0.shape[0]), int(state["down1.fc1.weight"].shape[0])),
            "embed_dim": int(state["down2.fc1.weight"].shape[0]),
            "n_cell_types": (int(state["cell_type_head.weight"].shape[0])
                             if "cell_type_head.weight" in state else None),
        }
    except KeyError as exc:  # pragma: no cover - a genuinely foreign checkpoint
        raise SystemExit(f"not an InstanceEncoder state_dict (missing {exc})")


def load_encoder(path: Path, device: str = "cpu"):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    dims = infer_dims(state)
    enc = InstanceEncoder(
        input_dim=dims["input_dim"], embed_dim=dims["embed_dim"],
        n_cell_types=dims["n_cell_types"], hidden_dims=dims["hidden_dims"],
    )
    enc.load_state_dict(state)
    return enc.to(device).eval(), dims


def untrained_like(dims: dict, seed: int, device: str = "cpu"):
    """The no-collapse ceiling: a freshly initialised encoder at matched dims.

    Without it there is no scale on which to read "PR = 47 of 512".
    """
    torch.manual_seed(seed)
    enc = InstanceEncoder(
        input_dim=dims["input_dim"], embed_dim=dims["embed_dim"],
        n_cell_types=dims["n_cell_types"], hidden_dims=dims["hidden_dims"],
    )
    return enc.to(device).eval()


# ---------------------------------------------------------------------------
# the three gene -> embedding maps
# ---------------------------------------------------------------------------

@torch.no_grad()
def onehot_matrix(enc, device: str, batch: int = 512):
    """Z_oh[g] = E(e_g) - E(0), one forward pass per gene (the encoder is frozen)."""
    G = enc.input_dim
    zero = torch.zeros(1, G, device=device)
    h0 = enc(zero)[0]
    out = torch.empty(G, enc.embed_dim, dtype=torch.float32)
    for s in range(0, G, batch):
        e = min(s + batch, G)
        X = torch.zeros(e - s, G, device=device)
        X[torch.arange(e - s), torch.arange(s, e)] = 1.0
        out[s:e] = (enc(X) - h0).cpu()
    return out.numpy(), h0.cpu().numpy()


def jacobian_matrix(enc, x: torch.Tensor):
    """Z_jac[g] = dE(x)/dx_g at a single real cell. Returns (G, embed_dim)."""
    def f(v):
        return enc(v.unsqueeze(0)).squeeze(0)
    try:
        from torch.func import jacrev
        J = jacrev(f)(x)                                    # (embed_dim, G)
    except Exception:                                       # pragma: no cover
        J = torch.autograd.functional.jacobian(f, x, vectorize=True)
    return J.detach().T.cpu().numpy()                       # (G, embed_dim)


def w1_matrix(enc):
    """Z_w1[g] = W1[:, g] -- the literal input squeeze, before any nonlinearity."""
    return enc.input_proj[0].weight.detach().T.cpu().numpy()  # (G, h0)


# ---------------------------------------------------------------------------
# geometry statistics -- identical for every probe, so they stay comparable
# ---------------------------------------------------------------------------

def _spectrum_stats(Z: np.ndarray, prefix: str) -> dict:
    sv = np.linalg.svd(Z, compute_uv=False)
    p = sv ** 2
    tot = p.sum()
    if tot <= 0:
        return {f"{prefix}pr": np.nan, f"{prefix}eff_rank": np.nan}
    p = p / tot
    pr = 1.0 / np.sum(p ** 2)                     # participation ratio
    eff = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    return {
        f"{prefix}pr": float(pr),
        f"{prefix}eff_rank": eff,
        f"{prefix}var_top1": float(p[0]),
        f"{prefix}var_top5": float(p[:5].sum()),
        f"{prefix}var_top10": float(p[:10].sum()),
    }


def _gini(v: np.ndarray) -> float:
    v = np.sort(np.abs(v))
    n = v.size
    if n == 0 or v.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def geometry_stats(Z: np.ndarray, *, want_cos: bool = True, return_spectrum: bool = False):
    """Rank / concentration of the gene -> embedding map.

    Returns (scalars, per_gene) -- plus the singular-value spectrum when asked -- where
    per_gene carries the columns the gene-level link needs: each gene's response norm and
    its alignment with the top embedding direction.
    """
    d = Z.shape[1]
    norms = np.linalg.norm(Z, axis=1)
    stats = {"embed_dim": d, "n_genes": Z.shape[0]}
    stats.update(_spectrum_stats(Z, "raw_"))
    stats["raw_pr_frac"] = stats["raw_pr"] / d

    keep = norms > 0
    Zn = np.zeros_like(Z)
    Zn[keep] = Z[keep] / norms[keep, None]
    stats.update(_spectrum_stats(Zn, "unit_"))
    stats["unit_pr_frac"] = stats["unit_pr"] / d

    stats["norm_gini"] = _gini(norms)
    stats["norm_median"] = float(np.median(norms))
    stats["norm_p99_over_median"] = (
        float(np.percentile(norms, 99) / np.median(norms)) if np.median(norms) > 0 else np.nan
    )

    # top right-singular direction: how aligned is each gene with the dominant axis?
    _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    cos_u1 = np.abs(Zn @ Vt[0])
    stats["mean_cos_u1"] = float(cos_u1.mean())

    if want_cos:
        C = Zn @ Zn.T
        n = C.shape[0]
        off = (np.abs(C).sum() - np.abs(np.diag(C)).sum()) / (n * n - n)
        stats["mean_abs_cos"] = float(off)
        del C

    pg = {"norm": norms, "cos_u1": cos_u1}
    return (stats, pg, sv) if return_spectrum else (stats, pg)


# ---------------------------------------------------------------------------
# operating points for the Jacobian probe
# ---------------------------------------------------------------------------

def sample_cells(shard_dir: Path, donor: str, n_each: int, seed: int):
    """n_each control cells + n_each stimulated cells, from real tubes.

    Real cells rather than a constructed mean vector: a more faithful operating point, and
    it involves no arithmetic that resembles the method's own.
    """
    from cytokine_mil.analysis.full90_tube_io import load_tube_set, read_meta

    meta = read_meta(shard_dir)
    ts = load_tube_set(shard_dir, donors=[donor], tube_indices=[0])
    ctrl = ts.control_label
    rng = np.random.default_rng(seed)

    def draw(is_ctrl: bool):
        pool = [t.X for t in ts.tubes if (t.condition == ctrl) == is_ctrl]
        X = np.concatenate(pool, axis=0)
        idx = rng.choice(X.shape[0], size=min(n_each, X.shape[0]), replace=False)
        return X[np.sort(idx)].astype(np.float32)

    cells = np.concatenate([draw(True), draw(False)], axis=0)
    labels = ["control"] * n_each + ["stimulated"] * (cells.shape[0] - n_each)
    return cells, labels, [str(g) for g in meta["gene_names"]]


# ---------------------------------------------------------------------------
# per-fit driver
# ---------------------------------------------------------------------------

def signature_frequency(fit: dict, genes: list, top_n: int):
    """freq[g] = how many of THIS fit's own signatures contain gene g at top_n."""
    if not fit.get("signatures"):
        return None, 0
    p = REPO_ROOT / fit["signatures"]
    if not p.exists():
        log(f"    [freq] no signature table at {fit['signatures']} - gene link skipped")
        return None, 0
    df = pd.read_parquet(p)
    if "epoch" in df.columns:
        df = df[df.epoch == df.epoch.max()]
    if "rank_ig" not in df.columns:
        df = df.sort_values(["cytokine", "ig"], ascending=[True, False])
        df["rank_ig"] = df.groupby("cytokine").cumcount()
    df = df[df.rank_ig < top_n]
    if fit.get("conditions"):
        df = df[df.cytokine.isin(fit["conditions"])]
    counts = df.groupby("gene").cytokine.nunique()
    freq = pd.Series(0, index=genes, dtype=int)
    common = counts.index.intersection(freq.index)
    freq.loc[common] = counts.loc[common].astype(int)
    return freq.to_numpy(), int(df.cytokine.nunique())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def probe_fit(fit: dict, cells: np.ndarray, cell_labels: list, genes: list, args) -> tuple:
    path = REPO_ROOT / fit["encoder"]
    if not path.exists():
        log(f"[skip] {fit['key']}: {fit['encoder']} not found")
        return [], None, None
    t0 = time.time()
    enc, dims = load_encoder(path, args.device)
    if dims["input_dim"] != len(genes):
        raise SystemExit(
            f"{fit['key']}: encoder takes {dims['input_dim']} genes but the shard gene list "
            f"has {len(genes)} - the gene index would not mean the same thing across fits."
        )
    log(f"[{fit['key']}] embed={dims['embed_dim']} hidden={dims['hidden_dims']} "
        f"({fit['code_path']})")

    rows, per_gene, spectra = [], {"gene": genes}, []

    def record(probe: str, stats: dict, pg: dict, extra=None):
        r = {"fit": fit["key"], "probe": probe, "code_path": fit["code_path"],
             "panel": fit["panel"], "diagnostic_only": fit["diagnostic_only"]}
        r.update(stats)
        if extra:
            r.update(extra)
        rows.append(r)
        per_gene[f"{probe}_norm"] = pg["norm"]
        per_gene[f"{probe}_cos_u1"] = pg["cos_u1"]

    # --- probe 1: one-hot (primary, as specified) --------------------------
    Z_oh, h_zero = onehot_matrix(enc, args.device)
    st, pg, sv = geometry_stats(Z_oh, return_spectrum=True)
    spectra.append(("onehot", sv))
    record("onehot", st, pg, extra={
        # how much of the raw one-hot response is just the shared bias? Reported so bias
        # domination is visible rather than hidden by the E(0) subtraction.
        "bias_norm_over_median_response":
            float(np.linalg.norm(h_zero) / max(np.median(np.linalg.norm(Z_oh, axis=1)), 1e-12)),
    })

    # --- probe 2: Jacobian at real cells (the IG-relevant control) ----------
    per_cell, acc_norm, acc_cos = [], np.zeros(len(genes)), np.zeros(len(genes))
    for i in range(cells.shape[0]):
        x = torch.from_numpy(cells[i]).to(args.device)
        Zj = jacobian_matrix(enc, x)
        s, p, sv = geometry_stats(Zj, want_cos=False, return_spectrum=True)
        if i == 0:
            spectra.append(("jacobian", sv))
        s["cell_kind"] = cell_labels[i]
        per_cell.append(s)
        acc_norm += p["norm"]
        acc_cos += p["cos_u1"]
    pc = pd.DataFrame(per_cell)
    num = pc.select_dtypes(include=[np.number])
    st = {k: float(num[k].mean()) for k in num.columns}
    st.update({f"{k}_sd": float(num[k].std()) for k in ("raw_pr", "unit_pr", "raw_var_top1")})
    for kind in ("control", "stimulated"):
        sub = pc[pc.cell_kind == kind]
        if len(sub):
            st[f"raw_pr_{kind}"] = float(sub.raw_pr.mean())
    st["n_jac_cells"] = int(cells.shape[0])
    record("jacobian", st,
           {"norm": acc_norm / cells.shape[0], "cos_u1": acc_cos / cells.shape[0]})

    # --- probe 3: first-layer weights (free linear reference) ---------------
    st, pg, sv = geometry_stats(w1_matrix(enc), return_spectrum=True)
    spectra.append(("w1", sv))
    record("w1", st, pg)

    # --- the gene-level link -----------------------------------------------
    freq, n_sig = signature_frequency(fit, genes, args.top_n)
    if freq is not None:
        per_gene["sig_freq"] = freq
        for r in rows:
            probe = r["probe"]
            r["n_signature_cytokines"] = n_sig
            r["rho_freq_norm"] = spearman(freq.astype(float), per_gene[f"{probe}_norm"])
            r["rho_freq_cos_u1"] = spearman(freq.astype(float), per_gene[f"{probe}_cos_u1"])
            insig = freq > 0
            if insig.sum() > 2 and (~insig).sum() > 2:
                n_in = per_gene[f"{probe}_norm"][insig]
                n_out = per_gene[f"{probe}_norm"][~insig]
                r["sig_gene_norm_ratio"] = float(np.median(n_in) / max(np.median(n_out), 1e-12))
                r["n_signature_genes"] = int(insig.sum())

    # --- untrained control at matched dims ---------------------------------
    if args.untrained:
        u = untrained_like(dims, args.seed, args.device)
        Zu, _ = onehot_matrix(u, args.device)
        su, _ = geometry_stats(Zu, want_cos=False)
        rows.append({"fit": f"{fit['key']}__untrained", "probe": "onehot",
                     "code_path": "control", "panel": fit["panel"],
                     "diagnostic_only": True, **su})

    spec = pd.concat([
        pd.DataFrame({"fit": fit["key"], "probe": name, "k": np.arange(len(sv)),
                      "singular_value": sv})
        for name, sv in spectra
    ], ignore_index=True)

    log(f"    done in {time.time() - t0:.0f}s")
    return rows, pd.DataFrame(per_gene), spec


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "results" / "encoder_geometry"))
    ap.add_argument("--shard_dir", default=str(R.SHARD_DIR))
    ap.add_argument("--fits", nargs="*", default=None, help="subset of registry keys")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--n_jac_cells", type=int, default=8,
                    help="per kind; 8 gives 8 control + 8 stimulated = 16 Jacobians")
    ap.add_argument("--jac_donor", default="Donor1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-untrained", dest="untrained", action="store_false")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "per_gene").mkdir(parents=True, exist_ok=True)

    fits = [f for f in R.FITS if args.fits is None or f["key"] in args.fits]
    if not fits:
        raise SystemExit(f"no registry fits matched {args.fits}")

    log(f"[cells] sampling Jacobian operating points from {args.jac_donor}, tube_idx 0")
    cells, labels, genes = sample_cells(
        Path(args.shard_dir), args.jac_donor, args.n_jac_cells, args.seed)
    log(f"[cells] {cells.shape[0]} cells x {len(genes)} genes "
        f"({labels.count('control')} control, {labels.count('stimulated')} stimulated)")

    all_rows, all_spec = [], []
    for fit in fits:
        rows, pg, spec = probe_fit(fit, cells, labels, genes, args)
        all_rows += rows
        if pg is not None:
            pg.to_parquet(out / "per_gene" / f"{fit['key']}.parquet", index=False)
        if spec is not None:
            all_spec.append(spec)

    if not all_rows:
        raise SystemExit("no fit could be probed")
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "gene_geometry.csv", index=False)
    if all_spec:
        pd.concat(all_spec, ignore_index=True).to_parquet(out / "spectra.parquet", index=False)
    log(f"\n[write] {out / 'gene_geometry.csv'}  ({len(df)} rows)")

    (out / "probe_meta.json").write_text(json.dumps({
        "n_fits": len(fits), "top_n": args.top_n, "n_jac_cells": cells.shape[0],
        "jac_donor": args.jac_donor, "seed": args.seed, "device": args.device,
        "shard_dir": str(args.shard_dir),
    }, indent=2))

    show = ["fit", "probe", "raw_pr", "raw_pr_frac", "var_top1", "mean_abs_cos",
            "norm_gini", "rho_freq_norm"]
    show = [c for c in show if c in df.columns]
    log("\n" + df[df.probe.isin(("onehot", "jacobian"))][show]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
