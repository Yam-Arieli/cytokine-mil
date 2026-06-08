"""
Toy: complex-valued directional gene->gene influence from a cell "soup".

GOAL
----
Decide cheaply (on simulated data, before touching real data or the package)
whether a *complex-valued, hollow-diagonal bilinear* gene->gene operator can
recover the DIRECTION of a planted cross-cell cascade, and whether complex
buys anything over real-valued baselines.

THE TASK (user's reframe)
-------------------------
From a "soup" of cells, predict ONE held-out cell from the OTHER cells, with
each gene predicted only from *other* genes (hollow diagonal -> no self-copy).
Direction is read off the asymmetry of the learned gene->gene operator M.

WHY DIRECTION IS IDENTIFIABLE HERE (the crux — read before trusting results)
----------------------------------------------------------------------------
A naive model `y = M a + noise` with `a` carrying a tube-level common factor is
NOT direction-identifiable: if drivers vary at the tube level and responses
reflect them, responses predict drivers just as well as the reverse (symmetric
correlation; the trap that defeated the cytokine cross_asym work). Linear-
Gaussian *snapshot* correlation cannot orient an edge.

The asymmetry that DOES work — realizing the user's "1 cell is affected by the
499, not the 499 by the 1":

  * DRIVERS are cell-autonomous: each cell draws its driver independently (no
    shared tube-level value for downstream genes to reflect back).
  * RESPONSES react to the leave-target-out POPULATION AGGREGATE of drivers:
        r_i = beta * mean_{j != i} d_j  + noise
  * Prediction input `a_i` = leave-i-out mean over the OTHER cells.

=> Held-out RESPONSE depends on the public aggregate -> predictable
   -> M[response, driver] ~ beta.
   Held-out DRIVER is the cell's OWN private draw, independent of `a`
   -> NOT predictable -> M[driver, *] ~ 0 (only an O(1/N) leak).
   Direction = "downstream genes are predictable from the soup, upstream genes
   are not." Recovered by PLAIN LINEAR regression; the leave-one-out is what
   makes it work. No time axis, no non-Gaussianity required.

PREPROCESSING (regression hygiene, defeats the copy trap by construction)
-------------------------------------------------------------------------
Targets are residualised against the per-cell-type mean (y - mu_type) and inputs
are centered per gene (a - a_mean), both using TRAIN statistics. This removes
constant offsets so models fit covariation (= direction), and removes the
"detect type, copy its mean" cheat from the target. What remains for a model to
exploit is the cross-gene structure only.

HONESTY CONTROL
---------------
A free hollow REAL matrix already recovers this direction, so complex numbers
are NOT justified by Stage-1 recovery. Real baselines are reported to show this
plainly. Complex must earn its place in Stage 2 (composition / phase additivity)
or we drop it for real low-rank.

Stage 1: recovery go/no-go (primary, well-validated result).
Stage 2: composition on chains d->b->c (exploratory; complex vs real low-rank).

Models (all predict the residual from centered a:  yhat = M @ a_c):
  complex_hollow      ComplEx-grounded M = Re(sum_k w_k e_hk conj(e_gk)), diag=0
  real_full_hollow    free real M, diag=0           (identifiability control)
  real_lowrank_hollow M = V U^T, diag=0             (param-matched to complex)
  diagonal_copy_cheat M = diag only                 (copy trap: R^2 but no direction)
  type_mean_only      M = 0                          (floor: residual=0)

CLAUDE.md alignment: standalone script, no edits to the cytokine_mil package,
no h5ad/manifest, runs on the cluster as a SLURM GPU job (cuda w/ CPU fallback).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

_LOG_FH = None


def _log(msg=""):
    print(msg, flush=True)
    if _LOG_FH is not None:
        print(msg, file=_LOG_FH, flush=True)


# ===========================================================================
# Simulation
# ===========================================================================
class GeneLayout:
    """Index bookkeeping for the simulated gene panel."""

    def __init__(self, n_edges, n_forks, n_types, markers_per_type, n_background):
        self.drivers = list(range(n_edges))
        c = n_edges
        self.responses = list(range(c, c + n_edges)); c += n_edges
        self.fork_responses = list(range(c, c + n_forks))
        self.fork_driver_of = {self.fork_responses[i]: self.drivers[i] for i in range(n_forks)}
        c += n_forks
        self.markers = {}
        for t in range(n_types):
            self.markers[t] = list(range(c, c + markers_per_type)); c += markers_per_type
        self.background = list(range(c, c + n_background)); c += n_background
        self.n_genes = c
        self.edges = {self.responses[k]: self.drivers[k] for k in range(n_edges)}
        self.edges.update(self.fork_driver_of)


def build_M_true(layout, beta):
    G = layout.n_genes
    M = np.zeros((G, G), dtype=np.float64)
    for resp, drv in layout.edges.items():
        M[resp, drv] = beta
    return M


def simulate_tubes(layout, n_tubes, n_cells, n_types, beta, noise_sigma, rng):
    """Cell-autonomous drivers / aggregate responses. Returns list of (X, cell_types)."""
    G = layout.n_genes
    mu = np.full((n_types, G), 0.10, dtype=np.float64)
    for t in range(n_types):
        for g in layout.markers[t]:
            mu[t, g] = 1.20
    for g in layout.background:
        mu[:, g] = 0.30
    drivers = np.array(layout.drivers)
    tubes = []
    for _ in range(n_tubes):
        cell_types = np.repeat(np.arange(n_types), n_cells // n_types)
        if len(cell_types) < n_cells:
            cell_types = np.concatenate([cell_types, rng.integers(0, n_types, n_cells - len(cell_types))])
        rng.shuffle(cell_types)
        N = len(cell_types)
        X = mu[cell_types].copy()
        D = rng.gamma(1.0, 1.0, size=(N, len(drivers)))            # cell-autonomous
        X[:, drivers] += D
        a_loo = (D.sum(0, keepdims=True) - D) / (N - 1)            # leave-one-out aggregate
        for resp, drv in layout.edges.items():
            k = layout.drivers.index(drv)
            X[:, resp] += beta * a_loo[:, k]
        X += rng.normal(0.0, noise_sigma, size=X.shape)
        X = np.clip(X, 0.0, None)
        tubes.append((X.astype(np.float32), cell_types.astype(np.int64)))
    return tubes


def simulate_chain_tubes(lc, n_tubes, n_cells, n_types, beta, noise_sigma, rng):
    """Stage-2 chains d -> b -> c (b responds to agg d; c to agg b)."""
    G = lc["n_genes"]
    d_idx = np.array(lc["d"]); b_idx = np.array(lc["b"]); c_idx = np.array(lc["c"])
    mu = np.full((n_types, G), 0.10, dtype=np.float64)
    for t, gs in lc["markers"].items():
        for g in gs:
            mu[t, g] = 1.20
    tubes = []
    loo = lambda Z, N: (Z.sum(0, keepdims=True) - Z) / (N - 1)
    for _ in range(n_tubes):
        cell_types = np.repeat(np.arange(n_types), n_cells // n_types)
        if len(cell_types) < n_cells:
            cell_types = np.concatenate([cell_types, rng.integers(0, n_types, n_cells - len(cell_types))])
        rng.shuffle(cell_types)
        N = len(cell_types)
        X = mu[cell_types].copy()
        D = rng.gamma(1.0, 1.0, size=(N, len(d_idx))); X[:, d_idx] += D
        B = beta * loo(D, N) + rng.normal(0, noise_sigma, size=(N, len(b_idx))); X[:, b_idx] += B
        C = beta * loo(B, N) + rng.normal(0, noise_sigma, size=(N, len(c_idx))); X[:, c_idx] += C
        X += rng.normal(0, noise_sigma, size=X.shape)
        X = np.clip(X, 0, None)
        tubes.append((X.astype(np.float32), cell_types.astype(np.int64)))
    M_true = np.zeros((G, G))
    for k in range(len(d_idx)):
        M_true[b_idx[k], d_idx[k]] = beta
        M_true[c_idx[k], b_idx[k]] = beta
    return tubes, M_true


def tubes_to_raw(tubes, n_types):
    """Leave-one-out samples. Returns A(raw loo mean), Y(raw cell), CT(type)."""
    A_list, Y_list, CT_list = [], [], []
    for X, ct in tubes:
        N = X.shape[0]
        A_list.append((X.sum(0, keepdims=True) - X) / (N - 1))
        Y_list.append(X)
        CT_list.append(ct)
    return (np.concatenate(A_list), np.concatenate(Y_list), np.concatenate(CT_list))


def fit_preprocess(A, Y, CT, n_types):
    a_mean = A.mean(0)
    typemean = np.zeros((n_types, Y.shape[1]), dtype=np.float64)
    for t in range(n_types):
        m = CT == t
        typemean[t] = Y[m].mean(0) if m.any() else Y.mean(0)
    return a_mean, typemean


def apply_preprocess(A, Y, CT, a_mean, typemean, device):
    Ac = (A - a_mean).astype(np.float32)
    Yr = (Y - typemean[CT]).astype(np.float32)
    return (torch.tensor(Ac, device=device), torch.tensor(Yr, device=device))


# ===========================================================================
# Models   (predict residual from centered a:  yhat = M @ a_c)
# ===========================================================================
class _Base(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def M(self):
        raise NotImplementedError

    def forward(self, a):
        return a @ self.M().T


class ComplexHollowBilinear(_Base):
    def __init__(self, G, d):
        super().__init__(G)
        s = 1.0 / np.sqrt(d)
        self.Er = nn.Parameter(torch.randn(G, d) * s)
        self.Ei = nn.Parameter(torch.randn(G, d) * s)
        self.wr = nn.Parameter(torch.randn(d) * s)
        self.wi = nn.Parameter(torch.randn(d) * s)
        self.register_buffer("offdiag", 1.0 - torch.eye(G))

    def _Mc(self):
        e = torch.complex(self.Er, self.Ei)
        w = torch.complex(self.wr, self.wi)
        return ((e.conj() * w) @ e.T) * self.offdiag   # [g,h]=sum_k conj(e_gk) w_k e_hk

    def M(self):
        return self._Mc().real

    def M_complex(self):
        return self._Mc()


class RealFullHollow(_Base):
    def __init__(self, G):
        super().__init__(G)
        self.W = nn.Parameter(torch.randn(G, G) * (1.0 / np.sqrt(G)))
        self.register_buffer("offdiag", 1.0 - torch.eye(G))

    def M(self):
        return self.W * self.offdiag


class RealLowRankHollow(_Base):
    def __init__(self, G, d):
        super().__init__(G)
        s = 1.0 / np.sqrt(d)
        self.U = nn.Parameter(torch.randn(G, d) * s)
        self.V = nn.Parameter(torch.randn(G, d) * s)
        self.register_buffer("offdiag", 1.0 - torch.eye(G))

    def M(self):
        return (self.V @ self.U.T) * self.offdiag


class DiagonalCopyCheat(_Base):
    """M = diagonal only: predicts gene g from the population's OWN gene g.
    The copy trap — earns R^2 on predictable genes but has zero off-diagonal,
    so direction AUC ~ 0.5."""

    def __init__(self, G):
        super().__init__(G)
        self.diag = nn.Parameter(torch.zeros(G))

    def M(self):
        return torch.diag(self.diag)


class _Zero(_Base):
    """M = 0: predicts the per-type mean (residual 0). Floor / chance direction."""

    def __init__(self, G, device):
        super().__init__(G)
        self._z = torch.zeros(G, G, device=device)

    def M(self):
        return self._z

    def forward(self, a):
        return torch.zeros(a.shape[0], self.G, device=a.device)


def train_model(model, A, Y, epochs, lr, weight_decay, name):
    params = list(model.parameters())
    if not params:  # parameter-free floor (M=0); nothing to optimise
        _log(f"    [{name}] no parameters — skipping optimisation (floor model)")
        return model
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    lossf = nn.MSELoss()
    n = A.shape[0]
    bs = min(4096, n)
    for ep in range(epochs):
        perm = torch.randperm(n, device=A.device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(A[idx]), Y[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            _log(f"    [{name}] epoch {ep:4d}  train_mse {tot / n:.4f}")
    return model


# ===========================================================================
# Evaluation
# ===========================================================================
def r2_score(model, A, Y):
    with torch.no_grad():
        pred = model(A)
        ss_res = ((Y - pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-9)


def direction_metrics(M_rec, layout):
    edges = [(r, d) for r, d in layout.edges.items()]
    pos = np.array([M_rec[r, d] for r, d in edges])
    neg = np.array([M_rec[d, r] for r, d in edges])
    auc = float(np.mean(pos[:, None] > neg[None, :]))
    sign_acc = float(np.mean(pos > neg))
    row_norm = np.linalg.norm(M_rec, axis=1)
    fork_asyms = []
    for fr, drv in layout.fork_driver_of.items():
        k = layout.drivers.index(drv); sib = layout.responses[k]
        fork_asyms.append(abs(M_rec[fr, sib] - M_rec[sib, fr]))
    bg = layout.background
    rng = np.random.default_rng(0)
    noedge = [abs(M_rec[i, j] - M_rec[j, i])
              for i, j in (rng.choice(bg, 2, replace=False) for _ in range(min(50, len(bg) * (len(bg) - 1))))]
    return {
        "direction_auc": auc,
        "direction_sign_acc": sign_acc,
        "mean_asym_planted": float(np.mean(pos - neg)),
        "mean_pos_edge": float(np.mean(pos)),
        "mean_reverse_edge": float(np.mean(neg)),
        "predictability_driver": float(np.mean([row_norm[d] for _, d in edges])),
        "predictability_response": float(np.mean([row_norm[r] for r, _ in edges])),
        "fork_control_asym": float(np.mean(fork_asyms)) if fork_asyms else float("nan"),
        "noedge_control_asym": float(np.mean(noedge)) if noedge else float("nan"),
    }


def recovery_corr(M_rec, M_true):
    mask = ~np.eye(M_true.shape[0], dtype=bool)
    a, b = M_rec[mask], M_true[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ===========================================================================
# Stages
# ===========================================================================
def run_stage1(args, device, out_dir, rng):
    _log("\n=== STAGE 1: direction recovery (go/no-go) ===")
    n_bg = max(4, args.n_genes - (2 * args.n_edges + args.n_forks + 6 * args.n_celltypes))
    layout = GeneLayout(args.n_edges, args.n_forks, args.n_celltypes, 6, n_bg)
    G = layout.n_genes
    _log(f"  genes={G} edges={args.n_edges} forks={args.n_forks} types={args.n_celltypes} beta={args.beta}")
    M_true = build_M_true(layout, args.beta)

    tr = simulate_tubes(layout, args.n_tubes, args.n_cells, args.n_celltypes, args.beta, args.noise_sigma, rng)
    te = simulate_tubes(layout, args.n_tubes // 3 + 1, args.n_cells, args.n_celltypes, args.beta, args.noise_sigma, rng)
    A_tr, Y_tr, CT_tr = tubes_to_raw(tr, args.n_celltypes)
    A_te, Y_te, CT_te = tubes_to_raw(te, args.n_celltypes)
    a_mean, typemean = fit_preprocess(A_tr, Y_tr, CT_tr, args.n_celltypes)
    A, Y = apply_preprocess(A_tr, Y_tr, CT_tr, a_mean, typemean, device)
    Ate, Yte = apply_preprocess(A_te, Y_te, CT_te, a_mean, typemean, device)
    _log(f"  train samples={A.shape[0]} test samples={Ate.shape[0]}")

    builders = {
        "complex_hollow": lambda: ComplexHollowBilinear(G, args.embed_dim),
        "real_full_hollow": lambda: RealFullHollow(G),
        "real_lowrank_hollow": lambda: RealLowRankHollow(G, args.embed_dim),
        "diagonal_copy_cheat": lambda: DiagonalCopyCheat(G),
        "type_mean_only": lambda: _Zero(G, device),
    }
    results, M_recs = {}, {}
    for name, build in builders.items():
        _log(f"  -- training {name}")
        model = build().to(device)
        train_model(model, A, Y, args.epochs, args.lr, args.weight_decay, name)
        M_rec = model.M().detach().cpu().numpy().astype(np.float64)
        M_recs[name] = M_rec
        m = direction_metrics(M_rec, layout)
        m["test_r2_residual"] = r2_score(model, Ate, Yte)
        m["recovery_corr"] = recovery_corr(M_rec, M_true)
        results[name] = m
        _log(f"     dir_AUC={m['direction_auc']:.3f} sign_acc={m['direction_sign_acc']:.3f} "
             f"recov_corr={m['recovery_corr']:.3f} test_R2={m['test_r2_residual']:.3f} "
             f"fork_asym={m['fork_control_asym']:.3f}")

    np.save(out_dir / "M_true_stage1.npy", M_true)
    for name, M_rec in M_recs.items():
        np.save(out_dir / f"M_rec_{name}.npy", M_rec)

    ch, cheat = results["complex_hollow"], results["type_mean_only"]
    rf = results["real_full_hollow"]
    results["_stage1_verdict"] = "GO" if (ch["direction_auc"] > 0.8 and cheat["direction_auc"] < 0.65) else "NO-GO"
    gap = ch["direction_auc"] - rf["direction_auc"]
    results["_complex_minus_realfull_auc"] = float(gap)
    results["_complex_justified_stage1"] = bool(gap > 0.05)
    _log(f"  STAGE 1 VERDICT: {results['_stage1_verdict']} "
         f"(complex AUC {ch['direction_auc']:.3f}, floor AUC {cheat['direction_auc']:.3f}, "
         f"real_full AUC {rf['direction_auc']:.3f})")
    _log(f"  complex - real_full AUC = {gap:+.3f} -> complex_justified_stage1={results['_complex_justified_stage1']} "
         "(False = real recovers direction just as well; complex must earn it in Stage 2)")
    if HAVE_MPL:
        _plot_stage1(out_dir, M_true, M_recs, results)
    return results


def run_stage2(args, device, out_dir, rng):
    _log("\n=== STAGE 2: composition on chains d->b->c (exploratory) ===")
    nc, nt = args.n_chains, args.n_celltypes
    d = list(range(nc)); b = list(range(nc, 2 * nc)); c = list(range(2 * nc, 3 * nc)); cc = 3 * nc
    markers = {}
    for t in range(nt):
        markers[t] = list(range(cc, cc + 4)); cc += 4
    lc = {"d": d, "b": b, "c": c, "markers": markers, "n_genes": cc + 10}
    tr, M_true = simulate_chain_tubes(lc, args.n_tubes, args.n_cells, nt, args.beta, args.noise_sigma, rng)
    A_tr, Y_tr, CT_tr = tubes_to_raw(tr, nt)
    a_mean, typemean = fit_preprocess(A_tr, Y_tr, CT_tr, nt)
    A, Y = apply_preprocess(A_tr, Y_tr, CT_tr, a_mean, typemean, device)
    G = lc["n_genes"]
    _log(f"  chains={nc} genes={G} train_samples={A.shape[0]}")

    out, models = {}, {}
    for name, build in {
        "complex_hollow": lambda: ComplexHollowBilinear(G, args.embed_dim),
        "real_lowrank_hollow": lambda: RealLowRankHollow(G, args.embed_dim),
    }.items():
        _log(f"  -- training {name}")
        model = build().to(device)
        train_model(model, A, Y, args.epochs, args.lr, args.weight_decay, name)
        M_rec = model.M().detach().cpu().numpy().astype(np.float64)
        models[name] = model
        fwd = np.mean([M_rec[b[k], d[k]] for k in range(nc)] + [M_rec[c[k], b[k]] for k in range(nc)])
        rev = np.mean([M_rec[d[k], b[k]] for k in range(nc)] + [M_rec[b[k], c[k]] for k in range(nc)])
        trans_acc = float(np.mean([M_rec[c[k], d[k]] > M_rec[d[k], c[k]] for k in range(nc)]))
        trans_asym = float(np.mean([M_rec[c[k], d[k]] - M_rec[d[k], c[k]] for k in range(nc)]))
        out[name] = {"chain_fwd_mean": float(fwd), "chain_rev_mean": float(rev),
                     "transitive_dc_direction_acc": trans_acc, "transitive_dc_asym_mean": trans_asym}
        _log(f"     chain fwd={fwd:.3f} rev={rev:.3f} | transitive d->c acc={trans_acc:.3f} asym={trans_asym:.3f}")

    Mc = models["complex_hollow"].M_complex().detach().cpu().numpy()
    ph = lambda g, h: np.angle(Mc[g, h])
    add = np.array([ph(b[k], d[k]) + ph(c[k], b[k]) for k in range(nc)])
    direct = np.array([ph(c[k], d[k]) for k in range(nc)])
    diff = np.angle(np.exp(1j * (direct - add)))
    out["complex_hollow"]["phase_additivity_circ_std"] = float(np.std(diff))
    _log(f"  phase additivity circ-std (0=perfect)={np.std(diff):.3f}")
    np.save(out_dir / "M_true_stage2.npy", M_true)
    out["_stage2_decision"] = ("complex_helps"
                               if out["complex_hollow"]["transitive_dc_direction_acc"]
                               > out["real_lowrank_hollow"]["transitive_dc_direction_acc"] + 0.1
                               else "complex_not_clearly_better")
    _log(f"  STAGE 2 DECISION: {out['_stage2_decision']}")
    if HAVE_MPL:
        _plot_stage2(out_dir, direct, add)
    return out


def _plot_stage1(out_dir, M_true, M_recs, results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmax = float(np.percentile(np.abs(M_true), 99)) or 1.0
    axes[0].imshow(M_true, cmap="coolwarm", vmin=-vmax, vmax=vmax); axes[0].set_title("M_true")
    axes[1].imshow(M_recs["complex_hollow"], cmap="coolwarm", vmin=-vmax, vmax=vmax); axes[1].set_title("M_rec complex_hollow")
    names = [n for n in results if not n.startswith("_")]
    axes[2].barh(names, [results[n]["direction_auc"] for n in names])
    axes[2].axvline(0.5, color="k", ls="--", lw=1); axes[2].set_xlim(0, 1); axes[2].set_title("direction AUC")
    fig.tight_layout(); fig.savefig(out_dir / "stage1_summary.png", dpi=120); plt.close(fig)


def _plot_stage2(out_dir, direct, add):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(add, direct, alpha=0.7); ax.plot([-np.pi, np.pi], [-np.pi, np.pi], "k--", lw=1)
    ax.set_xlabel("phase(d->b)+phase(b->c)"); ax.set_ylabel("phase(d->c)")
    ax.set_title("phase additivity (complex)")
    fig.tight_layout(); fig.savefig(out_dir / "stage2_phase_additivity.png", dpi=120); plt.close(fig)


# ===========================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="toy complex directional gene influence")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", choices=["1", "2", "both"], default="both")
    p.add_argument("--n_genes", type=int, default=120)
    p.add_argument("--n_edges", type=int, default=16)
    p.add_argument("--n_forks", type=int, default=4)
    p.add_argument("--n_celltypes", type=int, default=4)
    p.add_argument("--n_tubes", type=int, default=150)
    p.add_argument("--n_cells", type=int, default=60)
    p.add_argument("--beta", type=float, default=12.0)
    p.add_argument("--noise_sigma", type=float, default=0.3)
    p.add_argument("--embed_dim", type=int, default=48)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--n_chains", type=int, default=12)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    global _LOG_FH
    args = _parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else \
        REPO_ROOT / "results" / "toy_complex_gene_direction" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FH = open(out_dir / "train.log", "w")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    _log(f"toy_complex_gene_direction | seed={args.seed} device={device} matplotlib={HAVE_MPL}")
    _log(f"args: {vars(args)}")

    res = {"args": vars(args), "device": str(device)}
    if args.stage in ("1", "both"):
        res["stage1"] = run_stage1(args, device, out_dir, rng)
    if args.stage in ("2", "both"):
        res["stage2"] = run_stage2(args, device, out_dir, rng)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    _log(f"\nDONE in {res['elapsed_sec']}s -> {out_dir}")
    _LOG_FH.close()


if __name__ == "__main__":
    main()
