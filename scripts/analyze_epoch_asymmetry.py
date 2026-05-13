"""
Epoch-dependent PBS-RC directional bias asymmetry.

Uses existing dynamics_stage3.pkl centroid trajectory data — NO new experiments.

For each epoch t, pair (A, B), cell type T, training donor d:
  b_fwd(A→B, T, d, t) = (mu_tilde_{A,T}(d,t) - mu_tilde_A(d,t)) · u_hat_{A→B}(t)

where:
  mu_tilde_{A,T}(d,t)  = attention-weighted centroid of cell type T in A-tubes
                          for training donor d, at epoch t
  mu_tilde_A(d,t)      = mean of mu_tilde_{A,T}(d,t) across all cell types T
                          (A's "global" embedding at epoch t)
  u_hat_{A→B}(t)       = (mu_tilde_B(t) - mu_tilde_A(t)) / ||...||
                          (global direction from A to B at epoch t)

Asymmetry: Asym(A→B, T, t) = b_fwd(A→B, T, t) - b_fwd(B→A, T, t)

Hypothesis: if A→B, Asym is positive at early epochs (model has learned A's
direct signal + relay T cells shifting toward B, but not yet the reverse-cascade).
As training continues, if B also feeds back to A, symmetry is restored.

Outputs (to --output_dir):
  epoch_asymmetry.pkl          - full bias arrays per seed/pair/cell_type/epoch
  epoch_asymmetry_<A>_<B>.png  - Asym vs epoch for each known pair
  epoch_asymmetry_summary.csv  - max asymmetry + direction call per pair
"""
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


KNOWN_CASCADES = [
    ("IL-12",     "IFN-gamma"),
    ("IL-1beta",  "IL-6"),
    ("IL-2",      "IL-15"),
    ("IL-33",     "IL-13"),
    ("IL-18",     "IFN-gamma"),
    ("IL-21",     "IL-10"),
    ("TNF",       "IL-6"),
    ("IFN-alpha", "IFN-gamma"),
    ("IL-10",     "IL-6"),
    ("IL-4",      "IL-13"),
    ("IL-27",     "IFN-gamma"),
]

VAL_DONORS = {"Donor2", "Donor3"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs",  nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--traj_key",   default="attn_centroid_trajectory",
                   choices=["attn_centroid_trajectory", "centroid_trajectory"])
    return p.parse_args()


def load_dynamics(seed_dir, traj_key):
    with open(Path(seed_dir) / "dynamics_stage3.pkl", "rb") as f:
        d = pickle.load(f)
    return d[traj_key], d["centroid_logged_epochs"]


def get_meta(traj):
    keys = list(traj[0].keys())
    cell_types = sorted(set(ct for (_, ct, _) in keys))
    cytokines  = sorted(set(cy for (cy, _, _) in keys))
    donors     = sorted(set(do for (_, _, do) in keys))
    return cell_types, cytokines, donors


def centroid_at(snap, cyto, ct, donors):
    """Mean centroid for (cyto, ct) over given donors at one snapshot."""
    vecs = [snap[(cyto, ct, d)] for d in donors if (cyto, ct, d) in snap]
    return np.mean(vecs, axis=0) if vecs else None


def compute_bias_trajectory(traj, epochs, cyto_a, cyto_b, cell_types,
                             train_donors):
    """
    Returns b_fwd_A (n_epochs, n_ct) and b_fwd_B (n_epochs, n_ct) arrays.
    b_fwd_A[t, i] = b_fwd(A→B, cell_types[i], pooled_donors, epoch t)
    """
    n_ep = len(epochs)
    n_ct = len(cell_types)
    b_fwd_A = np.full((n_ep, n_ct), np.nan)
    b_fwd_B = np.full((n_ep, n_ct), np.nan)

    for t, snap in enumerate(traj):
        # Global centroids at this epoch (mean across cell types, pooled donors)
        mu_A_vecs = [centroid_at(snap, cyto_a, ct, train_donors)
                     for ct in cell_types]
        mu_B_vecs = [centroid_at(snap, cyto_b, ct, train_donors)
                     for ct in cell_types]
        mu_A_vecs = [v for v in mu_A_vecs if v is not None]
        mu_B_vecs = [v for v in mu_B_vecs if v is not None]
        if not mu_A_vecs or not mu_B_vecs:
            continue
        mu_A = np.mean(mu_A_vecs, axis=0)   # global centroid of cyto A
        mu_B = np.mean(mu_B_vecs, axis=0)   # global centroid of cyto B

        diff = mu_B - mu_A
        norm = np.linalg.norm(diff)
        if norm < 1e-10:
            continue
        u_AB = diff / norm                  # direction A→B
        u_BA = -u_AB                        # direction B→A

        for i, ct in enumerate(cell_types):
            mu_AT = centroid_at(snap, cyto_a, ct, train_donors)
            mu_BT = centroid_at(snap, cyto_b, ct, train_donors)
            if mu_AT is not None:
                b_fwd_A[t, i] = np.dot(mu_AT - mu_A, u_AB)
            if mu_BT is not None:
                b_fwd_B[t, i] = np.dot(mu_BT - mu_B, u_BA)

    return b_fwd_A, b_fwd_B


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = [Path(d) for d in args.seed_dirs]

    # Collect b_fwd per seed, per pair
    # results[(A,B)][seed_idx] = (b_fwd_A, b_fwd_B)  arrays (n_ep, n_ct)
    from collections import defaultdict
    results    = defaultdict(dict)
    epochs_ref = None
    cell_types_ref = None

    for s_idx, sd in enumerate(seed_dirs):
        traj, epochs = load_dynamics(sd, args.traj_key)
        cell_types, cytokines, donors = get_meta(traj)
        train_donors = [d for d in donors if d not in VAL_DONORS]

        if epochs_ref is None:
            epochs_ref     = epochs
            cell_types_ref = cell_types

        for cyto_a, cyto_b in KNOWN_CASCADES:
            if cyto_a not in cytokines or cyto_b not in cytokines:
                continue
            b_fwd_A, b_fwd_B = compute_bias_trajectory(
                traj, epochs, cyto_a, cyto_b, cell_types, train_donors)
            results[(cyto_a, cyto_b)][s_idx] = (b_fwd_A, b_fwd_B)

        print(f"Done: {sd.name}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(out_dir / "epoch_asymmetry.pkl", "wb") as f:
        pickle.dump({"results": dict(results),
                     "epochs": epochs_ref,
                     "cell_types": cell_types_ref}, f)

    # ── Per-pair plot ─────────────────────────────────────────────────────────
    n_ct   = len(cell_types_ref)
    epochs = np.array(epochs_ref)
    summary_rows = []

    for (cyto_a, cyto_b), seed_data in results.items():
        if not seed_data:
            continue

        # Mean b_fwd_A and b_fwd_B across seeds (n_ep, n_ct)
        b_A_seeds = np.array([v[0] for v in seed_data.values()])  # (n_seeds,n_ep,n_ct)
        b_B_seeds = np.array([v[1] for v in seed_data.values()])
        mean_bA = np.nanmean(b_A_seeds, axis=0)   # (n_ep, n_ct)
        mean_bB = np.nanmean(b_B_seeds, axis=0)
        # Asymmetry per cell type
        asym = mean_bA - mean_bB                   # (n_ep, n_ct)

        # Pick top-4 cell types by max |asym| over epochs
        max_asym_per_ct = np.nanmax(np.abs(asym), axis=0)
        top4_idx = np.argsort(max_asym_per_ct)[::-1][:4]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: b_fwd_A and b_fwd_B per cell type over epochs
        ax = axes[0]
        cmap = plt.cm.tab10
        for rank, ci in enumerate(top4_idx):
            ct = cell_types_ref[ci]
            color = cmap(rank)
            ax.plot(epochs, mean_bA[:, ci], "-o", color=color, ms=4,
                    label=f"{ct} [A]")
            ax.plot(epochs, mean_bB[:, ci], "--s", color=color, ms=4,
                    alpha=0.6, label=f"{ct} [B]")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xlabel("Epoch"); ax.set_ylabel("b_fwd")
        ax.set_title(f"b_fwd(A) = (µ_{{A,T}} − µ_A)·û_{{A→B}}\n"
                     f"b_fwd(B) = (µ_{{B,T}} − µ_B)·û_{{B→A}}\n"
                     f"[top 4 cell types by max |asymmetry|]")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        # Right: asymmetry = b_fwd_A - b_fwd_B per cell type
        ax = axes[1]
        for rank, ci in enumerate(top4_idx):
            ct   = cell_types_ref[ci]
            color = cmap(rank)
            ax.plot(epochs, asym[:, ci], "-o", color=color, ms=4, label=ct)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Asym(A→B, T, t) = b_fwd_A − b_fwd_B")
        ax.set_title("Asymmetry over epochs\n"
                     "Positive = A→B (A-relay signal > B-relay signal)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Direction call: mean asymmetry averaged over epochs and top-4 cell types
        mean_asym_overall = float(np.nanmean(asym[:, top4_idx]))
        call = "A→B" if mean_asym_overall > 0 else "B→A"

        fig.suptitle(
            f"Epoch asymmetry  |  {cyto_a} → {cyto_b}\n"
            f"Mean Asym = {mean_asym_overall:.4f}  →  Call: {call}\n"
            f"Mean across {len(seed_data)} seeds  |  {args.traj_key}",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        a_safe = cyto_a.replace("-", "_").replace(" ", "_")
        b_safe = cyto_b.replace("-", "_").replace(" ", "_")
        plt.savefig(out_dir / f"epoch_asym_{a_safe}_{b_safe}.png",
                    dpi=130, bbox_inches="tight")
        plt.close()

        # Best relay cell type (max mean positive asym)
        mean_asym_per_ct = np.nanmean(asym, axis=0)
        best_ct_idx = int(np.nanargmax(mean_asym_per_ct))
        best_ct     = cell_types_ref[best_ct_idx]

        summary_rows.append({
            "A": cyto_a, "B": cyto_b,
            "mean_asym_all_ct_all_epochs": mean_asym_overall,
            "best_relay_ct": best_ct,
            "best_ct_mean_asym": float(mean_asym_per_ct[best_ct_idx]),
            "n_seeds": len(seed_data),
            "call": call,
        })
        print(f"  {cyto_a}→{cyto_b}  Asym={mean_asym_overall:.4f}  "
              f"best_relay={best_ct}  call={call}", flush=True)

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / "epoch_asymmetry_summary.csv", index=False)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Pair':<35}  {'Mean Asym':>10}  {'Best relay T':<22}  {'Call':>8}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"  {row.A:<15} → {row.B:<15}  "
              f"{row.mean_asym_all_ct_all_epochs:>10.4f}  "
              f"{row.best_relay_ct:<22}  {row.call:>8}")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
