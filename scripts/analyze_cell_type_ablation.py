"""
Cell-type ablation test for cascade directionality.

For each tube of cytokine A and each target cytokine B:
  relay_score(A→B, T) = P(B | full tube) - P(B | tube with cell type T removed)

Positive score: removing T reduces the tube's "B-likeness" → T carries the
  signal that makes A-tubes look B-like → T is a relay candidate for A→B.

Direction test for pair (A, B):
  If relay_score(A→B, T*) > relay_score(B→A, T*) for the best T → A→B.
  Wilcoxon on relay_score across tubes (within training donors).

Outputs (to --output_dir):
  ablation_scores.pkl          - full relay_score tensors per seed
  ablation_directionality.csv  - per-pair directionality calls
  ablation_known_pairs.png     - relay_score bar chart for known cascade pairs
"""
import argparse
import pickle
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import anndata
from scipy import stats

from cytokine_mil.experiment_setup import build_mil_model, build_encoder
from cytokine_mil.data.label_encoder import CytokineLabel


KNOWN_CASCADES = [
    ("IL-12",      "IFN-gamma"),
    ("IL-1-beta",  "IL-6"),
    ("IL-2",       "IL-15"),
    ("IL-33",      "IL-13"),
    ("IL-18",      "IFN-gamma"),
    ("IL-21",      "IL-10"),
    ("TNF-alpha",  "IL-6"),
    ("IFN-alpha1", "IFN-gamma"),
    ("IL-10",      "IL-6"),
    ("IL-4",       "IL-13"),
    ("IL-27",      "IFN-gamma"),
]

VAL_DONORS = {"Donor2", "Donor3"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs",   nargs="+", required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--n_genes",     type=int, default=4000)
    p.add_argument("--embed_dim",   type=int, default=128)
    p.add_argument("--attn_dim",    type=int, default=64)
    p.add_argument("--n_classes",   type=int, default=91)
    p.add_argument("--pairs", nargs="*", default=None,
                   help="Restrict to specific pairs, e.g. 'IL-12:IFN-gamma'. "
                        "Default: all known cascade pairs.")
    return p.parse_args()


def load_model(seed_dir, n_genes, embed_dim, attn_dim, n_classes, device):
    encoder = build_encoder(n_input_genes=n_genes, n_cell_types=18,
                            embed_dim=embed_dim)
    model   = build_mil_model(encoder, embed_dim=embed_dim,
                               attention_hidden_dim=attn_dim,
                               n_classes=n_classes, encoder_frozen=False)
    state   = torch.load(seed_dir / "model_stage3.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def load_label_encoder(seed_dir):
    """Load from {"cytokines": [...]} JSON format saved by train_oesinghaus_full."""
    with open(seed_dir / "label_encoder.json") as f:
        data = json.load(f)
    cytos = data["cytokines"]
    le = CytokineLabel()
    le._label_to_idx = {c: i for i, c in enumerate(cytos)}
    le._idx_to_label = {i: c for i, c in enumerate(cytos)}
    return le


def load_manifest(seed_dir):
    with open(seed_dir / "manifest_train.json") as f:
        return json.load(f)


@torch.no_grad()
def get_softmax(model, X_tensor, device):
    """Run model on (N, G) tensor, return (K,) softmax."""
    y_hat, _, _ = model(X_tensor.to(device))
    return torch.softmax(y_hat, dim=-1).squeeze().cpu().numpy()


def load_tube(path):
    """Return (X: float32 ndarray, cell_types: list)."""
    adata = anndata.read_h5ad(path)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)
    ct = (adata.obs["cell_type"].values
          if "cell_type" in adata.obs.columns
          else np.array(["unknown"] * X.shape[0]))
    return X, ct


def ablate_tube(model, X, cell_types, le, target_cyto, device):
    """
    Returns dict: cell_type_name -> relay_score for target_cyto.
    relay_score = P(target | full) - P(target | tube_without_this_ct)
    """
    j = le.encode(target_cyto)
    X_t = torch.FloatTensor(X)
    p_full = get_softmax(model, X_t, device)[j]

    unique_cts = np.unique(cell_types)
    scores = {}
    for ct in unique_cts:
        mask = cell_types != ct
        if mask.sum() < 5:          # skip if fewer than 5 cells remain
            continue
        X_sub = torch.FloatTensor(X[mask])
        p_drop = get_softmax(model, X_sub, device)[j]
        scores[ct] = float(p_full - p_drop)
    return scores


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cpu")

    pairs = KNOWN_CASCADES
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]

    seed_dirs = [Path(d) for d in args.seed_dirs]

    # scores[seed_name][(A, B, T)] = list of relay scores across tubes
    all_scores = {sd.name: {} for sd in seed_dirs}

    for sd in seed_dirs:
        print(f"\n=== {sd.name} ===", flush=True)
        le       = load_label_encoder(sd)
        manifest = load_manifest(sd)
        model    = load_model(sd, args.n_genes, args.embed_dim,
                               args.attn_dim, args.n_classes, device)

        # Only training donors
        train_entries = [e for e in manifest
                         if e.get("donor", "") not in VAL_DONORS]

        for pair_a, pair_b in pairs:
            if pair_a not in le.cytokines or pair_b not in le.cytokines:
                print(f"  SKIP {pair_a}→{pair_b}: not in label set")
                continue

            # Collect relay scores for A-tubes (target = B) and B-tubes (target = A)
            for source, target in [(pair_a, pair_b), (pair_b, pair_a)]:
                source_tubes = [e for e in train_entries
                                if e["cytokine"] == source]
                print(f"  {source}→{target}: {len(source_tubes)} tubes",
                      flush=True)
                for entry in source_tubes:
                    try:
                        X, cts = load_tube(entry["path"])
                    except Exception as e:
                        print(f"    WARN {Path(entry['path']).name}: {e}")
                        continue
                    ct_scores = ablate_tube(model, X, cts, le, target, device)
                    for ct, sc in ct_scores.items():
                        key = (source, target, ct)
                        all_scores[sd.name].setdefault(key, []).append(sc)

    # ── Aggregate across seeds ────────────────────────────────────────────────
    # For each (A, B, T): pool scores across seeds → Wilcoxon
    from collections import defaultdict
    pooled = defaultdict(list)
    for seed_name, scores in all_scores.items():
        for key, vals in scores.items():
            pooled[key].extend(vals)

    rows = []
    for (source, target, ct), vals in pooled.items():
        vals = np.array(vals)
        if len(vals) < 3:
            continue
        stat, pval = stats.wilcoxon(vals, alternative="greater") \
            if len(np.unique(vals)) > 1 \
            else (0.0, 1.0)
        rows.append({
            "source": source, "target": target, "cell_type": ct,
            "mean_relay_score": float(vals.mean()),
            "median_relay_score": float(np.median(vals)),
            "n": len(vals),
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(pval),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results — check that pair cytokines exist in label set.")
        return

    df.to_csv(out_dir / "ablation_scores.csv", index=False)

    # ── Save pkl ──────────────────────────────────────────────────────────────
    with open(out_dir / "ablation_scores.pkl", "wb") as f:
        pickle.dump({"all_scores": all_scores, "pooled": dict(pooled)}, f)

    # ── Direction calls ───────────────────────────────────────────────────────
    print(f"\n{'Pair':<35}  {'Best relay T':<22}  {'Fwd score':>10}  {'Rev score':>10}  {'Call':>8}")
    print("-" * 90)
    dir_rows = []
    for a, b in pairs:
        fwd = df[(df.source == a) & (df.target == b)]
        rev = df[(df.source == b) & (df.target == a)]
        if fwd.empty or rev.empty:
            print(f"  {a}→{b}: insufficient data")
            continue
        best_fwd = fwd.loc[fwd.mean_relay_score.idxmax()]
        best_rev = rev.loc[rev.mean_relay_score.idxmax()]
        # best relay T by combined (one that is relay in fwd but not rev)
        fwd_top = fwd.sort_values("mean_relay_score", ascending=False).iloc[0]
        rev_top = rev.sort_values("mean_relay_score", ascending=False).iloc[0]
        score_fwd = float(fwd_top.mean_relay_score)
        score_rev = float(rev_top.mean_relay_score)
        relay_t   = fwd_top.cell_type
        call = "A→B" if score_fwd > score_rev else "B→A" if score_rev > score_fwd else "shared"
        print(f"  {a:<15} → {b:<15}  {relay_t:<22}  {score_fwd:>10.5f}  {score_rev:>10.5f}  {call:>8}")
        dir_rows.append({"A": a, "B": b, "relay_T": relay_t,
                          "score_A_to_B": score_fwd, "score_B_to_A": score_rev,
                          "call": call})

    pd.DataFrame(dir_rows).to_csv(out_dir / "ablation_directionality.csv", index=False)

    # ── Plot known pairs ──────────────────────────────────────────────────────
    n_pairs = len(dir_rows)
    if n_pairs == 0:
        print("No pairs to plot."); return

    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 5), sharey=False)
    if n_pairs == 1:
        axes = [axes]

    for ax, row in zip(axes, dir_rows):
        a, b = row["A"], row["B"]
        fwd  = df[(df.source == a) & (df.target == b)].sort_values(
                   "mean_relay_score", ascending=False)
        rev  = df[(df.source == b) & (df.target == a)].sort_values(
                   "mean_relay_score", ascending=False)

        cts_shown = fwd.cell_type.tolist()[:8]  # top 8 by fwd score
        fwd_scores = fwd.set_index("cell_type").reindex(cts_shown).mean_relay_score.fillna(0)
        rev_scores = rev.set_index("cell_type").reindex(cts_shown).mean_relay_score.fillna(0)

        x = np.arange(len(cts_shown))
        w = 0.35
        ax.bar(x - w/2, fwd_scores, w, color="#e74c3c", alpha=0.8, label=f"{a}→{b}")
        ax.bar(x + w/2, rev_scores, w, color="#3498db", alpha=0.8, label=f"{b}→{a}")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(cts_shown, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Mean relay score\n(P(target|full) − P(target|no T))", fontsize=8)
        ax.set_title(f"{a} → {b}\n[{row['call']}]", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Cell-type ablation relay scores\n"
        "Positive = removing this cell type reduces target-cytokine probability\n"
        "Red bar > Blue bar → A→B directionality",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_known_pairs.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
