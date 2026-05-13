"""
Confusion matrix asymmetry for cascade directionality.

For each seed: load model_stage3.pt, run forward pass on all training tubes,
build mean-softmax confusion matrix C[i,j] = mean P(class j | tube of class i).

Asymmetry: Asym(A→B) = C[A,B] - C[B,A]
  Positive → A-tubes look more B-like than B-tubes look A-like → A→B.

Outputs (to --output_dir):
  confusion_asymmetry.pkl  - per-seed and mean matrices
  confusion_asymmetry.csv  - all ordered pairs ranked by asymmetry
  confusion_asymmetry.png  - full heatmap
"""
import argparse
import pickle
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import anndata

from cytokine_mil.experiment_setup import build_mil_model, build_encoder
from cytokine_mil.data.label_encoder import CytokineLabel


# ── Known cascade pairs (for annotation in output) ───────────────────────────
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_genes",    type=int, default=4000)
    p.add_argument("--embed_dim",  type=int, default=128)
    p.add_argument("--attn_dim",   type=int, default=64)
    p.add_argument("--n_classes",  type=int, default=91)
    return p.parse_args()


def load_model(seed_dir, n_genes, embed_dim, attn_dim, n_classes, device):
    encoder = build_encoder(n_input_genes=n_genes, n_cell_types=18,
                            embed_dim=embed_dim)
    model = build_mil_model(encoder, embed_dim=embed_dim,
                            attention_hidden_dim=attn_dim,
                            n_classes=n_classes, encoder_frozen=False)
    state = torch.load(seed_dir / "model_stage3.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def load_label_encoder(seed_dir):
    """Load from {"cytokines": [...]} JSON format saved by train_oesinghaus_full."""
    with open(seed_dir / "label_encoder.json") as f:
        data = json.load(f)
    cytos = data["cytokines"]          # ordered list; index = class index
    le = CytokineLabel()
    le._label_to_idx = {c: i for i, c in enumerate(cytos)}
    le._idx_to_label = {i: c for i, c in enumerate(cytos)}
    return le


def load_manifest(seed_dir):
    with open(seed_dir / "manifest_train.json") as f:
        return json.load(f)


@torch.no_grad()
def tube_softmax(model, path, device):
    """Return (K,) mean softmax over one tube."""
    adata = anndata.read_h5ad(path)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = torch.FloatTensor(np.array(X, dtype=np.float32)).to(device)
    y_hat, _, _ = model(X)
    return torch.softmax(y_hat, dim=-1).squeeze().cpu().numpy()


def build_confusion(model, manifest, le, device, n_classes):
    acc = np.zeros((n_classes, n_classes), dtype=np.float64)
    cnt = np.zeros(n_classes, dtype=np.int64)
    for entry in manifest:
        cyto = entry["cytokine"]
        try:
            idx = le.encode(cyto)
        except Exception:
            continue
        try:
            probs = tube_softmax(model, entry["path"], device)
        except Exception as e:
            print(f"  WARN {Path(entry['path']).name}: {e}")
            continue
        acc[idx] += probs
        cnt[idx] += 1
    conf = np.zeros_like(acc)
    mask = cnt > 0
    conf[mask] = acc[mask] / cnt[mask, np.newaxis]
    return conf


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cpu")

    seed_dirs = [Path(d) for d in args.seed_dirs]
    all_conf  = []
    le        = None

    for sd in seed_dirs:
        print(f"\n=== {sd.name} ===", flush=True)
        le       = load_label_encoder(sd)
        manifest = load_manifest(sd)
        model    = load_model(sd, args.n_genes, args.embed_dim,
                               args.attn_dim, args.n_classes, device)
        conf = build_confusion(model, manifest, le, device, args.n_classes)
        all_conf.append(conf)
        print(f"  done – n_tubes={len(manifest)}", flush=True)

    all_conf  = np.array(all_conf)          # (n_seeds, K, K)
    mean_conf = all_conf.mean(axis=0)       # (K, K)
    asym      = mean_conf - mean_conf.T     # Asym[i,j] = P(j|i) - P(i|j)

    cytos = le.cytokines   # ordered list

    # ── Save pkl ──────────────────────────────────────────────────────────────
    with open(out_dir / "confusion_asymmetry.pkl", "wb") as f:
        pickle.dump({"per_seed": all_conf, "mean": mean_conf,
                     "asym": asym, "cytokines": cytos}, f)

    # ── CSV ───────────────────────────────────────────────────────────────────
    rows = []
    for i, ca in enumerate(cytos):
        for j, cb in enumerate(cytos):
            if i == j:
                continue
            known_fwd = (ca, cb) in KNOWN_CASCADES
            known_rev = (cb, ca) in KNOWN_CASCADES
            rows.append({
                "A": ca, "B": cb,
                "P_B_given_A":    float(mean_conf[i, j]),
                "P_A_given_B":    float(mean_conf[j, i]),
                "asym_A_to_B":    float(asym[i, j]),
                "known_fwd":      known_fwd,
                "known_rev":      known_rev,
            })
    df = pd.DataFrame(rows).sort_values("asym_A_to_B", ascending=False)
    df.to_csv(out_dir / "confusion_asymmetry.csv", index=False)
    print(f"\nTop 20 asymmetric pairs:")
    print(df[["A", "B", "asym_A_to_B", "known_fwd"]].head(20).to_string(index=False))

    # ── Print known cascade recovery ──────────────────────────────────────────
    print(f"\n{'Pair':<35}  {'Asym(A→B)':>10}  {'Asym(B→A)':>10}  {'Direction':>10}")
    print("-" * 70)
    for a, b in KNOWN_CASCADES:
        if a not in cytos or b not in cytos:
            print(f"  {a}→{b}: NOT IN LABEL SET")
            continue
        i, j = cytos.index(a), cytos.index(b)
        ab = asym[i, j]
        ba = asym[j, i]
        call = "A→B" if ab > 0 else ("B→A" if ba > 0 else "shared/none")
        print(f"  {a:<15} → {b:<15}  {ab:>10.5f}  {ba:>10.5f}  {call:>10}")

    # ── Heatmap (exclude PBS) ─────────────────────────────────────────────────
    pbs_idx = cytos.index("PBS") if "PBS" in cytos else None
    idxs    = [i for i in range(len(cytos)) if i != pbs_idx]
    sub     = asym[np.ix_(idxs, idxs)]
    labels  = [cytos[i] for i in idxs]

    fig, ax = plt.subplots(figsize=(24, 20))
    vmax = np.percentile(np.abs(sub), 98)
    im   = ax.imshow(sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(
        f"Confusion asymmetry: P(B|A-tube) − P(A|B-tube)\n"
        f"Blue = A→B (A-tubes confused with B more than reverse)\n"
        f"Mean across {len(seed_dirs)} seeds  |  model_stage3",
        fontsize=11, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Asym(A→B)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_asymmetry.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
