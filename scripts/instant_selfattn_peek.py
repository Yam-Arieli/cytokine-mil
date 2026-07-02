"""
Instant §34 peek: from the LATEST checkpoint only + a few tubes per key cytokine,
print G0 (do cells interact?) + the interaction-direction D for the known cascades.
No trajectory, no full subset — a fast interim read while training runs.

Usage:
    python scripts/instant_selfattn_peek.py --run_dir results/selfattn_partial/seed_42
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import anndata, numpy as np, torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_selfattn_model

HVG = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
KEY = ["IL-12", "IFN-gamma", "IL-2", "IL-15", "TNF-alpha", "IL-6"]
CASCADES = [("IL-12", "IFN-gamma"), ("IL-2", "IFN-gamma"), ("IL-15", "IFN-gamma")]
CONTROL = [("IL-6", "TNF-alpha")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--hvg_path", default=HVG)
    ap.add_argument("--n_per_cyt", type=int, default=3)
    args = ap.parse_args()
    rd = Path(args.run_dir)
    dev = torch.device("cpu")

    ck = sorted((rd / "checkpoints").glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    ckpt = ck[-1]
    epoch = int(ckpt.stem.split("_")[1])
    genes = json.load(open(args.hvg_path))
    le = CytokineLabel(); d = json.load(open(rd / "label_encoder.json"))
    le._label_to_idx = {c: i for i, c in enumerate(d["cytokines"])}
    le._idx_to_label = {i: c for i, c in enumerate(d["cytokines"])}
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    nct = state["encoder.cell_type_head.weight"].shape[0]
    enc = build_encoder(len(genes), n_cell_types=nct, embed_dim=128)
    m = build_selfattn_model(enc, embed_dim=128, attention_hidden_dim=64,
                             n_classes=le.n_classes(), encoder_frozen=True, sab_heads=4, sab_layers=1)
    m.load_state_dict(state); m.eval()

    entries = json.load(open(rd / "manifest_train.json"))
    by_cyt = defaultdict(list)
    for e in entries:
        if e["cytokine"] in KEY:
            by_cyt[e["cytokine"]].append(e)

    # per cytokine: collect off-diag fraction, pooling-primary, and M[τ,σ]
    offdiag_all, primary, Msum, Mcnt = [], {}, defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))
    pool_ct = defaultdict(lambda: defaultdict(list))
    for cyt, es in by_cyt.items():
        for e in sorted(es, key=lambda x: x.get("tube_idx", 0))[:args.n_per_cyt]:
            ad = anndata.read_h5ad(e["path"])
            avail = [g for g in genes if g in ad.var_names]
            X = ad[:, avail].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            cts = np.array(ad.obs["cell_type"].astype(str))
            with torch.no_grad():
                _, a, _, A = m.forward_with_interaction(torch.tensor(X, dtype=torch.float32))
            a = a.numpy(); A = A.numpy()
            # off-diagonal (cross-type) fraction per cell
            within = np.zeros(len(cts))
            for t in np.unique(cts):
                mt = cts == t
                within[mt] = A[mt][:, mt].sum(1)
            offdiag_all.append(float(np.mean(1 - within)))
            for t in np.unique(cts):
                pool_ct[cyt][t].append(float(a[cts == t].mean()))
                for s in np.unique(cts):
                    Msum[cyt][(t, s)] += float(A[cts == t][:, cts == s].sum(1).mean())
                    Mcnt[cyt][(t, s)] += 1
    for cyt in pool_ct:
        primary[cyt] = max(pool_ct[cyt], key=lambda t: np.mean(pool_ct[cyt][t]))

    def M(cyt, t, s):
        k = (t, s)
        return Msum[cyt][k] / Mcnt[cyt][k] if Mcnt[cyt][k] else float("nan")

    print(f"=== §34 INSTANT PEEK — epoch {epoch}, {args.n_per_cyt} tubes/cytokine ===")
    g0 = float(np.mean(offdiag_all))
    print(f"G0 off-diagonal (cross-cell-type) attention fraction: {g0:.3f} "
          f"({'cells INTERACT' if g0 > 0.2 else 'weak/none'})")
    print("pooling attention-primary:", {c: primary.get(c) for c in KEY if c in primary})
    print("\nInteraction direction  D = M^A[T_B,T_A] - M^B[T_A,T_B]  (>0 => A->B):")
    for A_, B_ in CASCADES + CONTROL:
        if A_ not in primary or B_ not in primary:
            print(f"  {A_}->{B_}: (missing)"); continue
        TA, TB = primary[A_], primary[B_]
        D = M(A_, TB, TA) - M(B_, TA, TB)
        kind = "cascade" if (A_, B_) in CASCADES else "control"
        print(f"  {A_}->{B_} [{kind}]  T_A={TA} T_B={TB}  D={D:+.4f}  "
              f"=> {'A->B' if D > 0 else 'B->A' if D < 0 else 'none'}")


if __name__ == "__main__":
    main()
