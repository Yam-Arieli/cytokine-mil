"""
Method 5 Analysis: Joint attention + geometry trajectory.

For each candidate cascade pair (A→B) and cell type T:
  At each epoch t:
    - Geo signal:  ASYM(A,B,t)                             from geo_trajectory.pkl
    - Attn signal: mean_attention(T in A-tubes, t)         from attention_trajectory.pkl

  Cross-correlation: Spearman rho between geo_traj and attn_traj(T) for all T
  Best-correlated cell type per pair = proposed relay cell type for that cascade.

  Concordance rule:
    - Both signals rise together → T is the relay: it gains attention AND the
      geometry displacement toward B increases simultaneously.
    - Only geo rises → geometry signal is not mediated by a specific attended cell
    - Only attn rises → cell type becomes attended for a different reason

Output: results/instrumented_analysis/method5_joint/report.txt + summary.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE    = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full")
OUT_DIR = BASE / "instrumented_analysis" / "method5_joint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_DIRS = {s: BASE / f"instrumented_seed{s}" for s in [11, 12, 13, 14, 15]}

PAIRS_OF_INTEREST = [
    ("IFN-beta",  "IL-27",    "top stable pair"),
    ("IL-27",     "IL-22",    "top stable pair"),
    ("IL-12",     "IFN-gamma","positive control"),
    ("IFN-gamma", "IL-12",    "reverse"),
    ("IL-6",      "IL-10",    "negative control"),
    ("IFN-beta",  "TNF-alpha","top stable pair"),
]


def find_cyt_key(d, name):
    exact = [k for k in d if k.lower() == name.lower()]
    if exact: return exact[0]
    partial = [k for k in d if name.lower() in k.lower()]
    return partial[0] if partial else None


def find_idx(cytokine_names, name):
    exact = [i for i, c in enumerate(cytokine_names) if c.lower() == name.lower()]
    if exact: return exact[0]
    partial = [i for i, c in enumerate(cytokine_names) if name.lower() in c.lower()]
    return partial[0] if partial else None


def main():
    lines = []
    W = 72

    # Load both trajectory types for each seed
    seed_geo  = {}
    seed_attn = {}
    for seed, sdir in SEED_DIRS.items():
        g = sdir / "geo_trajectory.pkl"
        a = sdir / "attention_trajectory.pkl"
        if not g.exists() or not a.exists():
            print(f"  Seed {seed}: missing pkl, skipping.")
            continue
        with open(g, "rb") as f:
            seed_geo[seed] = pickle.load(f)
        with open(a, "rb") as f:
            seed_attn[seed] = pickle.load(f)
        print(f"  Seed {seed}: loaded geo+attn")

    if not seed_geo:
        print("No data found. Exiting.")
        return

    lines += ["=" * W, "Method 5: Joint Attention + Geometry Trajectory Analysis", "=" * W, ""]
    lines += [f"Seeds: {sorted(seed_geo.keys())}", ""]
    lines += ["For each cascade pair (A→B) and cell type T:",
              "  rho = Spearman(ASYM(A,B,t), mean_attn(T in A-tubes, t))",
              "  High rho + both rising → T is the relay cell type.", ""]

    summary_pairs = []

    for src, tgt, label in PAIRS_OF_INTEREST:
        lines.append(f"{'─'*W}")
        lines.append(f"  {src} → {tgt}  [{label}]")

        # Collect cross-seed evidence
        best_relay_votes = {}   # cell_type → list of rho values (one per seed)

        for seed in sorted(seed_geo.keys()):
            geo  = seed_geo[seed]
            attn = seed_attn[seed]

            si = find_idx(geo["cytokine_names"], src)
            if si is None:
                continue
            ti = find_idx(geo["cytokine_names"], tgt)
            if ti is None:
                continue

            geo_traj = geo["asymmetry_traj"][:, si, ti]   # (n_epochs,)

            # Find the attention trajectory for src cytokine
            src_cyt_key = find_cyt_key(attn["trajectory"], src)
            if src_cyt_key is None:
                continue

            cell_type_rhos = {}
            for ct, attn_traj in attn["trajectory"][src_cyt_key].items():
                if len(attn_traj) != len(geo_traj):
                    continue
                r, _ = spearmanr(geo_traj, attn_traj)
                if not np.isnan(r):
                    cell_type_rhos[ct] = float(r)

            if not cell_type_rhos:
                continue

            top_ct = max(cell_type_rhos, key=cell_type_rhos.get)
            top_rho = cell_type_rhos[top_ct]
            top3 = sorted(cell_type_rhos, key=cell_type_rhos.get, reverse=True)[:3]

            lines.append(f"    Seed {seed}: best relay={top_ct} (rho={top_rho:.3f})  "
                         f"top3={top3}")

            for ct, rho in cell_type_rhos.items():
                best_relay_votes.setdefault(ct, []).append(rho)

        # Aggregate: mean rho across seeds per cell type
        if best_relay_votes:
            relay_scores = {
                ct: float(np.mean(rhos))
                for ct, rhos in best_relay_votes.items()
                if len(rhos) >= max(1, len(seed_geo) // 2)
            }
            if relay_scores:
                top_relay = max(relay_scores, key=relay_scores.get)
                top_rho_  = relay_scores[top_relay]
                top3_relay = sorted(relay_scores, key=relay_scores.get, reverse=True)[:3]

                lines.append(f"  → Cross-seed top relay: {top_relay}  "
                             f"(mean rho={top_rho_:.3f})")
                lines.append(f"  → Top-3 candidates: "
                             f"{[(c, round(relay_scores[c],3)) for c in top3_relay]}")

                summary_pairs.append({
                    "pair": f"{src} → {tgt}",
                    "label": label,
                    "top_relay": top_relay,
                    "top_relay_rho": top_rho_,
                    "top3_relays": [(c, relay_scores[c]) for c in top3_relay],
                })
            else:
                lines.append("  → Insufficient cross-seed data for relay identification.")
        else:
            lines.append("  → No attention-geo correlation data.")
        lines.append("")

    report_str = "\n".join(lines)
    print(report_str)
    with open(OUT_DIR / "report.txt", "w") as f:
        f.write(report_str + "\n")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"pairs": summary_pairs, "seeds": sorted(seed_geo.keys())}, f, indent=2)

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
