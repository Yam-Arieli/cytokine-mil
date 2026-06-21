"""
Signature-pruning test (follow-up to §31): does a stability-pruned S_X beat the raw
top-50 on the §26 direction benchmark?

Uses the recurrent-IG trajectory already produced by run_recurrent_ig_oesinghaus.py.
For each labeled directional pair it recomputes cross_asym (cascadir
directional_asymmetry_test) under three per-cytokine signature definitions, per seed,
then aggregates across seeds and scores the sign against the audited expected_sign:

  * raw_top50   — final-epoch top-50 by IG (baseline; the §31 analysis value)
  * anchor_only — final top-50 kept only if category == "Anchor" (early & stable)
  * stable_half — final top-50 kept if stab >= that cytokine's median stab (more-stable half)

ALL three use the SAME cross_asym aggregation, so the comparison is internally fair.

Usage:
    python scripts/run_signature_pruning_test.py \
        --ig_dir results/recurrent_ig --output_dir results/recurrent_ig/pruning_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))

from cytokine_mil.analysis.oesinghaus_cell_loader import load_oesinghaus_cells_by_pair  # noqa: E402
from cascadir.cross_asym import directional_asymmetry_test  # noqa: E402

VAL_DONORS = ["Donor2", "Donor3"]
MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
AUDITED_CSV = REPO_ROOT / "reports" / "cascade_pairs" / "cytokine_axes_audited.csv"


def variant_sets(rec_seed: pd.DataFrame, variant: str) -> dict:
    """{cytokine: set(genes)} for one seed under a signature definition."""
    out = {}
    for cyt, g in rec_seed.groupby("cytokine"):
        final = g[g["final_member"]]
        if variant == "raw_top50":
            genes = set(final["gene"])
        elif variant == "anchor_only":
            genes = set(g[g["category"] == "Anchor"]["gene"])
        elif variant == "stable_half":
            if len(final) == 0:
                genes = set()
            else:
                thr = final["stab"].median()
                genes = set(final[final["stab"] >= thr]["gene"])
        else:
            raise ValueError(variant)
        if genes:
            out[cyt] = genes
    return out


def pair_cross_asym(cells_by_pair, sig_idx, a, b):
    df = directional_asymmetry_test(cells_by_pair, sig_idx, a, b, control_label="PBS", min_cells=10)
    if df is None or len(df) == 0:
        return np.nan
    return float(np.median(df["cross_asym"].to_numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ig_dir", default="results/recurrent_ig")
    ap.add_argument("--output_dir", default="results/recurrent_ig/pruning_test")
    ap.add_argument("--manifest_path", default=MANIFEST_PATH)
    ap.add_argument("--hvg_path", default=HVG_PATH)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(Path(args.ig_dir) / "stats" / "recruitment_table.csv")
    seeds = sorted(rec["seed"].unique())

    aud = pd.read_csv(AUDITED_CSV)
    labeled = aud[aud.get("counts_in_benchmark", False) == True].copy()  # noqa: E712
    labeled = labeled[labeled["expected_sign"].isin([1, -1])]
    cyts = sorted(set(labeled["axis_a"]) | set(labeled["axis_b"]))

    print(f"Loading cells for {len(cyts)} labeled cytokines (train donors)...", flush=True)
    cells_by_pair, gene_names = load_oesinghaus_cells_by_pair(
        manifest_path=args.manifest_path, cytokines=cyts, hvg_path=args.hvg_path,
        pbs_label="PBS", exclude_donors=VAL_DONORS)
    gidx = {g: i for i, g in enumerate(gene_names)}
    labeled = labeled[labeled["axis_a"].isin(cells_keys(cells_by_pair)) &
                      labeled["axis_b"].isin(cells_keys(cells_by_pair))]

    variants = ["raw_top50", "anchor_only", "stable_half"]
    rows, sizes = [], {v: [] for v in variants}
    for v in variants:
        # per seed -> sig_idx, then per pair cross_asym; aggregate across seeds
        per_seed_idx = {}
        for s in seeds:
            sets = variant_sets(rec[rec["seed"] == s], v)
            per_seed_idx[s] = {c: np.array([gidx[g] for g in gs if g in gidx], dtype=np.int64)
                               for c, gs in sets.items()}
            sizes[v] += [len(per_seed_idx[s].get(c, [])) for c in cyts if c in per_seed_idx[s]]
        for _, L in labeled.iterrows():
            a, b, exp = L["axis_a"], L["axis_b"], int(L["expected_sign"])
            xs = []
            for s in seeds:
                sidx = per_seed_idx[s]
                if a in sidx and b in sidx and len(sidx[a]) and len(sidx[b]):
                    xs.append(pair_cross_asym(cells_by_pair, sidx, a, b))
            xs = [x for x in xs if np.isfinite(x)]
            ca = float(np.mean(xs)) if xs else np.nan
            rows.append({"variant": v, "axis_a": a, "axis_b": b, "expected_sign": exp,
                         "cross_asym": ca,
                         "correct": (np.sign(ca) == exp) if np.isfinite(ca) else np.nan})

    res = pd.DataFrame(rows)
    res.to_csv(out / "pruning_pairs.csv", index=False)

    lines = ["# Signature-pruning test — does a stability-pruned S_X beat raw top-50?\n",
             "cross_asym direction accuracy on the labeled non-AMBIGUOUS pairs "
             "(same aggregation for all variants):\n",
             "| variant | accuracy | n scored | mean set size |",
             "|---|---|---|---|"]
    summary = []
    for v in variants:
        sub = res[res["variant"] == v].dropna(subset=["cross_asym"])
        acc = float(sub["correct"].mean()) if len(sub) else float("nan")
        msz = float(np.mean(sizes[v])) if sizes[v] else float("nan")
        lines.append(f"| {v} | {acc:.3f} | {len(sub)} | {msz:.0f} |")
        summary.append((v, acc, len(sub), msz))
    base = next((a for vv, a, _, _ in summary if vv == "raw_top50"), float("nan"))
    best = max((a for _, a, _, _ in summary if np.isfinite(a)), default=float("nan"))
    lines.append("")
    lines.append(f"**Baseline (raw_top50) = {base:.3f}; best pruned = {best:.3f} "
                 f"({'pruning HELPS' if best > base + 1e-9 else 'pruning does NOT help'}).**")
    (out / "pruning_summary.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def cells_keys(cbp):
    return {k[0] for k in cbp}


if __name__ == "__main__":
    main()
