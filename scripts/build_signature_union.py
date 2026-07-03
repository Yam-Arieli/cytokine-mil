"""
Build the signature-gene union U = ∪_X S_X for the confusion-direction experiment.

Reads the merged binary-IG signatures (long format: cytokine, gene, ig, rank_ig from
scripts/run_binary_ig_probe.py / merge_binary_ig_parquets.py), keeps the top-N genes
per cytokine (rank_ig <= top_n), unions them, and intersects with the model's HVG list
(S_X ⊆ HVG). Writes a JSON gene list to pass as train_oesinghaus_full.py --hvg_path.

Usage:
    python scripts/build_signature_union.py \
        --binary_ig results/group_u/ig_merge/binary_ig.parquet \
        --top_n 50 --out datasets/signature_union/gene_list_signature_union.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--binary_ig", required=True, help="merged binary_ig.parquet (cytokine,gene,rank_ig)")
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--hvg_path", default=HVG_PATH, help="JSON HVG list to intersect with")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = _parse_args()
    df = pd.read_parquet(args.binary_ig)
    if "rank_ig" not in df.columns or "gene" not in df.columns or "cytokine" not in df.columns:
        sys.exit(f"binary_ig must have columns cytokine,gene,rank_ig — got {list(df.columns)}")

    top = df[df["rank_ig"] <= args.top_n]
    per_cyt = top.groupby("cytokine")["gene"].nunique()
    union = sorted(set(top["gene"].astype(str)))
    n_cyt = top["cytokine"].nunique()

    with open(args.hvg_path) as f:
        hvg = set(json.load(f))
    in_hvg = [g for g in union if g in hvg]
    dropped = [g for g in union if g not in hvg]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(in_hvg))

    print(f"cytokines with signatures: {n_cyt} (top_n={args.top_n})")
    print(f"  genes/cytokine: min={per_cyt.min()} median={int(per_cyt.median())} max={per_cyt.max()}")
    print(f"union |U| = {len(union)} genes; in HVG: {len(in_hvg)}; dropped (not in HVG): {len(dropped)}")
    if dropped[:10]:
        print(f"  sample dropped: {dropped[:10]}")
    print(f"Saved {len(in_hvg)} genes -> {out}")


if __name__ == "__main__":
    main()
