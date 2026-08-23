#!/usr/bin/env python
"""Stage 3 of the Oesinghaus full-90 DAG — merge chunk signatures, verify provenance.

Concatenates every `signatures_chunk_*.parquet` into `signatures_all90.parquet` and fails
loudly unless the chunks are one coherent fit:

  * every chunk records the SAME encoder sha256 and the SAME tube-shards sha256
    (CLAUDE.md §27.6 — signatures from different encoders/tubes are not comparable, and
    `cross_asym` compares them across cytokines);
  * every expected stimulus is present exactly once;
  * every signature has exactly `top_n` genes.

Usage: python scripts/merge_full90_signatures.py --output_dir results/oes_full90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _full90_config as C  # noqa: E402
from cytokine_mil.analysis.full90_tube_io import read_meta  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="results/oes_full90")
    args = ap.parse_args()

    import pandas as pd

    out = Path(args.output_dir)
    metas = sorted(out.glob("chunk_*_meta.json"))
    parquets = sorted(out.glob("signatures_chunk_*.parquet"))
    if not metas or len(metas) != len(parquets):
        raise SystemExit(
            f"FATAL: {len(metas)} chunk meta files vs {len(parquets)} parquets in {out}"
        )

    loaded = [json.loads(p.read_text()) for p in metas]
    enc_shas = {m["encoder_sha256"] for m in loaded}
    tube_shas = {m["tubes_shards_sha256"] for m in loaded}
    if len(enc_shas) != 1:
        raise SystemExit(
            f"FATAL: chunks used {len(enc_shas)} different encoders: {sorted(enc_shas)}. "
            "This is exactly the CLAUDE.md §27.6 failure — abort rather than emit "
            "non-comparable signatures."
        )
    if len(tube_shas) != 1:
        raise SystemExit(f"FATAL: chunks used {len(tube_shas)} different tube sets.")

    tube_meta = read_meta(out / "tubes")
    expected = sorted(c for c in tube_meta["conditions"] if c != C.CONTROL)

    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    got = sorted(df["cytokine"].unique().tolist())
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    if missing or extra:
        raise SystemExit(
            f"FATAL: signature set does not match the tubes.\n  missing: {missing}\n"
            f"  unexpected: {extra}"
        )

    sizes = df.groupby("cytokine").size()
    bad = sizes[sizes != C.TOP_N]
    if len(bad):
        raise SystemExit(f"FATAL: {len(bad)} signatures do not have {C.TOP_N} genes:\n{bad}")
    if df.duplicated(["cytokine", "gene"]).any():
        raise SystemExit("FATAL: duplicate (cytokine, gene) rows in the merged signatures")

    merged = out / "signatures_all90.parquet"
    df.sort_values(["cytokine", "rank_ig"]).to_parquet(merged, index=False)

    C.write_json(out / "signatures_meta.json", {
        "n_conditions": len(got),
        "top_n": C.TOP_N,
        "n_rows": int(len(df)),
        "encoder_sha256": next(iter(enc_shas)),
        "tubes_shards_sha256": next(iter(tube_shas)),
        "chunks": [m["chunk_id"] for m in loaded],
        "n_unique_genes": int(df["gene"].nunique()),
    })
    print(f"[done] {merged}: {len(got)} signatures x {C.TOP_N} genes "
          f"({df['gene'].nunique()} distinct genes)", flush=True)
    print(f"[provenance] encoder={next(iter(enc_shas))[:16]}...  "
          f"tubes={next(iter(tube_shas))[:16]}...", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
