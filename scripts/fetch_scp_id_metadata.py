"""
Fetch per-cell metadata for the Immune Dictionary (Cui 2024) from the PUBLIC
Single Cell Portal REST API (SCP2554) — no authentication required.

This is the demux key the GEO deposit lacks: the GEO RAW.tar has raw 10x gene-
expression matrices per hashed lane but NO per-cell cytokine/cell-type labels
(the MULTI-seq sample tags were processed separately). The SCP-rendered public
study exposes per-cell annotations through the cluster endpoint, which we fetch
here and join (by index, order-verified) into one table.

Output columns:
    cell_name   e.g. "AAACCCAAGACAGTCG-01"   (SCP barcode + channel suffix)
    cyt         machine cytokine name        ("IL1b", "IFNg", "TNFa", "PBS", ...)
    celltype    expert annotation            ("T_cell_CD4", "NK_cell", ...)
    rep         biological replicate         ("rep01".."rep14")
    channel     2-digit channel/lane         ("01".."45")  -> GEO lane samplesNN
    barcode16   16bp 10x barcode             join key to GEO barcodes (strip "-1")

The committed copy at data/immune_dictionary_scp_metadata.parquet is the
authoritative artifact used by build_pseudotubes_immune_dictionary.py; this
script regenerates it from the live API for provenance / refresh.

Usage:
    python scripts/fetch_scp_id_metadata.py            # -> data/immune_dictionary_scp_metadata.parquet
    python scripts/fetch_scp_id_metadata.py --out /path/to.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "reports" / "immune_dictionary" / "scp_metadata.parquet"

API = ("https://singlecell.broadinstitute.org/single_cell/api/v1/studies/"
       "SCP2554/clusters/tsne_all")
N_EXPECTED = 386703


def _fetch_annotation(name: str, want_cells: bool = False, timeout: int = 180):
    """Fetch one study-scope group annotation in cluster (cell) order."""
    fields = "annotation,cells" if want_cells else "annotation"
    url = (f"{API}?annotation_name={name}&annotation_scope=study"
           f"&annotation_type=group&subsample=all&fields={fields}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.load(resp)
    data = payload["data"]
    ann = data["annotations"]
    cells = data.get("cells")
    return ann, cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    print("Fetching SCP2554 per-cell annotations (public API, no auth)...")
    cyt, cells = _fetch_annotation("cyt", want_cells=True)
    print(f"  cyt:      {len(cyt)} values; cells: {len(cells) if cells else 0}")
    celltype, _ = _fetch_annotation("celltype")
    print(f"  celltype: {len(celltype)} values")
    rep, _ = _fetch_annotation("rep")
    print(f"  rep:      {len(rep)} values")

    # Order-consistency: all four pulls are the same cluster at subsample=all,
    # so they share the canonical cell order. Hard-assert lengths.
    n = len(cells)
    if not (len(cyt) == len(celltype) == len(rep) == n == N_EXPECTED):
        print(f"FATAL: length mismatch / unexpected n "
              f"(cyt={len(cyt)}, celltype={len(celltype)}, rep={len(rep)}, "
              f"cells={n}, expected={N_EXPECTED})", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame({"cell_name": cells, "cyt": cyt,
                       "celltype": celltype, "rep": rep})
    df["channel"] = df["cell_name"].str.rsplit("-", n=1).str[1]
    df["barcode16"] = df["cell_name"].str.rsplit("-", n=1).str[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Wrote {out}  ({len(df)} cells, {df['channel'].nunique()} channels, "
          f"{df['cyt'].nunique()} cytokines incl PBS)")


if __name__ == "__main__":
    main()
