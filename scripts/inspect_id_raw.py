"""
One-shot inspector for the extracted Immune Dictionary (GSE202186) raw data.

Prints exactly what the build_pseudotubes_immune_dictionary.py adapter needs to
verify its 4 TODOs against reality:
  1. The extracted/ directory layout (per-GSM MTX vs single combined files).
  2. The xlsx mapping file's column names + first rows (GSM_id / cytokine /
     mouse_id?  Or something else?).
  3. The SOFT family file's per-sample characteristics keys.
  4. The actual control-condition label string in the cytokine column (so
     PBS_CONTROL_STRINGS can be confirmed).

Run on the cluster:
    /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/inspect_id_raw.py
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

RAW = Path("/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary/raw")
EXTRACTED = RAW / "extracted"
XLSX = RAW / "GSE202186_map-scRNAseq-cytokines-dictionary.xlsx"
SOFT = RAW / "GSE202186_family.soft.gz"


def sec(title: str) -> None:
    print("\n" + "#" * 4 + " " + title)


def inspect_extracted() -> None:
    sec("EXTRACTED LAYOUT")
    if not EXTRACTED.exists():
        print(f"  MISSING: {EXTRACTED}")
        return
    entries = sorted(EXTRACTED.iterdir())
    print(f"  total entries: {len(entries)}")
    for e in entries[:30]:
        kind = "dir" if e.is_dir() else f"{e.stat().st_size} bytes"
        print(f"    {e.name}  ({kind})")
    if len(entries) > 30:
        print(f"    ... and {len(entries) - 30} more")
    # If the first entry is a dir, show its contents (per-GSM 10x layout).
    dirs = [e for e in entries if e.is_dir()]
    if dirs:
        sec("FIRST GSM SUBDIR CONTENTS")
        sub = sorted(dirs[0].iterdir())
        print(f"  {dirs[0].name}/:")
        for s in sub[:15]:
            print(f"    {s.name}  ({s.stat().st_size} bytes)")
    # Look for any *.mtx / *.h5 / *.csv at the top level
    sec("FILE-TYPE TALLY (top level of extracted/)")
    suffixes: dict = {}
    for e in entries:
        suf = e.suffix or ("<dir>" if e.is_dir() else "<none>")
        suffixes[suf] = suffixes.get(suf, 0) + 1
    for suf, n in sorted(suffixes.items()):
        print(f"    {suf}: {n}")


def inspect_xlsx() -> None:
    sec("XLSX MAPPING FILE")
    if not XLSX.exists():
        print(f"  MISSING: {XLSX}")
        return
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not available")
        return
    # Try every sheet
    xl = pd.ExcelFile(XLSX)
    print(f"  sheets: {xl.sheet_names}")
    for sheet in xl.sheet_names:
        df = pd.read_excel(XLSX, sheet_name=sheet)
        print(f"\n  sheet={sheet!r}  shape={df.shape}")
        print(f"  columns: {list(df.columns)}")
        # show first rows compactly
        with pd.option_context("display.max_columns", None,
                               "display.width", 200):
            print(df.head(8).to_string())
        # If a column looks like a cytokine/condition column, list unique values
        for col in df.columns:
            cl = str(col).lower()
            if any(k in cl for k in ("cytokine", "condition", "treat",
                                     "stim", "ligand", "sample_title", "title")):
                uniq = df[col].astype(str).unique()
                print(f"  unique values in {col!r} ({len(uniq)}): "
                      f"{sorted(uniq)[:40]}")


def inspect_soft() -> None:
    sec("SOFT FAMILY FILE (sample characteristics)")
    if not SOFT.exists():
        print(f"  MISSING: {SOFT}")
        return
    char_lines = []
    title_lines = []
    n_samples = 0
    with gzip.open(SOFT, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                n_samples += 1
            if line.startswith("!Sample_characteristics_ch1"):
                char_lines.append(line.strip())
            if line.startswith("!Sample_title"):
                title_lines.append(line.strip())
    print(f"  n Sample_geo_accession lines: {n_samples}")
    print(f"  first 12 Sample_title lines:")
    for t in title_lines[:12]:
        print(f"    {t}")
    print(f"  first 16 Sample_characteristics_ch1 lines:")
    for c in char_lines[:16]:
        print(f"    {c}")


def main() -> None:
    print("Immune Dictionary raw-data inspection")
    print(f"RAW = {RAW}")
    inspect_extracted()
    inspect_xlsx()
    inspect_soft()
    print("\nDONE inspection.")


if __name__ == "__main__":
    main()
