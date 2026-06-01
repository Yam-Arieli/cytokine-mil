"""
Fast, low-memory pre-check that the SCP per-cell metadata joins to the GEO MTX
lanes by (channel, 16bp barcode). Loads ONLY barcode lists (not matrices), so it
is safe to run on the gateway node. Exits non-zero if the match rate is below
threshold for any checked channel (i.e., the channel<->lane mapping is wrong).

The risky assumption it guards: SCP cell "<bc>-NN"  <->  GEO lane
cytokine-samplesNN, barcode "<bc>-1".  If the numbering doesn't line up, match
rate is ~0% and we must NOT launch the (expensive) GPU DAG.

Usage:
    python scripts/check_id_barcode_join.py --channels 01,02,03
    python scripts/check_id_barcode_join.py --channels 01 --min_frac 0.5
"""
from __future__ import annotations

import argparse
import gzip
import glob
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = str(REPO_ROOT / "reports" / "immune_dictionary" / "scp_metadata.parquet")
DEFAULT_EXTRACTED = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary/raw/extracted"


def _count_matches_streaming(extracted: str, channel: str, want: set):
    """Stream the GEO barcodes.tsv.gz for `channel` and count how many of the
    (small) SCP `want` barcodes appear. Streaming avoids materialising the full
    6.79M raw-whitelist barcode set (which OOMs the ~4 GB gateway node).
    Returns (n_matched, geo_total) or (None, None) if the lane file is missing.
    """
    hits = glob.glob(f"{extracted}/*cytokine-samples{channel}-barcodes.tsv.gz")
    if len(hits) != 1:
        return None, None
    matched = 0
    total = 0
    with gzip.open(hits[0], "rt") as fh:
        for ln in fh:
            total += 1
            if ln.strip().rsplit("-", 1)[0] in want:
                matched += 1
    return matched, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--extracted", default=DEFAULT_EXTRACTED)
    ap.add_argument("--channels", default="01,02,03")
    ap.add_argument("--min_frac", type=float, default=0.5)
    args = ap.parse_args()

    meta = pd.read_parquet(args.meta)
    channels = args.channels.split(",")
    print(f"Barcode-join check: meta={args.meta}  extracted={args.extracted}")
    ok = True
    for ch in channels:
        want = set(meta[meta["channel"] == ch]["barcode16"])
        matched, geo_total = _count_matches_streaming(args.extracted, ch, want)
        if matched is None:
            print(f"  channel {ch}: GEO barcodes file not found / ambiguous -> FAIL")
            ok = False
            continue
        frac = matched / max(1, len(want))
        flag = "OK" if frac >= args.min_frac else "LOW"
        print(f"  channel {ch}: SCP={len(want)} GEO={geo_total} "
              f"matched={matched} ({100*frac:.1f}%)  [{flag}]")
        if frac < args.min_frac:
            ok = False

    if not ok:
        print("JOIN CHECK FAILED: channel<->lane mapping looks wrong "
              "(low/zero barcode match). NOT safe to launch the DAG.",
              file=sys.stderr)
        sys.exit(1)
    print("JOIN CHECK PASSED: SCP labels join cleanly to GEO matrices.")


if __name__ == "__main__":
    main()
