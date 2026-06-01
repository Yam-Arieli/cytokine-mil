#!/usr/bin/env bash
# Download the Cui et al. 2024 mouse Immune Dictionary (GSE202186) to the
# cluster. Free, no auth. ~7.6 GB tar plus 2 small metadata files.
#
# Usage (from the cluster, lab workdir):
#   bash cytokine-mil/scripts/download_immune_dictionary.sh
#
# Output:
#   /cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary/raw/
#     GSE202186_RAW.tar                                      (7.6 GB, archive)
#     GSE202186_map-scRNAseq-cytokines-dictionary.xlsx       (cytokine ↔ GSM map)
#     GSE202186_family.soft.gz                               (sample-level metadata)
#     extracted/ (the un-tarred MTX/TSV per-GSM count matrices)
#     download_manifest.txt (post-extraction ls output)
# NOTE: deliberately NOT using `set -o pipefail`. The manifest-writing block
# below uses `ls | head`, which sends SIGPIPE (signal 13) to `ls` when head
# closes the pipe early; under pipefail+`set -e` that aborted the whole script
# AFTER a successful extraction (job 30716600 failed here, exit 13). `set -eu`
# (no pipefail) plus a `|| true` guard on the cosmetic manifest block is robust.
set -eu

DEST=/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary/raw
BASE_URL=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202186

mkdir -p "$DEST/extracted"
cd "$DEST"

echo "[$(date -Iseconds)] Starting Immune Dictionary download to $DEST"

# Resume-friendly downloads
wget -c "$BASE_URL/suppl/GSE202186_RAW.tar"
wget -c "$BASE_URL/suppl/GSE202186_map-scRNAseq-cytokines-dictionary.xlsx"
wget -c "$BASE_URL/soft/GSE202186_family.soft.gz"

echo "[$(date -Iseconds)] Downloads complete; extracting RAW tar..."
# Idempotent: skip re-extraction if a prior run already completed it. tar stops
# at the end-of-archive marker, so trailing bytes from an over-resumed download
# (the .tar grew past the server size in an earlier session) are ignored.
if [ -f extracted/.extracted_ok ]; then
    echo "[$(date -Iseconds)] extracted/.extracted_ok present; skipping tar -xf"
else
    tar -xf GSE202186_RAW.tar -C extracted
    touch extracted/.extracted_ok
fi

echo "[$(date -Iseconds)] Extraction complete; writing manifest..."
# Cosmetic manifest only — guarded with `|| true` so a SIGPIPE from `ls | head`
# (or any other pipeline quirk) can never fail the build job. The extracted
# data is already on disk at this point; this block is informational.
{
    echo "# Immune Dictionary (GSE202186) raw download manifest"
    echo "# Generated $(date -Iseconds)"
    echo ""
    echo "## Top-level files"
    ls -la "$DEST" | grep -v "^total" | grep -v "^d" || true
    echo ""
    echo "## Extracted entries (first 20)"
    ls "$DEST/extracted" | head -20 || true
    echo ""
    echo "## Total extracted entry count"
    ls "$DEST/extracted" | wc -l || true
    echo ""
    echo "## Disk usage"
    du -sh "$DEST" || true
} > "$DEST/download_manifest.txt" 2>&1 || true

echo "[$(date -Iseconds)] DONE. Manifest written to $DEST/download_manifest.txt"
