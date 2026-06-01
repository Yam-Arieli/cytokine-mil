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
set -euo pipefail

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
tar -xf GSE202186_RAW.tar -C extracted

echo "[$(date -Iseconds)] Extraction complete; writing manifest..."
{
    echo "# Immune Dictionary (GSE202186) raw download manifest"
    echo "# Generated $(date -Iseconds)"
    echo ""
    echo "## Top-level files"
    ls -la "$DEST" | grep -v "^total" | grep -v "^d"
    echo ""
    echo "## Extracted per-GSM files (first 10)"
    ls "$DEST/extracted" | head -20
    echo ""
    echo "## Total extracted file count"
    ls "$DEST/extracted" | wc -l
    echo ""
    echo "## Disk usage"
    du -sh "$DEST"
} > "$DEST/download_manifest.txt"

echo "[$(date -Iseconds)] DONE. Manifest written to $DEST/download_manifest.txt"
