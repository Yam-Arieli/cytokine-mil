#!/usr/bin/env bash
# Download the Stephenson et al. 2021 COVID-19 PBMC atlas (Haniffa lab;
# Ncl-Cambridge-UCL) to the cluster. Free, no auth. ~7.2 GB single .h5ad.
#
# Citation: Stephenson E. et al., "Single-cell multi-omics analysis of the
#   immune response in COVID-19", Nature Medicine 27, 904-916 (2021).
# Source: COVID-19 Cell Atlas (Wellcome Sanger), public COG bucket.
#
# ~647K PBMCs, ~130 donors spanning Healthy / Asymptomatic / Mild / Moderate /
# Severe / Critical, with author cell-type annotations. This is the
# disease-progression substrate for the §30 cascade-direction extension: an
# ordered clinical-severity axis serves as the direction oracle (less-severe is
# "upstream" of more-severe).
#
# Usage (from the cluster, lab workdir):
#   bash cytokine-mil/scripts/download_covid_haniffa.sh
#
# Output:
#   /cs/labs/mornitzan/yam.arieli/datasets/COVID_Haniffa/raw/
#     haniffa21.processed.h5ad        (~7.2 GB, processed log-norm + annotations)
#     download_manifest.txt
set -eu

DEST=/cs/labs/mornitzan/yam.arieli/datasets/COVID_Haniffa/raw
URL=https://covid19.cog.sanger.ac.uk/submissions/release1/haniffa21.processed.h5ad

mkdir -p "$DEST"
cd "$DEST"

echo "[$(date -Iseconds)] Starting COVID-Haniffa download to $DEST"

# Resume-friendly download (-c). The COG bucket serves a stable processed h5ad.
wget -c "$URL"

echo "[$(date -Iseconds)] Download complete; writing manifest..."
{
    echo "# COVID-Haniffa (Stephenson 2021) raw download manifest"
    echo "# Generated $(date -Iseconds)"
    echo ""
    echo "## Files"
    ls -la "$DEST" | grep -v "^total" || true
    echo ""
    echo "## Disk usage"
    du -sh "$DEST" || true
} > "$DEST/download_manifest.txt" 2>&1 || true

echo "[$(date -Iseconds)] DONE. Manifest written to $DEST/download_manifest.txt"
