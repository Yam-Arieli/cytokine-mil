#!/usr/bin/env bash
# Download the whole-PBMC CITE-seq object from the multimodal SARS-CoV-2
# vaccination + infection atlas (Nat Immunol 2023) to the cluster.
# Free, no auth (Zenodo). ~1.6 GB single .rds (R Seurat object).
#
# Citation: "Multimodal single-cell datasets characterize antigen-specific CD8+
#   T cells across SARS-CoV-2 vaccination and infection", Nature Immunology 2023.
# Source: Zenodo record 7555405 (processed data; raw is dbGaP phs003322, gated —
#   we never need raw). PBMC_vaccine_CITE.rds = 3' RNA + 173 TotalSeq-A surface
#   proteins, whole PBMC, day 0 / 2 / 10 / 28, ~6 donors.
#
# This is the T-cell maturation-cascade substrate for the §32 cascade-direction
# extension: the naive -> effector -> memory differentiation order is the
# direction oracle (naive is "upstream"). NOTE: Zenodo ships a Seurat .rds, so a
# .rds -> AnnData conversion (convert_vaccine_rds_to_h5ad.R + assemble_vaccine_h5ad.py)
# runs next; cascadir needs AnnData.
#
# Usage (from the cluster, lab workdir):
#   bash cytokine-mil/scripts/download_vaccine_multimodal.sh
#
# Output:
#   /cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/raw/
#     PBMC_vaccine_CITE.rds        (~1.6 GB)
#     download_manifest.txt
set -eu

DEST=/cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/raw
URL="https://zenodo.org/records/7555405/files/PBMC_vaccine_CITE.rds?download=1"
OUT="PBMC_vaccine_CITE.rds"

mkdir -p "$DEST"
cd "$DEST"

echo "[$(date -Iseconds)] Starting SARS-CoV-2 vaccine CITE download to $DEST"

# Resume-friendly download (-c). Zenodo serves a stable processed .rds.
# -O fixes the filename (the URL has a ?download=1 query that wget would keep).
wget -c -O "$OUT" "$URL"

echo "[$(date -Iseconds)] Download complete; writing manifest..."
{
    echo "# SARS-CoV-2 vaccine CITE (Nat Immunol 2023) raw download manifest"
    echo "# Generated $(date -Iseconds)"
    echo "# Source: Zenodo 7555405 / PBMC_vaccine_CITE.rds"
    echo ""
    echo "## Files"
    ls -la "$DEST" | grep -v "^total" || true
    echo ""
    echo "## Disk usage"
    du -sh "$DEST" || true
} > "$DEST/download_manifest.txt" 2>&1 || true

echo "[$(date -Iseconds)] DONE. Manifest written to $DEST/download_manifest.txt"
