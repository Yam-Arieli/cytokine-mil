#!/usr/bin/env bash
# Download the Oelen et al. 2022 1M-scBloodNL human PBMC dataset (free, no auth) from
# the eQTLGen/molgenis portal. ~120 donors, PBMC stimulated with C. albicans (CA),
# P. aeruginosa (PA), M. tuberculosis (MTB) at 3h/24h + untreated (UT), ~1.3M cells.
# NOTE: the molgenis TLS cert is expired -> wget --no-check-certificate.
# Only the processed *expression matrices + metadata* are public here; raw reads +
# genotypes are EGA-gated (EGAS00001005376).
#
# Knob mapping (for the adapter):
#   donor      = assignment  (the genotype-demuxed individual; ~120)  <- 120-donor gate!
#   condition  = pathogen parsed from `timepoint` (3hCA/24hPA/...); UT -> PBS
#   timepoint  = 3h / 24h parsed from `timepoint`
#   cell_type  = cell_type_lowerres (CD4T/CD8T/monocyte/NK/B/DC/...)
#   chem       = V2 / V3 (batch covariate)
#
# Usage (cluster, lab workdir):  bash cytokine-mil/scripts/download_oelen2022.sh
set -eu

DEST=/cs/labs/mornitzan/yam.arieli/datasets/Oelen2022/raw
BASE=https://molgenis26.gcc.rug.nl/downloads/1m-scbloodnl
mkdir -p "$DEST"
cd "$DEST"

echo "[$(date -Iseconds)] Downloading Oelen 1M-scBloodNL to $DEST"
for f in \
    10x_v2_RNA_matrix.mtx.gz 10x_v2_RNA_features.tsv.gz 10x_v2_barcodes.tsv.gz \
    10x_v3_RNA_matrix.mtx.gz 10x_v3_RNA_features.tsv.gz 10x_v3_barcodes.tsv.gz \
    1M_assignments_conditions_expid.tsv 1M_cell_types.tsv ; do
    echo "[$(date -Iseconds)] wget $f"
    wget --no-check-certificate -c "$BASE/$f"
done

echo "[$(date -Iseconds)] Done downloading. Sizes:"
ls -la "$DEST"

echo "===================== METADATA PEEKS ====================="
echo "----- timepoint value_counts (condition x time) -----"
cut -f3 1M_assignments_conditions_expid.tsv | tail -n +2 | sort | uniq -c | sort -rn
echo "----- chem value_counts -----"
cut -f4 1M_assignments_conditions_expid.tsv | tail -n +2 | sort | uniq -c
echo "----- n distinct donors (assignment) -----"
cut -f2 1M_assignments_conditions_expid.tsv | tail -n +2 | sort -u | wc -l
echo "----- cell_type_lowerres value_counts -----"
cut -f3 1M_cell_types.tsv | tail -n +2 | sort | uniq -c | sort -rn
echo "[$(date -Iseconds)] DONE."
