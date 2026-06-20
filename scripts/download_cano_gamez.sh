#!/usr/bin/env bash
# Download the Cano-Gamez et al. 2020 CD4+ T-cell cytokine scRNA-seq (effectorness)
# processed data to the cluster. Free, no auth (BioStudies S-BSST2978; the EGA raw
# EGAS00001003215 is access-controlled, but the processed UMI counts + metadata are
# public here). ~8 GB zip.
#
# Conditions: resting / Th0 / Th1 / Th2 / Th17 / iTreg / IFN-beta; naive + memory CD4 T.
# Known cytokine cascades: IL-12->STAT4->IFN-g (Th1), IL-4->STAT6 (Th2),
# TGF-b->SMAD (iTreg), IFN-b->ISG. (See reports/cano_gamez/.)
#
# Usage (from the cluster, lab workdir):
#   bash cytokine-mil/scripts/download_cano_gamez.sh
#
# Output:
#   /cs/labs/mornitzan/yam.arieli/datasets/CanoGamez/raw/
#     scRNAseq.zip                 (8.03 GB archive)
#     extracted/                   (un-zipped UMI counts + metadata + features)
#     extracted/.extracted_ok      (idempotency marker)
#     download_manifest.txt        (post-extraction file tree + metadata peek)
set -eu

DEST=/cs/labs/mornitzan/yam.arieli/datasets/CanoGamez/raw
URL=https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BSST/978/S-BSST2978/Files/scRNAseq.zip

mkdir -p "$DEST/extracted"
cd "$DEST"

echo "[$(date -Iseconds)] Downloading Cano-Gamez scRNAseq.zip to $DEST"
wget -c "$URL"

echo "[$(date -Iseconds)] Download complete; extracting (idempotent)..."
if [ -f extracted/.extracted_ok ]; then
    echo "[$(date -Iseconds)] extracted/.extracted_ok present; skipping unzip"
else
    unzip -o scRNAseq.zip -d extracted
    touch extracted/.extracted_ok
fi

echo "[$(date -Iseconds)] Writing manifest + metadata peek..."
{
    echo "# Cano-Gamez (S-BSST2978) scRNAseq download manifest"
    echo "# Generated $(date -Iseconds)"
    echo ""
    echo "## File tree (extracted/)"
    find extracted -maxdepth 3 -type f -exec ls -la {} \; 2>/dev/null || true
    echo ""
    echo "## Disk usage"
    du -sh "$DEST" 2>/dev/null || true
} > "$DEST/download_manifest.txt" 2>&1 || true

echo "===================== EXTRACTED FILE TREE ====================="
find extracted -maxdepth 3 -type f -exec ls -la {} \; 2>/dev/null || true

# Peek at small text files (metadata / features) so we can design the adapter.
echo "===================== TEXT-FILE PEEKS ====================="
for f in $(find extracted -maxdepth 3 -type f \( -name "*.txt" -o -name "*.tsv" -o -name "*.csv" -o -name "*.tsv.gz" -o -name "*.csv.gz" -o -name "*.txt.gz" \) 2>/dev/null); do
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    echo "----- $f  (${sz} bytes) -----"
    if echo "$f" | grep -q "\.gz$"; then
        zcat "$f" 2>/dev/null | head -4 || true
        echo "... ncols(first row):"
        zcat "$f" 2>/dev/null | head -1 | awk -F'[\t,]' '{print NF}' || true
    else
        head -4 "$f" 2>/dev/null || true
        echo "... ncols(first row):"
        head -1 "$f" 2>/dev/null | awk -F'[\t,]' '{print NF}' || true
    fi
done

echo "[$(date -Iseconds)] DONE."
