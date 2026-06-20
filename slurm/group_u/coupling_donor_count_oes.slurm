#!/bin/bash
#SBATCH --job-name=oes_coupldc
#SBATCH --array=0-3
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_coupldc_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_coupldc_%A_%a.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=short

# Donor-count test on Oesinghaus (reflective-crunching-yeti plan): is the
# vsPBS-vs-vsPanel COUPLING-gate preference driven by donor COUNT or the dataset?
#
# Holds dataset (Oes), method (cell-level degree coupling), signatures
# (binary_ig_all24, all donors) and benchmark (cytokine_axes_audited.csv,
# counts_in_benchmark) CONSTANT; varies ONLY the donors entering the coupling:
#   task 0 high_all10 : all 10 train donors (cell-level anchor; same path as LOW)
#   task 1 low_g1     : Donor1 Donor4 Donor5
#   task 2 low_g2     : Donor6 Donor7 Donor8
#   task 3 low_g3     : Donor9 Donor10 Donor11
# Each run computes BOTH variants (IG_vsPBS, IG_vsPanel) x {raw, hub}. First
# cell-level-degree run on Oesinghaus. CPU-only; reuses existing signatures.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

IG=results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json
AXES=reports/cascade_pairs/cytokine_axes_audited.csv

case $SLURM_ARRAY_TASK_ID in
  0) TAG=high_all10; DARGS="--exclude_donors Donor2 Donor3" ;;
  1) TAG=low_g1;     DARGS="--include_donors Donor1 Donor4 Donor5" ;;
  2) TAG=low_g2;     DARGS="--include_donors Donor6 Donor7 Donor8" ;;
  3) TAG=low_g3;     DARGS="--include_donors Donor9 Donor10 Donor11" ;;
  *) echo "bad array idx $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

OUT=results/coupling_donor_count_oes/$TAG
mkdir -p "$OUT"
echo "=== Oes donor-count coupling: task=$SLURM_ARRAY_TASK_ID tag=$TAG donors=$DARGS ==="

$PY scripts/run_cell_degree_coupling.py \
    --dataset oesinghaus \
    --binary_ig_parquet "$IG" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --axes_csv "$AXES" \
    $DARGS \
    --top_n 50 --min_cells 10 --n_perm 500 \
    --output_dir "$OUT"

echo "DONE $TAG. report at $OUT/cell_degree_report.md"
