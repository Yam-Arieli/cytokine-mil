#!/bin/bash
#SBATCH --job-name=gu_pipe
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_pipe_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_pipe_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short

# Group-U (§27): run the full Path A -> Bridge -> Path B pipeline over ALL 121
# Path A axes with the §27.2 DIRECTION-permutation null (n=1000). Two donor
# settings, mirroring full19:
#   pooled     (all 12 donors)  -> §26 regression anchor (the published 15/17)
#   train_only (D2/D3 excluded) -> pre-registered PRIMARY (matches the binary
#                                  models' training split; GROUP_U_PREREGISTRATION.md)

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

AXES=reports/cascade_pairs/cytokine_axes.csv
IG=results/group_u/binary_ig_all45/binary_ig.parquet
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json
mkdir -p results/group_u/pipeline_full121_pooled results/group_u/pipeline_full121_train_only

echo "=== Run 1/2: pooled (all donors) — §26 regression anchor ==="
$PY scripts/run_pipeline_a_bridge_b.py \
    --axes_csv          "$AXES" \
    --binary_ig_parquet "$IG" \
    --manifest_path     "$MAN" \
    --hvg_path          "$HVG" \
    --output_dir        results/group_u/pipeline_full121_pooled \
    --top_n 50 --min_cells 10 \
    --n_null_perms 100 \
    --n_direction_perms 1000 --direction_null_seed 123

echo ""
echo "=== Run 2/2: train donors only (D2/D3 excluded) — pre-registered primary ==="
$PY scripts/run_pipeline_a_bridge_b.py \
    --axes_csv          "$AXES" \
    --binary_ig_parquet "$IG" \
    --manifest_path     "$MAN" \
    --hvg_path          "$HVG" \
    --output_dir        results/group_u/pipeline_full121_train_only \
    --top_n 50 --min_cells 10 \
    --n_null_perms 100 \
    --n_direction_perms 1000 --direction_null_seed 123 \
    --exclude_donors    Donor2 Donor3

echo ""
echo "DONE. per_axis_summary.csv in both pipeline_full121_* dirs."
