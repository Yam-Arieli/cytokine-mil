#!/bin/bash
#SBATCH --job-name=oes_dir76
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_direction/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_direction/slurm_%j.err
#
# cross_asym for all 76 coupling-passing Oesinghaus pairs, via cascadir.
#
# Reuses binary_ig_all24 -- the SAME signatures behind the published 88%
# (slurm/run_pipeline_full19.slurm) -- so coupling and direction stay on one
# provenance and the new numbers are comparable to the thesis. No retraining.
#
# Aborts unless the 17 audited pairs reproduce 15/17 first.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

DS=/cs/labs/mornitzan/yam.arieli/datasets
OUT=${OUT:-results/oes_direction}
mkdir -p "$OUT"

$PY scripts/run_oesinghaus_direction_coupled76.py \
    --coupling_csv      reports/coupling_figures_draft/donor_coupling_hub_IG_vsPBS.csv \
    --audit_csv         reports/cascade_pairs/cytokine_axes_audited.csv \
    --binary_ig_parquet results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet \
    --manifest_path     "$DS/Oesinghaus_pseudotubes/manifest.json" \
    --hvg_path          "$DS/Oesinghaus_pseudotubes/hvg_list.json" \
    --output_dir        "$OUT" \
    --top_n 50 --min_cells 10 --n_null_perms 100

echo "=== DONE ==="
ls -la "$OUT"
