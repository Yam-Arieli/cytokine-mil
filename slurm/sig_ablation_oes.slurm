#!/bin/bash
#SBATCH --job-name=sig_ablation_oes
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_oes_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_oes_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=short

# Signature-definition ablation (2x2): {IG, DE} x {vs-PBS, vs-panel} on Oesinghaus.
# CPU-only, no training. Uses the canonical SECTION-26 binary_ig_all24 (24 labeled
# cytokines, covers all 17 benchmark pairs, reproduces the published ~88% direction)
# -- NOT the regressed all-45 group_u one. Train donors only (D2/D3 excluded),
# matching Path A's PBS-RC split.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

IG=results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json
AUDIT=reports/cascade_pairs/cytokine_axes_audited.csv

mkdir -p results/sig_ablation

$PY scripts/run_signature_ablation.py \
    --dataset oesinghaus \
    --binary_ig_parquet "$IG" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --audit_csv "$AUDIT" \
    --exclude_donors Donor2 Donor3 \
    --top_n 50 --min_cells 10 --n_perm 1000 \
    --output_dir results/sig_ablation/oes

echo "DONE. report at results/sig_ablation/oes/ablation_report.md"
