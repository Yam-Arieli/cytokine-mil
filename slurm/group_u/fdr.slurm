#!/bin/bash
#SBATCH --job-name=gu_fdr
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_fdr_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_fdr_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Group-U (§27.3): BH-FDR + Storey pi0 over Group U + verdict P1-P4.
# Primary = train_only (pre-registered); pooled is the §26 regression anchor.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

echo "=== PRIMARY (pre-registered): train donors only ==="
$PY scripts/run_group_u_fdr.py \
    --per_axis_csv results/group_u/pipeline_full121_train_only/per_axis_summary.csv \
    --audit_csv    reports/cascade_pairs/cytokine_axes_audited.csv \
    --out          reports/cascade_pairs/GROUP_U_RESULTS.md

echo ""
echo "=== ANCHOR (§26 comparison): pooled all donors ==="
$PY scripts/run_group_u_fdr.py \
    --per_axis_csv results/group_u/pipeline_full121_pooled/per_axis_summary.csv \
    --audit_csv    reports/cascade_pairs/cytokine_axes_audited.csv \
    --out          reports/cascade_pairs/GROUP_U_RESULTS_pooled.md

echo ""
echo "DONE. Bottom line: reports/cascade_pairs/GROUP_U_RESULTS.md"
