#!/bin/bash
#SBATCH --job-name=sig_abl_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_analysis_oes_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_analysis_oes_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=short

# Analysis + figures for the 2x2 signature ablation. Submit with
#   --dependency=afterok:<experiment_jobid>
# so it runs only after run_signature_ablation.py succeeds. Reads that job's
# CSV outputs AND re-derives the 4 signature sets (needs cells -> 96G) for the
# gene-level overlap / panel-effect figures.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

IG=results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json

$PY scripts/analyze_signature_ablation.py \
    --ablation_dir results/sig_ablation/oes \
    --dataset oesinghaus \
    --binary_ig_parquet "$IG" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --exclude_donors Donor2 Donor3 \
    --top_n 50

echo "DONE. analysis at results/sig_ablation/oes/analysis/analysis_stats.md"
echo "      figures at results/sig_ablation/oes/analysis/plots/"
