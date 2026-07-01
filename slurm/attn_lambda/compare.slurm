#!/bin/bash
#SBATCH --job-name=attnlam_compare
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_lambda/slurm_compare_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_lambda/slurm_compare_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=short

# §33 lambda sweep — final comparison (waits afterok for all 3 lambda analyses):
# baseline (lambda=0) vs L1/L10/L100 on P1@ep10, P1@final, collapse, p_correct.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY -u scripts/compare_attn_experiments.py \
    --baseline_dir results/attention_dynamics \
    --exp_base results/attn_lambda --exps L1 L10 L100 --seeds 42 123 7 \
    --out reports/attention_dynamics/LAMBDA_SWEEP_COMPARISON.md
