#!/bin/bash
#SBATCH --job-name=attnreg_compare
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_reg/slurm_compare_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_reg/slurm_compare_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=short

# §33 — final cross-experiment comparison (waits afterok for all 3 analyses):
# baseline vs A/B/C on P1@ep10, P1@final, collapse magnitude, and final p_correct.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY -u scripts/compare_attn_experiments.py \
    --baseline_dir results/attention_dynamics \
    --exp_base results/attn_reg --exps A B C --seeds 42 123 7 \
    --out reports/attention_dynamics/INTERVENTION_COMPARISON.md
