#!/bin/bash
#SBATCH --job-name=attnreg_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_reg/slurm_analysis_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_reg/slurm_analysis_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --partition=short

# §33 — per-experiment analysis (waits afterok for this EXP's extract). Runs the
# P1-P4 verdict per seed + the P1-over-epochs probe + the per-cell-type collapse
# plot on this EXP's 3 seeds.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUT=results/attn_reg/$EXP

for SEED in 42 123 7; do
    RUN=$OUT/seed_$SEED
    [ -f "$RUN/attention_trajectory.pkl" ] || { echo "skip seed $SEED (no trajectory)"; continue; }
    $PY scripts/analyze_attention_dynamics.py --run_dir "$RUN" \
        --report "$RUN/ATTENTION_DYNAMICS_RESULTS.md"
done
$PY -u scripts/probe_attention_p1_over_epochs.py --base_dir "$OUT" --seeds 42 123 7
$PY -u scripts/plot_attention_by_celltype.py --base_dir "$OUT" --seeds 42 123 7
