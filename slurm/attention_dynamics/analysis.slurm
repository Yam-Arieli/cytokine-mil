#!/bin/bash
#SBATCH --job-name=attn_dyn_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attention_dynamics/slurm_analysis_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attention_dynamics/slurm_analysis_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# §33 Part C — readouts P1..P4 + figures + verdict, per seed. The headline report
# is seed 42; seeds 123/7 land in their own run dirs for the multi-seed read
# (trajectories are point estimates until seeds agree — see PRE_REGISTRATION.md).

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p reports/attention_dynamics

PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

for SEED in 42 123 7; do
    RUN=results/attention_dynamics/seed_$SEED
    [ -f "$RUN/attention_trajectory.pkl" ] || { echo "skip seed $SEED (no trajectory)"; continue; }
    if [ "$SEED" = "42" ]; then
        REPORT=reports/attention_dynamics/ATTENTION_DYNAMICS_RESULTS.md
    else
        REPORT=$RUN/ATTENTION_DYNAMICS_RESULTS_seed$SEED.md
    fi
    $PY scripts/analyze_attention_dynamics.py --run_dir "$RUN" --report "$REPORT"
done
