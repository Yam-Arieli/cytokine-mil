#!/bin/bash
#SBATCH --job-name=selfattn_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/selfattn/slurm_analysis_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/selfattn/slurm_analysis_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# §34 analysis (afterok extract). Per seed: §33 pooling-head P1-P4 + the new
# interaction readout (G0 gate, directed graph, relay). Then ONCE across seeds:
# the direction-accuracy table vs the IG cross_asym 88% benchmark + §33 probe/plot.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p reports/selfattn_dynamics
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

for SEED in 42 123 7; do
    RUN=results/selfattn/seed_$SEED
    [ -f "$RUN/attention_trajectory.pkl" ] || { echo "skip seed $SEED (no trajectory)"; continue; }
    if [ "$SEED" = "42" ]; then
        POOL_REPORT=reports/selfattn_dynamics/POOLING_P1P4_RESULTS.md
        INT_REPORT=reports/selfattn_dynamics/INTERACTION_RESULTS.md
    else
        POOL_REPORT=$RUN/POOLING_P1P4_seed$SEED.md
        INT_REPORT=$RUN/INTERACTION_seed$SEED.md
    fi
    $PY scripts/analyze_attention_dynamics.py  --run_dir "$RUN" --report "$POOL_REPORT"
    $PY scripts/analyze_selfattn_interaction.py --run_dir "$RUN" --report "$INT_REPORT"
done

# §33 collapse probe + per-cell-type plot on the pooling head (cross-seed).
$PY -u scripts/probe_attention_p1_over_epochs.py --base_dir results/selfattn --seeds 42 123 7 || true
$PY -u scripts/plot_attention_by_celltype.py     --base_dir results/selfattn --seeds 42 123 7 || true

# The 88%-comparable direction-accuracy table (across seeds) — the headline deliverable.
$PY -u scripts/score_selfattn_direction.py --base_dir results/selfattn --seeds 42 123 7 \
    --out reports/selfattn_dynamics/SELFATTN_RESULTS.md
