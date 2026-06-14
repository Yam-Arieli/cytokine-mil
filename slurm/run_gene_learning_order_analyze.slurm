#!/bin/bash
#SBATCH --job-name=glo_analyze
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/gene_learning_order/slurm_analyze_%A.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/gene_learning_order/slurm_analyze_%A.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Gene learning-order experiment — attribution extraction + analysis (CPU).
# Run AFTER run_gene_learning_order_train.slurm has completed all 3 seeds.
#
# Steps per seed:
#   1. extract_gene_attribution_trajectory.py
#      -> results/gene_learning_order/seed_<seed>/gene_attribution_trajectory.parquet
#   2. compute_sheu_realtime_emergence.py (once; all seeds share the same raw data)
#      -> results/gene_learning_order/realtime_emergence.csv
#   3. analyze_gene_learning_order.py --real
#      -> results/gene_learning_order/seed_<seed>/learning_order_results.json
#
# Note: realtime_emergence is computed once (stimulus=polyIC) and reused across seeds.

set -e

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

PYTHON=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUT_BASE=results/gene_learning_order

# --- Step 1 & 3: per-seed extraction + analysis ---
for SEED in 42 123 7; do
    SEED_DIR=$OUT_BASE/seed_$SEED
    echo "=== Seed $SEED: attribution extraction ==="
    $PYTHON scripts/extract_gene_attribution_trajectory.py \
        --run_dir $SEED_DIR \
        --seed $SEED \
        --device cpu

    echo "=== Seed $SEED: analysis (--real) ==="
    $PYTHON scripts/analyze_gene_learning_order.py \
        --trajectory $SEED_DIR/gene_attribution_trajectory.parquet \
        --realtime   $OUT_BASE/realtime_emergence.csv \
        --output_dir $SEED_DIR
done

# --- Step 2: real-time emergence (once; requires raw Sheu data) ---
# Computed last so any per-seed attribution issues are caught first.
# If the raw data is unavailable, comment out this block and provide
# a hand-crafted realtime_emergence.csv.
echo "=== Computing real-time emergence (polyIC, single run) ==="
$PYTHON scripts/compute_sheu_realtime_emergence.py \
    --stimulus polyIC \
    --output_dir $OUT_BASE

echo "=== All done. Results in $OUT_BASE ==="
