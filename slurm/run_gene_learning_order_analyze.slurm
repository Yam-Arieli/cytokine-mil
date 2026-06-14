#!/bin/bash
#SBATCH --job-name=glo_analyze
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/gene_learning_order/slurm_analyze_%A.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/gene_learning_order/slurm_analyze_%A.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Gene learning-order — attribution extraction + real-time emergence + analysis (CPU).
# Run AFTER run_gene_learning_order_train.slurm completes all 3 seeds.
# ORDER MATTERS: realtime_emergence.csv must exist BEFORE the --real analysis.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PYTHON=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUT_BASE=results/gene_learning_order

# --- Step 1: per-seed attribution extraction (saves parquets even if later steps fail) ---
for SEED in 42 123 7; do
    echo "=== Seed $SEED: attribution extraction ==="
    $PYTHON scripts/extract_gene_attribution_trajectory.py \
        --run_dir $OUT_BASE/seed_$SEED --seed $SEED --device cpu
done

# --- Step 2: real-time emergence (once; shared by all seeds; MUST precede analysis) ---
echo "=== Computing real-time emergence (polyIC) ==="
$PYTHON scripts/compute_sheu_realtime_emergence.py \
    --stimulus polyIC --output_dir $OUT_BASE

# --- Step 3: per-seed H1/H2 analysis (needs realtime_emergence.csv) ---
for SEED in 42 123 7; do
    echo "=== Seed $SEED: analysis (--real) ==="
    $PYTHON scripts/analyze_gene_learning_order.py \
        --trajectory $OUT_BASE/seed_$SEED/gene_attribution_trajectory.parquet \
        --realtime   $OUT_BASE/realtime_emergence.csv \
        --output_dir $OUT_BASE/seed_$SEED
done

echo "=== All done. Results in $OUT_BASE ==="
