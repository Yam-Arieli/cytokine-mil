#!/bin/bash
#SBATCH --job-name=ip_analyze
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/inner_product/analyze_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/inner_product/analyze_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/two_stage_pipeline
SEEDS=(42 123 7 456 789 314 271 618)

EXP_DIRS=""
for IDX in 0 1 2 3 4 5 6 7; do
    EXP_DIRS+=" ${BASE}/exp_${IDX}_seed${SEEDS[$IDX]}"
done

echo "========================================================"
echo "Inner-Product Pair Detection — Cross-Experiment Analysis"
echo "  exp_dirs:"
for d in $EXP_DIRS; do
    echo "    $d"
done
echo "========================================================"

set -e

mkdir -p /cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/inner_product
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/analyze_inner_product_results.py \
    --exp_dirs ${EXP_DIRS} \
    --output_dir ${BASE}/inner_product_analysis

echo "Done — analysis complete. Results in ${BASE}/inner_product_analysis"
