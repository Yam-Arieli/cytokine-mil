#!/bin/bash
#SBATCH --job-name=two_stage_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/analysis_%A.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/analysis_%A.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/two_stage_pipeline

echo "========================================================"
echo "Two-Stage Pipeline Analysis"
echo "  base : ${BASE}"
echo "========================================================"

set -e

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

# Collect all 8 experiment directories
EXP_DIRS=$(ls -d ${BASE}/exp_*_seed* 2>/dev/null | sort | tr '\n' ' ')

if [ -z "${EXP_DIRS}" ]; then
    echo "ERROR: no experiment directories found under ${BASE}"
    exit 1
fi

echo "Experiment directories:"
for d in ${EXP_DIRS}; do echo "  ${d}"; done
echo ""

python scripts/analyze_two_stage_results.py \
    --exp_dirs ${EXP_DIRS} \
    --output_dir ${BASE}/analysis \
    --n_shards 4

echo "Done — analysis complete. Results in ${BASE}/analysis"
