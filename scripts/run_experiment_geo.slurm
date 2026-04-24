#!/bin/bash
#SBATCH --job-name=exp_geo_fix
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp_geo_fix_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp_geo_fix_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2

# Geometry engineering fixes: PBS-relative and h_residual.
# 3 tasks — one per MIL seed (42, 123, 7).
# No decoder training. Fully deterministic per MIL seed.
# Each task runs BOTH experiments (pbs_rel + h_residual) and saves results to:
#   <run_dir>/experiment_geo_pbs_rel/latent_geometry.pkl
#   <run_dir>/experiment_geo_hresid/latent_geometry.pkl

MIL_SEEDS=(42 123 7)
MIL_SEED=${MIL_SEEDS[$SLURM_ARRAY_TASK_ID]}

RESULTS_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full

RUN_DIR=$(ls -d ${RESULTS_DIR}/run_*_seed${MIL_SEED} 2>/dev/null | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: No run directory found for MIL seed=${MIL_SEED} in ${RESULTS_DIR}"
    exit 1
fi

echo "========================================================"
echo "Experiment: Geometry Fixes (PBS-relative + h_residual)"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  mil_seed   : ${MIL_SEED}"
echo "  run_dir    : ${RUN_DIR}"
echo "========================================================"

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/run_experiment_geo.py \
    --run_dir "${RUN_DIR}" \
    --device cpu \
    --n_permutations 1000

echo "Done — mil_seed=${MIL_SEED}."
