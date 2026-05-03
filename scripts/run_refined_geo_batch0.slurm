#!/bin/bash
#SBATCH --job-name=refined_geo_b0
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/refined_geo_b0_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/refined_geo_b0_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11

# Re-run PBS-RC geometry with the refined per-donor Wilcoxon pipeline on all
# Batch-0 (offset=0) trained models. Overwrites the old latent_geometry.pkl
# with a new file that contains BOTH the legacy asymmetry field (for backwards
# compatibility) AND the new `refined` field with Wilcoxon cascade calls.
#
# Array indices map to seeds:
#   0-8  -> seeds 1,2,3,4,5,6,8,9,10  (new_seeds_seed<N>)
#   9    -> seed 42   (run_20260412_161758_seed42)
#   10   -> seed 123  (run_20260412_161803_seed123)
#   11   -> seed 7    (run_20260412_161803_seed7)

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full

case ${SLURM_ARRAY_TASK_ID} in
  0)  RUN_DIR="${BASE}/new_seeds_seed1"  ;;
  1)  RUN_DIR="${BASE}/new_seeds_seed2"  ;;
  2)  RUN_DIR="${BASE}/new_seeds_seed3"  ;;
  3)  RUN_DIR="${BASE}/new_seeds_seed4"  ;;
  4)  RUN_DIR="${BASE}/new_seeds_seed5"  ;;
  5)  RUN_DIR="${BASE}/new_seeds_seed6"  ;;
  6)  RUN_DIR="${BASE}/new_seeds_seed8"  ;;
  7)  RUN_DIR="${BASE}/new_seeds_seed9"  ;;
  8)  RUN_DIR="${BASE}/new_seeds_seed10" ;;
  9)  RUN_DIR="${BASE}/run_20260412_161758_seed42"  ;;
  10) RUN_DIR="${BASE}/run_20260412_161803_seed123" ;;
  11) RUN_DIR="${BASE}/run_20260412_161803_seed7"   ;;
  *)  echo "Unexpected array task id: ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo "========================================================"
echo "Refined PBS-RC Geometry — Batch 0"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  run_dir    : ${RUN_DIR}"
echo "========================================================"

if [ ! -f "${RUN_DIR}/model_stage2.pt" ]; then
    echo "ERROR: model_stage2.pt not found in ${RUN_DIR}"
    exit 1
fi

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

python scripts/run_experiment_geo.py \
    --run_dir "${RUN_DIR}" \
    --device cpu \
    --n_permutations 1000

echo "Done — task=${SLURM_ARRAY_TASK_ID}, run_dir=${RUN_DIR}."
