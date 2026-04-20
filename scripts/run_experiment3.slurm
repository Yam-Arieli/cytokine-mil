#!/bin/bash
#SBATCH --job-name=exp3_aux
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp3_aux_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp3_aux_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2

# Array tasks map to seeds: 0->42, 1->123, 2->7
SEEDS=(42 123 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RESULTS_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full

RUN_DIR=$(ls -d ${RESULTS_DIR}/run_*_seed${SEED} 2>/dev/null | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: No run directory found for seed=${SEED} in ${RESULTS_DIR}"
    exit 1
fi

echo "============================================"
echo "Experiment 3: AuxDecoder (new pipeline)"
echo "  seed=${SEED}"
echo "  run_dir=${RUN_DIR}"
echo "============================================"

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/run_experiment3.py \
    --run_dir "${RUN_DIR}" \
    --device cpu \
    --epochs 50 \
    --lr 0.01 \
    --seed ${SEED} \
    --n_permutations 1000

echo "Done — seed=${SEED}."
