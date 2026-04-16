#!/bin/bash
#SBATCH --job-name=aux_decoder
#SBATCH --output=logs/aux_decoder_%j.out
#SBATCH --error=logs/aux_decoder_%j.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2

# Array tasks map to seeds: 0->42, 1->123, 2->7
SEEDS=(42 123 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RESULTS_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full

# Find run directory for this seed.
RUN_DIR=$(ls -d ${RESULTS_DIR}/run_*_seed${SEED} 2>/dev/null | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: No run directory found for seed=${SEED} in ${RESULTS_DIR}"
    exit 1
fi

echo "============================================"
echo "Experiment 3: AuxDecoder — seed=${SEED}"
echo "Run dir: ${RUN_DIR}"
echo "============================================"

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/train_aux_decoder.py \
    --run_dir "${RUN_DIR}" \
    --device cpu \
    --tau_values 0.3 0.5 1.0 \
    --epochs 50 \
    --n_permutations 1000

echo "Done."
