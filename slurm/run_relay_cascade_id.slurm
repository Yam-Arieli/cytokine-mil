#!/bin/bash
#SBATCH --job-name=relay_id
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/relay_cascade_id/slurm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/relay_cascade_id/slurm_%A_%a.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Stage 1 — directed cross-cell relay cascades on the Immune Dictionary.
# Hollow-ridge relay influence (CPU; no GPU needed). 3 seeds in parallel.
# Reads RAW ID tubes for the pre-registered conditions; evaluates gates G1-G4
# for the NK -> responder IFN-gamma relay. See reports/immune_dictionary/RELAY_PREREGISTRATION.md.

set -e
seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/relay_cascade_id

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/relay_cascade_id.py \
    --seed $SEED \
    --output_dir results/relay_cascade_id/seed_$SEED
