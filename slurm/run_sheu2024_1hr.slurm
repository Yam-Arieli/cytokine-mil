#!/bin/bash
#SBATCH --job-name=sheu_1hr
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_1hr_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_1hr_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Track B: Stage 1+2 training on the 1hr pseudotube manifest.
seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/sheu2024_1hr
OUT_DIR=results/sheu2024_1hr/seed_${SEED}
mkdir -p "$OUT_DIR"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_sheu2024_stage12.py \
    --seed "$SEED" \
    --output_dir "$OUT_DIR" \
    --manifest_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_1hr_pseudotubes/manifest.json \
    --hvg_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_1hr_pseudotubes/hvg_list.json
