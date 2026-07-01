#!/bin/bash
#SBATCH --job-name=attnlam_extract
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_lambda/slurm_extract_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_lambda/slurm_extract_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# §33 lambda sweep — per-cell-type attention trajectory per seed. Frozen encoder
# (cache H once), data as-is (no cell-type exclusion), Stage-2 checkpoints.
seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUT=results/attn_lambda/L$LAM/seed_$SEED

$PY -u scripts/extract_attention_trajectory.py --run_dir $OUT --device cpu
