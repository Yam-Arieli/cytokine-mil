#!/bin/bash
#SBATCH --job-name=rig_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/recurrent_ig/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/recurrent_ig/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Recurrent-IG (CLAUDE.md §31, Part B): one array task per seed. Each trains the
# shared Stage-1 encoder + all 45 binary AB-MIL models (wide config) and captures
# IG every 10 epochs into results/recurrent_ig/seed_<seed>/ig_traj.parquet.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/recurrent_ig

PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
seeds=(42 123 7)
SEED=${seeds[${SLURM_ARRAY_TASK_ID:-0}]}

$PY scripts/run_recurrent_ig_oesinghaus.py \
    --seed ${SEED} \
    --output_dir results/recurrent_ig/seed_${SEED} \
    --checkpoint_every 10
