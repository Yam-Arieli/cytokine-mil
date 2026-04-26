#!/bin/bash
#SBATCH --job-name=geo_traj_s3
#SBATCH --array=0-4
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short

seeds=(11 12 13 14 15)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

WORKDIR=/cs/labs/mornitzan/yam.arieli
PYTHON=${WORKDIR}/venvs/biovenv/bin/python
REPO=${WORKDIR}/cytokine-mil
RUN_DIR=${REPO}/results/oesinghaus_full/stage3_seed${SEED}

export MPLCONFIGDIR=${WORKDIR}/.matplotlib_cache

cd ${REPO}
${PYTHON} scripts/run_geo_trajectory.py \
    --run_dir ${RUN_DIR} \
    --checkpoint_subdir checkpoints_stage3 \
    --device cpu
