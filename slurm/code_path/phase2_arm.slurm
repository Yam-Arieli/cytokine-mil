#!/bin/bash
#SBATCH --job-name=p2_arm
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_p2arm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_p2arm_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-11%4

# Phase 1+2 stage B: 12 tasks = 4 arms x 3 seeds. Each trains 24 binary heads on the
# published-24 panel and derives every signature with ONE attribution implementation over
# ONE fixed tube set — which Phase 0 licenses (the two IG paths agreed at Jaccard 1.000),
# and which leaves the trained weights as the only variable.
#
#   idx  0-2 : cm encoder + cm stage2  (pure cytokine_mil — Phase 1 reference P)
#   idx  3-5 : cm encoder + cd stage2  (Phase 2 T1)
#   idx  6-8 : cd encoder + cm stage2  (Phase 2 T2)
#   idx 9-11 : cd encoder + cd stage2  (pure cascadir  — Phase 1 reference C)
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

ENC=(cm cm cm cm cm cm cd cd cd cd cd cd)
S2=( cm cm cm cd cd cd cm cm cm cd cd cd)
SEEDS=(42 123 7 42 123 7 42 123 7 42 123 7)
E=${ENC[$SLURM_ARRAY_TASK_ID]}
T=${S2[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "[stage B] arm=${E}_${T} seed=$S"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/phase2_train_arm.py \
    --encoder_path "$E" --stage2_path "$T" --seed "$S" --device cuda
