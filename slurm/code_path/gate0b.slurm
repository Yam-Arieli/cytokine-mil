#!/bin/bash
#SBATCH --job-name=cp_0b
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_gate0b_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_gate0b_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=short

# Phase 0b — IG transplant on the published anchor (SPEC.md §2). No training: run B's
# weights are reloaded and pushed through both attribution paths.
#
# ~1760 tube-level IG passes (16 cytokines x 4 cascadir configs, plus the production
# probe), each 20 interpolation steps through a 512-wide MLP — hours on CPU, minutes on a
# GPU, so it takes the GPU. Memory is dominated by the main tube subset held in RAM
# (~40 control tubes + ~40 per cytokine), not by the model.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/phase0_ig_transplant.py \
    --out_dir results/code_path/phase0 \
    --device cuda
