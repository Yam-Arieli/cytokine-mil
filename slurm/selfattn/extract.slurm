#!/bin/bash
#SBATCH --job-name=selfattn_extract
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/selfattn/slurm_extract_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/selfattn/slurm_extract_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# §34 Part B — extract BOTH trajectories from each seed's every-epoch checkpoints
# (CPU). Frozen encoder ⇒ cache H once then reconstruct per checkpoint:
#   attention_trajectory.pkl   (pooling weights → §33 P1-P4, comparable to AB-MIL)
#   interaction_trajectory.pkl (cell×cell → cell-type×cell-type M[τ,σ] + off-diag)

seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/extract_selfattn_trajectory.py \
    --run_dir results/selfattn/seed_$SEED \
    --sab_heads 4 --sab_layers 1 \
    --device cpu
