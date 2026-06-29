#!/bin/bash
#SBATCH --job-name=attn_dyn_extract
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attention_dynamics/slurm_extract_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attention_dynamics/slurm_extract_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# §33 Part B — extract per-cell-type attention trajectory from each seed's
# every-epoch checkpoints (CPU). Caches H once (frozen encoder) then reconstructs
# attention per checkpoint — ~minutes for 250 checkpoints. Produces
# seed_<seed>/attention_trajectory.pkl. 6h ceiling is generous.

seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/extract_attention_trajectory.py \
    --run_dir results/attention_dynamics/seed_$SEED \
    --device cpu
