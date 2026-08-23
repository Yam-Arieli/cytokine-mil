#!/bin/bash
#SBATCH --job-name=o90_dir
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_direction_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_direction_%j.err
#SBATCH --mem=180G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-08:00:00
#SBATCH --partition=short

# Stage 5: cross_asym direction for all 4005 pairs + the audited-pair regression check.
# Runtime is dominated by the per-pair random-gene-set null (4005 x 100 permutations).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/run_oesinghaus_full90_direction.py --output_dir results/oes_full90

touch results/oes_full90/DONE_direction
