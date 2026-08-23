#!/bin/bash
#SBATCH --job-name=o90_coup
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_coupling_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_coupling_%j.err
#SBATCH --mem=260G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=short

# Stage 4: donor-level, degree-corrected signature coupling for all 4005 pairs, + BH-FDR
# over the neutral family. CPU-only. The memory ask is the whole point of the stage:
# tube_set + pooled cells_by_pair + the 10 per-donor cells_by_pair dicts, ~3 x 64 GB.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/run_oesinghaus_full90_coupling.py --output_dir results/oes_full90

touch results/oes_full90/DONE_coupling
