#!/bin/bash
#SBATCH --job-name=e50_coup
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure_ep50/slurm_coupling_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure_ep50/slurm_coupling_%j.err
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=short

# Stage 6: signature-space coupling for all 4005 pairs, donor-level gate + degree
# correction, then BH-FDR over the whole family. One k=4 tube-set copy is ~25 GB and the
# donor-level path holds ~3 (it adds the per-donor cells_by_pair dicts).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/run_oes90_pure_coupling.py --out_dir results/oes90_pure_ep50
