#!/bin/bash
#SBATCH --job-name=dc_coup
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_coupling_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_coupling_%j.err
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=short

# Stage 6 (§40): signature-space coupling for one ARM, donor-level gate + degree
# correction, then BH-FDR over the whole pair family. One k=4 tube-set copy is ~25 GB and
# the donor-level path holds ~3 (it adds the per-donor cells_by_pair dicts). top_n does
# not change tube memory, so §37's ask carries over unchanged.
#
# Env: ARM ("curated" = the §40 result, "raw" = the uncurated control). Both are run, as
# independent branches off merge, so their wall-clock overlaps.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

ARM="${ARM:-curated}"
echo "[coupling] arm=$ARM"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/run_oes90_dc_coupling.py \
    --out_dir results/oes90_dc \
    --arm "$ARM"
