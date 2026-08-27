#!/bin/bash
#SBATCH --job-name=dc_dmrg
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_dirmerge_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_dirmerge_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --partition=short

# Stage 8 (§40): concatenate one arm's four direction shards into the single per-arm
# tables, verifying the shards agree on provenance and together cover the pair list
# exactly once. Pure concatenation — no statistic is computed here. Writes DONE_direction_<ARM>.
#
# Env: ARM ("curated" or "raw").
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

ARM="${ARM:-curated}"
echo "[dirmerge] arm=$ARM"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/merge_oes90_dc_direction.py \
    --out_dir results/oes90_dc \
    --arm "$ARM" \
    --n_shards 4
