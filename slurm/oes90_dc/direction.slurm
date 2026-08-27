#!/bin/bash
#SBATCH --job-name=dc_dir
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_direction_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_direction_%A_%a.err
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-3

# Stage 7 (§40): cross_asym direction + the per-cell-type engagement values, for one ARM,
# SHARDED four ways. ~2 tube-set copies at ~25 GB each; the per-pair random-gene-set null
# dominates runtime and scales with signature size, so top-200 roughly doubles §37's 20 h
# for 4005 pairs. Four shards keep every task well inside the `short` partition's 2-day
# limit, where one 40 h job would be a single point of failure.
#
# Env: ARM ("curated" or "raw"). merge_oes90_dc_direction.py stitches the shards.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

ARM="${ARM:-curated}"
echo "[direction] arm=$ARM shard=${SLURM_ARRAY_TASK_ID:-0}/4"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/run_oes90_dc_direction.py \
    --out_dir results/oes90_dc \
    --arm "$ARM" \
    --pairs_shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --n_shards 4
