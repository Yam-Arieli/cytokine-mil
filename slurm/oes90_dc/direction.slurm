#!/bin/bash
#SBATCH --job-name=dc_dir
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_direction_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_direction_%A_%a.err
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-3%1

# Stage 7 (§40): cross_asym direction + the per-cell-type engagement values, for one ARM,
# SHARDED four ways. The per-pair random-gene-set null dominates runtime and scales with
# signature size, so top-200 roughly doubles §37's 20 h for 4005 pairs. Four shards keep
# every task well inside the `short` partition's 2-day limit, where one 40 h job would be
# a single point of failure.
#
# MEMORY (measured, not estimated). The first attempt asked 80G and died: sacct on job
# 31395278 shows shard 0 COMPLETED at MaxRSS 76.9 GiB — 96% of its own limit — while
# shards 1-3 landed together on one node and were OOM-killed at ~78 GiB each. So the ask
# was simply too tight for a single shard, and co-location removed what little was left
# (shard 1's tube load took 1553 s against shard 0's 96 s: IO/page-cache thrash). 140G
# gives ~1.8x the measured peak, and `%1` stops an arm from racing itself onto one node.
# Do NOT lower either without re-measuring: the peak is set by the tube set, which every
# shard loads in full regardless of how few pairs it scores.
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
