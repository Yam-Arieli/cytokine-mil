#!/bin/bash
#SBATCH --job-name=s1_sign
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_sign_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_sign_%A_%a.err
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=short
#SBATCH --array=0-3%2

# Optional secondary readout: what fraction of each signature's genes are actually
# UP-regulated vs PBS, and does IG's rank order track how much a gene moves. Loads the
# panel's tubes (~18 GB at k=10), so it is capped at 2 concurrent tasks. The analysis
# stage depends on this with afterany — a failure here costs the sign table, not the run.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

ARMS=(pub_replica pub_replica_clean vol_small vol_large)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
# `signatures.parquet` is written by the ANALYSIS stage, which runs after this one, so on
# a first pass it does not exist yet (this is what failed in the §38.3 run). Merge the IG
# chunks here first, reusing analyze_encsweep.merge_chunks — the same function and the same
# output file, so nothing about the merged result depends on which stage got there first.
/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python -c \
  "import sys;sys.path.insert(0,'scripts');from pathlib import Path;\
from analyze_encsweep import merge_chunks;\
d=Path('results/s1sweep/$ARM');\
print('[merge]', 'ok' if merge_chunks(d) is not None else 'NO CHUNKS', d)"

PANEL=$(/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python -c \
  "import json;print(' '.join(json.load(open('results/s1sweep/encsweep_meta.json'))['panel']))")
echo "[slurm] arm=$ARM panel=$PANEL"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/diagnose_oes90_pure_signature_sign.py \
    --out_dir "results/s1sweep/$ARM" \
    --signatures signatures.parquet \
    --conditions $PANEL
