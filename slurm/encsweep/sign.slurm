#!/bin/bash
#SBATCH --job-name=es_sign
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encsweep/slurm_sign_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encsweep/slurm_sign_%A_%a.err
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

ARMS=(pbs_only rand18 rand45 all90)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
PANEL=$(/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python -c \
  "import json;print(' '.join(json.load(open('results/encsweep/encsweep_meta.json'))['panel']))")
echo "[slurm] arm=$ARM panel=$PANEL"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/diagnose_oes90_pure_signature_sign.py \
    --out_dir "results/encsweep/$ARM" \
    --signatures signatures.parquet \
    --conditions $PANEL
