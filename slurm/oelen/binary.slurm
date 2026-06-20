#!/bin/bash
#SBATCH --job-name=oelen_bin
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen/slurm_bin_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen/slurm_bin_%j.err
#
# Oelen Bridge: binary AB-MIL (condition vs PBS) for the 6 pathogen x time conditions.
# Shared Stage-1 encoder + 6 binary heads (WIDE config). Reuses the ID trainer.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

RES=${RES:-results/oelen}
MAN=${MAN:-/cs/labs/mornitzan/yam.arieli/datasets/Oelen2022_pseudotubes/manifest.json}
HVG=${HVG:-/cs/labs/mornitzan/yam.arieli/datasets/Oelen2022_pseudotubes/hvg_list.json}
OUT=$RES/binary
mkdir -p "$OUT"

# Pick a real held-out donor from the manifest (composite chem_assignment).
VAL=$($PY -c "import json; m=json.load(open('$MAN')); print(sorted({e['donor'] for e in m})[0])")
echo "=== Oelen binary AB-MIL  MAN=$MAN  val_donor=$VAL ==="
$PY scripts/train_immune_dictionary_binary.py \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --targets 3hCA 24hCA 3hPA 24hPA 3hMTB 24hMTB \
    --val_donors "$VAL" \
    --output_dir "$OUT"
echo "=== DONE oelen binary ==="
