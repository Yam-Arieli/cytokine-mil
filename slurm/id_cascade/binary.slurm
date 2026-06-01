#!/bin/bash
#SBATCH --job-name=id_bin
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_bin_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_bin_%j.err
#
# Bridge: binary AB-MIL (cytokine vs PBS) for the 12 benchmark cytokines.
# One shared Stage 1 encoder + 12 binary Stage 2 heads (WIDE config).
# Env: MAN, HVG, RES
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

RES=${RES:-results/id_cascade}
MAN=${MAN:-/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/manifest.json}
HVG=${HVG:-/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/hvg_list.json}

OUT=$RES/binary
mkdir -p "$OUT"
echo "=== ID binary AB-MIL (Bridge)  MAN=$MAN ==="
$PY scripts/train_immune_dictionary_binary.py \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --output_dir "$OUT"
echo "=== DONE binary ==="
