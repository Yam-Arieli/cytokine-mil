#!/bin/bash
#SBATCH --job-name=cg_bin
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cano_gamez/slurm_bin_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cano_gamez/slurm_bin_%j.err
#
# Bridge: binary AB-MIL (condition vs PBS) for the 4 Cano-Gamez stimulated
# conditions (Th0, Th2, Th17, iTreg). One shared Stage-1 encoder + 4 binary heads.
# Reuses train_immune_dictionary_binary.py (parameterised) with --targets +
# Cano-Gamez paths. WIDE config (4000 HVG). val donor = D2 (smallest).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

RES=${RES:-results/cano_gamez}
MAN=${MAN:-/cs/labs/mornitzan/yam.arieli/datasets/CanoGamez_pseudotubes/manifest.json}
HVG=${HVG:-/cs/labs/mornitzan/yam.arieli/datasets/CanoGamez_pseudotubes/hvg_list.json}

OUT=$RES/binary
mkdir -p "$OUT"
echo "=== Cano-Gamez binary AB-MIL (Bridge)  MAN=$MAN ==="
$PY scripts/train_immune_dictionary_binary.py \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --targets Th0 Th2 Th17 iTreg \
    --val_donors D2 \
    --output_dir "$OUT"
echo "=== DONE cg binary ==="
