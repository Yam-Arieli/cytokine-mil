#!/bin/bash
#SBATCH --job-name=cg_bridge
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cano_gamez/slurm_bridge_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cano_gamez/slurm_bridge_%j.err
#
# Cano-Gamez Bridge: IG (discover S_X) -> cell-level degree-corrected coupling.
# 4 donors -> donor-level null inapplicable (p-floor 1/2^4=0.0625) -> cell-level
# degree path (Sheu/ID style). Validates the §28.2 degree-correction fix on a 4th,
# human, new cell type (CD4 T): does HUB recover Th17<->iTreg (TGF-beta coupling),
# keep Th2 distinct, and SUPPRESS the Th0 activation hub? Depends on cg_bin.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

RES=${RES:-results/cano_gamez}
MAN=${MAN:-/cs/labs/mornitzan/yam.arieli/datasets/CanoGamez_pseudotubes/manifest.json}
HVG=${HVG:-/cs/labs/mornitzan/yam.arieli/datasets/CanoGamez_pseudotubes/hvg_list.json}

BIN_RUN=$RES/binary
IGDIR=$RES/binary_ig
COUPDIR=$RES/coupling_cell_degree
mkdir -p "$IGDIR" "$COUPDIR"

TARGETS=$(ls "$BIN_RUN"/model_*.pt 2>/dev/null | sed -E 's#.*/model_(.*)\.pt#\1#' | tr '\n' ' ')
echo "=== Cano-Gamez binary IG  BIN_RUN=$BIN_RUN  targets: [$TARGETS] ==="
if [ -z "$TARGETS" ]; then
    echo "FATAL: no model_*.pt under $BIN_RUN"; ls -la "$BIN_RUN" 2>&1 | head; exit 3
fi

$PY scripts/run_binary_ig_probe.py \
    --binary_run_dir "$BIN_RUN" \
    --output_dir "$IGDIR" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --targets $TARGETS \
    --top_n 50 --n_ig_steps 20 --max_tubes_per_cytokine 10 --device cpu

echo "=== Cano-Gamez cell-level degree-corrected coupling ==="
$PY scripts/run_cell_degree_coupling.py \
    --dataset cano_gamez \
    --binary_ig_parquet "$IGDIR/binary_ig.parquet" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --axes_csv reports/cano_gamez/cano_axes_labeled.csv \
    --top_n 50 --min_cells 10 --n_perm 500 \
    --output_dir "$COUPDIR"

echo "=== DONE cg bridge ==="
echo "BOTTOM LINE:"
cat "$COUPDIR/cell_degree_report.md" 2>/dev/null || echo "(report missing)"
