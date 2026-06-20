#!/bin/bash
#SBATCH --job-name=oelen_coup
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen/slurm_coup_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen/slurm_coup_%j.err
#
# Oelen "Both" analysis (depends on oelen_bin):
#   (1) DONOR-LEVEL coupling gate at ~121 donors -- the first run of the donor-level
#       null + degree correction at the scale it was designed for (open thread).
#   (2) cell-level cross_asym -> the 3h->24h PROGRESSION direction per pathogen.
# Same binary_ig signatures feed both. Benchmark = oelen_axes_labeled.csv
# (same-pathogen pairs coupled; expected_sign -1 = 3h upstream).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

RES=${RES:-results/oelen}
MAN=${MAN:-/cs/labs/mornitzan/yam.arieli/datasets/Oelen2022_pseudotubes/manifest.json}
HVG=${HVG:-/cs/labs/mornitzan/yam.arieli/datasets/Oelen2022_pseudotubes/hvg_list.json}
IG=$RES/binary_ig/binary_ig.parquet
BIN_RUN=$RES/binary
AXES=reports/oelen/oelen_axes_labeled.csv

# --- IG: discover S_X from the binary models ---
mkdir -p "$RES/binary_ig"
TARGETS=$(ls "$BIN_RUN"/model_*.pt 2>/dev/null | sed -E 's#.*/model_(.*)\.pt#\1#' | tr '\n' ' ')
echo "=== Oelen IG  targets: [$TARGETS] ==="
if [ -z "$TARGETS" ]; then echo "FATAL: no model_*.pt under $BIN_RUN"; exit 3; fi
$PY scripts/run_binary_ig_probe.py \
    --binary_run_dir "$BIN_RUN" --output_dir "$RES/binary_ig" \
    --manifest_path "$MAN" --hvg_path "$HVG" --targets $TARGETS \
    --top_n 50 --n_ig_steps 20 --max_tubes_per_cytokine 10 --device cpu

# --- (1) DONOR-LEVEL coupling gate @ ~121 donors ---
echo "=== Oelen DONOR-LEVEL coupling gate (~121 donors) ==="
$PY scripts/run_donor_coupling_null.py \
    --dataset oelen \
    --binary_ig_parquet "$IG" --manifest_path "$MAN" --hvg_path "$HVG" \
    --audit_csv "$AXES" \
    --min_donors 8 --top_n 50 --min_cells 10 --n_perm 200 \
    --output_dir "$RES/donor_coupling"

# --- (2) cell-level cross_asym -> 3h->24h progression direction ---
echo "=== Oelen cell-level cross_asym (progression direction) ==="
$PY scripts/run_cell_degree_coupling.py \
    --dataset oelen \
    --binary_ig_parquet "$IG" --manifest_path "$MAN" --hvg_path "$HVG" \
    --axes_csv "$AXES" \
    --top_n 50 --min_cells 10 --n_perm 500 \
    --output_dir "$RES/coupling_cell_degree"

echo "=== DONE oelen coupling ==="
echo "BOTTOM LINE (donor gate):"; cat "$RES/donor_coupling/donor_coupling_report.md" 2>/dev/null | head -30 || true
