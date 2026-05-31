#!/bin/bash
#SBATCH --job-name=sheu_brdg
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_brdg_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_brdg_%j.err
#
# Bridge IG -> Path B (§24 + cross_asym) -> eval, for one time point.
# Depends on the binary GPU job. Env: TP, MAN, HVG, RES
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

# Defensive: default RES if --export didn't propagate it (the original DAG
# failure: $RES was empty in the bridge job while $TP was set, so BIN_RUN
# pointed at /$TP/binary and matched no models -> empty --targets -> exit 2).
RES=${RES:-results/sheu_cascade}
if [ -z "$TP" ]; then echo "FATAL: TP unset"; exit 3; fi
BIN_RUN=$RES/$TP/binary
IGDIR=$RES/$TP/binary_ig
PBDIR=$RES/$TP/pathB
EVDIR=$RES/$TP/eval
mkdir -p "$IGDIR" "$PBDIR" "$EVDIR"

# Stimuli present (from model_*.pt files)
TARGETS=$(ls "$BIN_RUN"/model_*.pt 2>/dev/null | sed -E 's#.*/model_(.*)\.pt#\1#' | tr '\n' ' ')
echo "=== Sheu binary IG  TP=$TP  BIN_RUN=$BIN_RUN  targets: [$TARGETS] ==="
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

echo "=== Sheu Path B (cross_asym)  TP=$TP ==="
$PY scripts/run_pipeline_a_bridge_b_sheu.py \
    --axes_csv reports/sheu_cascade/sheu_axes_labeled.csv \
    --binary_ig_parquet "$IGDIR/binary_ig.parquet" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --output_dir "$PBDIR" \
    --time_filter "$TP" \
    --top_n 50 --min_cells 10 --n_null_perms 100

echo "=== Sheu eval (retally, cross_asym)  TP=$TP ==="
$PY scripts/retally_pipeline_against_audit.py \
    --pipeline_csv "$PBDIR/per_celltype.csv" \
    --audit_csv reports/sheu_cascade/sheu_axes_labeled.csv \
    --out "$EVDIR/pipeline_accuracy_${TP}.md" \
    --metric cross_asym || echo "retally note: directional-only benchmark"
echo "=== DONE bridge TP=$TP ==="
