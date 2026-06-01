#!/bin/bash
#SBATCH --job-name=id_brdg
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_brdg_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_brdg_%j.err
#
# Bridge IG -> Path B (cross_asym) -> eval (retally).  PRIMARY metric per §26.
# Depends on the binary GPU job.  Env: MAN, HVG, RES
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

# Defensive: default RES if --export didn't propagate (Sheu DAG had this bug).
RES=${RES:-results/id_cascade}
MAN=${MAN:-/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/manifest.json}
HVG=${HVG:-/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/hvg_list.json}

BIN_RUN_PARENT=$RES/binary
# When binary.slurm passes --output_dir, models are written DIRECTLY under
# $RES/binary/ (model_*.pt). Prefer that. Only fall back to a run_<ts>/ SUBDIR
# (trailing slash so the glob matches DIRECTORIES, not the run_log.txt file —
# the bug that made BIN_RUN=run_log.txt and failed job 30717270).
BIN_RUN=$BIN_RUN_PARENT
if ! ls "$BIN_RUN"/model_*.pt >/dev/null 2>&1; then
    SUB=$(ls -dt "$BIN_RUN_PARENT"/run_*/ 2>/dev/null | head -1)
    [ -n "$SUB" ] && BIN_RUN=${SUB%/}
fi

IGDIR=$RES/binary_ig
PBDIR=$RES/pathB
EVDIR=$RES/eval
mkdir -p "$IGDIR" "$PBDIR" "$EVDIR"

# Targets: 12 benchmark cytokines, discovered from model_*.pt files.
TARGETS=$(ls "$BIN_RUN"/model_*.pt 2>/dev/null | sed -E 's#.*/model_(.*)\.pt#\1#' | tr '\n' ' ')
echo "=== ID binary IG  BIN_RUN=$BIN_RUN  targets: [$TARGETS] ==="
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

echo "=== ID Path B (cross_asym) ==="
$PY scripts/run_pipeline_a_bridge_b.py \
    --axes_csv reports/immune_dictionary/id_axes_labeled.csv \
    --binary_ig_parquet "$IGDIR/binary_ig.parquet" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --output_dir "$PBDIR" \
    --top_n 50 --min_cells 10 --n_null_perms 100

echo "=== ID eval (retally, cross_asym) ==="
$PY scripts/retally_pipeline_against_audit.py \
    --pipeline_csv "$PBDIR/per_celltype.csv" \
    --audit_csv reports/immune_dictionary/id_axes_labeled.csv \
    --out "$EVDIR/pipeline_accuracy.md" \
    --metric cross_asym || echo "retally note: continuing"

echo "=== DONE bridge ==="
echo ""
echo "BOTTOM LINE:"
cat "$EVDIR/pipeline_accuracy.md" 2>/dev/null || echo "(pipeline_accuracy.md missing)"
