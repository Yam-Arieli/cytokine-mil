#!/bin/bash
#SBATCH --job-name=gu_ig_merge
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_ig_merge_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_ig_merge_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=short

# Group-U (§27): (1) binary IG probe over the newly-trained missing cytokines
# (union of all training chunks), then (2) merge old binary_ig_all24 + new ->
# binary_ig_all45 so the pipeline can direct all 121 Path A axes.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

MISS_DIR=results/group_u/binary_missing
IG_NEW=results/group_u/binary_ig_new
ALL24=results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet
ALL45=results/group_u/binary_ig_all45/binary_ig.parquet
mkdir -p "$IG_NEW" results/group_u/binary_ig_all45

# Union of the per-chunk cytokine lists written by the training array.
TARGETS=$(cat "$MISS_DIR"/groupu_cytokines_chunk*.txt 2>/dev/null | tr '\n' ' ')
echo "IG targets (union of chunks): $TARGETS"
if [ -z "$(echo $TARGETS | tr -d ' ')" ]; then
    echo "FATAL: no trained cytokines found in $MISS_DIR/groupu_cytokines_chunk*.txt" >&2
    exit 2
fi

echo ""
echo "=== Step 1/2: Binary IG probe on the newly-trained missing cytokines ==="
$PY scripts/run_binary_ig_probe.py \
    --binary_run_dir "$MISS_DIR" \
    --output_dir     "$IG_NEW" \
    --targets        $TARGETS \
    --top_n 50 \
    --n_ig_steps 20 \
    --max_tubes_per_cytokine 10 \
    --device cpu

echo ""
echo "=== Step 2/2: Merge binary_ig_all24 + new -> binary_ig_all45 ==="
$PY scripts/merge_binary_ig_parquets.py \
    "$ALL24" \
    "$IG_NEW/binary_ig.parquet" \
    "$ALL45"

echo ""
echo "DONE. Combined parquet: $ALL45"
$PY - <<PYEOF
import pandas as pd
d = pd.read_parquet("$ALL45")
print("all45 cytokines:", d["cytokine"].nunique())
PYEOF
