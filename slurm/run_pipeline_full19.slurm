#!/bin/bash
#SBATCH --job-name=ab_pipe_19
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_ab_pipe_19_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_ab_pipe_19_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Full-19 pipeline sequence — runs in one SLURM job:
#
#   1. Binary IG probe on the 16 newly-trained cytokines (~5-10 min)
#   2. Merge old + new binary_ig.parquet into a 24-cytokine parquet (~5s)
#   3. Pipeline driver (pooled all donors)             (~5 min)
#   4. Pipeline driver (D2/D3 excluded — train donors) (~5 min)
#
# After this completes, the 17 KNOWN_DIRECTIONAL + 2 PRE_REGISTERED ground-
# truth axes + several KNOWN_COREGULATED null-direction axes are all
# evaluable; this is the statistically powered evaluation.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

# Locate the most recent missing-16 training run dir
MISSING16_RUN=$(ls -t results/oesinghaus_binary_missing16/ | grep "^run_" | head -1)
if [ -z "$MISSING16_RUN" ]; then
    echo "FATAL: no run_* dir under results/oesinghaus_binary_missing16/" >&2
    exit 2
fi
echo "Using missing16 training run: $MISSING16_RUN"

mkdir -p results/gene_dynamics_phase0/binary_ig_missing16
mkdir -p results/gene_dynamics_phase0/binary_ig_all24
mkdir -p results/gene_dynamics_phase0/pipeline_a_b_full19
mkdir -p results/gene_dynamics_phase0/pipeline_a_b_full19_train_only

# ---------------------------------------------------------------
# Step 1: Binary IG probe on the 16 newly-trained cytokines
# ---------------------------------------------------------------
echo ""
echo "=== Step 1/4: Binary IG probe on missing 16 ==="
$PY scripts/run_binary_ig_probe.py \
    --binary_run_dir "results/oesinghaus_binary_missing16/$MISSING16_RUN" \
    --output_dir     "results/gene_dynamics_phase0/binary_ig_missing16" \
    --targets        IFN-gamma IFN-omega IFN-lambda1 IL-15 IL-17A IL-36-alpha \
                     IL-9 IL-13 IL-27 IL-16 CD30L Decorin VEGF GM-CSF TL1A IL-35 \
    --top_n 50 \
    --n_ig_steps 20 \
    --max_tubes_per_cytokine 10 \
    --device cpu

# ---------------------------------------------------------------
# Step 2: Merge old + new parquets into a 24-cytokine parquet
# ---------------------------------------------------------------
echo ""
echo "=== Step 2/4: Merge binary_ig parquets ==="
$PY scripts/merge_binary_ig_parquets.py \
    results/gene_dynamics_phase0/binary_ig/binary_ig.parquet \
    results/gene_dynamics_phase0/binary_ig_missing16/binary_ig.parquet \
    results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet

# ---------------------------------------------------------------
# Step 3: Pipeline (pooled all donors)
# ---------------------------------------------------------------
echo ""
echo "=== Step 3/4: Pipeline run, all 12 donors pooled ==="
$PY scripts/run_pipeline_a_bridge_b.py \
    --axes_csv          reports/cascade_pairs/cytokine_axes.csv \
    --binary_ig_parquet results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet \
    --manifest_path     /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json \
    --hvg_path          /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json \
    --output_dir        results/gene_dynamics_phase0/pipeline_a_b_full19 \
    --top_n             50 \
    --min_cells         10 \
    --n_null_perms      100

# ---------------------------------------------------------------
# Step 4: Pipeline (train donors only — D2/D3 excluded)
# ---------------------------------------------------------------
echo ""
echo "=== Step 4/4: Pipeline run, train donors only ==="
$PY scripts/run_pipeline_a_bridge_b.py \
    --axes_csv          reports/cascade_pairs/cytokine_axes.csv \
    --binary_ig_parquet results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet \
    --manifest_path     /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json \
    --hvg_path          /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json \
    --output_dir        results/gene_dynamics_phase0/pipeline_a_b_full19_train_only \
    --top_n             50 \
    --min_cells         10 \
    --n_null_perms      100 \
    --exclude_donors    Donor2 Donor3

echo ""
echo "=== DONE ==="
echo "Verdicts:"
echo "  pooled:    results/gene_dynamics_phase0/pipeline_a_b_full19/verdict.md"
echo "  train_only: results/gene_dynamics_phase0/pipeline_a_b_full19_train_only/verdict.md"
