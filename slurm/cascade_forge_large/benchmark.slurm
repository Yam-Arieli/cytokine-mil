#!/bin/bash
#SBATCH --job-name=cfl_bench
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_large/slurm_bench_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_large/slurm_bench_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --array=0-6

# Benchmark cascadir on each forged snapshot (one array task per h5ad in the manifest).
# Trains Stage-1 encoder on ~1M cells + 24 binary models (Stage-2 embedding cache speedup),
# then scores direction (cross_asym vs symmetric control) + signature coupling (donor-level).

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
MAN=results/cascade_forge_large/h5ad_manifest.txt

H5=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAN")
if [ -z "$H5" ]; then
    echo "no h5ad at manifest line $((SLURM_ARRAY_TASK_ID + 1)); nothing to do"
    exit 0
fi
echo "=== benchmarking $H5 ==="
$PY scripts/benchmark_large_cascade.py --h5ad "$H5"
