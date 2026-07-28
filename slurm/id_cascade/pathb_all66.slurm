#!/bin/bash
#SBATCH --job-name=id_pb66
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_pb66_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_pb66_%j.err
#
# Path B (cross_asym) over ALL 66 ID pairs, not just the 12 labeled benchmark ones.
#
# bridge.slurm passes --axes_csv reports/immune_dictionary/id_axes_labeled.csv, so the 54
# unlabeled pairs that the coupling gate ranks have never had a direction computed. This
# reuses the SAME binary-IG signatures behind the published 83% -- same model, more pairs,
# no retraining -- so the new values are directly comparable to the benchmark ones.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

RES=${RES:-results/id_cascade}
MAN=${MAN:-/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/manifest.json}
HVG=${HVG:-/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/hvg_list.json}
OUT=$RES/pathB_all66

echo "=== ID Path B over all 66 pairs ==="
$PY scripts/run_pipeline_a_bridge_b.py \
    --axes_csv reports/immune_dictionary/id_axes_all66.csv \
    --binary_ig_parquet "$RES/binary_ig/binary_ig.parquet" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --output_dir "$OUT" \
    --top_n 50 --min_cells 10 --n_null_perms 100

echo "=== DONE ==="
ls -la "$OUT"
