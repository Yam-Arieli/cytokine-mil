#!/bin/bash
#SBATCH --job-name=union_build
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/ablation_union/build_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/ablation_union/build_%j.err
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --partition=short

# Builds top_pairs_union.json per seed_dir by merging:
#   - alignment top_pairs_<ALIGN_VARIANT>.json
#   - geo latent_geo_results.pkl
#
# ALIGN_VARIANT defaults to inner_product_128D (best @5% recall in the
# recall_table.csv from inner_product_analysis/). Override via env var to
# try a more permissive variant, e.g. inner_product_6D.

ALIGN_VARIANT="${ALIGN_VARIANT:-inner_product_128D}"

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/two_stage_pipeline
SEEDS=(42 123 7 456 789 314 271 618)

mkdir -p /cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/ablation_union

echo "========================================================"
echo "Build union pair lists"
echo "  ALIGN_VARIANT : ${ALIGN_VARIANT}"
echo "========================================================"

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

for IDX in 0 1 2 3 4 5 6 7; do
    SEED=${SEEDS[$IDX]}
    EXP_DIR=${BASE}/exp_${IDX}_seed${SEED}
    echo "--- exp_${IDX}_seed${SEED} ---"
    python scripts/build_union_pair_list.py \
        --alignment_pairs_file ${EXP_DIR}/top_pairs_${ALIGN_VARIANT}.json \
        --geo_results          ${EXP_DIR}/latent_geo_results.pkl \
        --output               ${EXP_DIR}/top_pairs_union.json
done

echo "Done — union pair lists written."
