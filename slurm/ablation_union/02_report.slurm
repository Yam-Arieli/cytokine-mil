#!/bin/bash
#SBATCH --job-name=union_report
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/ablation_union/report_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/ablation_union/report_%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Cross-seed triple synthesis: applies geo ∧ ablation conjunction over all
# 8 seed dirs and emits cascade_triples.csv.

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/two_stage_pipeline
SEEDS=(42 123 7 456 789 314 271 618)

EXP_DIRS=""
for IDX in 0 1 2 3 4 5 6 7; do
    EXP_DIRS+=" ${BASE}/exp_${IDX}_seed${SEEDS[$IDX]}"
done

OUT_DIR=${BASE}/cascade_triples

echo "========================================================"
echo "Triple synthesis: geo ∧ ablation conjunction (>= 5/8 seeds)"
echo "  exp_dirs       : ${EXP_DIRS}"
echo "  ablation_subdir: ablation_union"
echo "  output         : ${OUT_DIR}/cascade_triples.csv"
echo "========================================================"

set -e

mkdir -p ${OUT_DIR}

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

for MODE in strict geo_t_only ablation_only; do
    echo "------ triple report, mode=${MODE} ------"
    python scripts/report_cascade_triples.py \
        --exp_dirs ${EXP_DIRS} \
        --output   ${OUT_DIR}/cascade_triples_${MODE}.csv \
        --alpha 0.05 \
        --min_seeds 5 \
        --ablation_subdir ablation_union \
        --mode ${MODE} \
        --per_seed_dump ${OUT_DIR}/per_seed_triples_${MODE}.json
done

echo "------ pair-level report ------"
python scripts/report_cascade_pairs.py \
    --exp_dirs ${EXP_DIRS} \
    --output_dir ${OUT_DIR} \
    --ablation_subdir ablation_union \
    --min_seeds 5

echo "Done — triple reports: ${OUT_DIR}/cascade_triples_<mode>.csv"
echo "       pair reports:   ${OUT_DIR}/cascade_pairs_per_seed_then_count.csv"
echo "                       ${OUT_DIR}/cascade_pairs_pool_then_call.csv"
