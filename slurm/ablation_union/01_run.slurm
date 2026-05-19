#!/bin/bash
#SBATCH --job-name=union_ablation
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/ablation_union/ablation_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/ablation_union/ablation_%A_%a.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --array=0-31

# 32 tasks = 8 experiments × 4 shards.
# Reads ${EXP_DIR}/top_pairs_union.json (built by 00_build_union.slurm) and
# writes ablation_scores_shard_<SHARD>.pkl/.csv next to it. We deliberately
# DO NOT clobber the geo-based ablation outputs from prior runs — those live
# in the same exp_dir under the same filename, so the union ablation outputs
# go to a separate subdirectory.

SEEDS=(42 123 7 456 789 314 271 618)
EXP_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
SHARD=$(( SLURM_ARRAY_TASK_ID % 4 ))
SEED=${SEEDS[$EXP_IDX]}

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/two_stage_pipeline
EXP_DIR=${BASE}/exp_${EXP_IDX}_seed${SEED}
OUT_DIR=${EXP_DIR}/ablation_union

echo "========================================================"
echo "Union-pair Cell-Type Ablation"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  exp_idx    : ${EXP_IDX}   seed=${SEED}"
echo "  shard      : ${SHARD}/4"
echo "  exp_dir    : ${EXP_DIR}"
echo "  out_dir    : ${OUT_DIR}"
echo "========================================================"

set -e

mkdir -p ${OUT_DIR}

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/analyze_cell_type_ablation.py \
    --seed_dirs ${EXP_DIR} \
    --output_dir ${OUT_DIR} \
    --pairs_file ${EXP_DIR}/top_pairs_union.json \
    --shard_idx ${SHARD} \
    --n_shards 4

echo "Done — union ablation exp=${EXP_IDX} shard=${SHARD}."
