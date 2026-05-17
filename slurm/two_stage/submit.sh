#!/bin/bash
# Master submit script for the two-stage cascade detection pipeline.
# Chains SLURM array jobs with afterok dependencies.
# Usage: bash slurm/two_stage/submit.sh [--dry-run]

set -e

DRY_RUN=0
if [[ "${1}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[DRY RUN — no jobs will be submitted]"
fi

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/two_stage_pipeline
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage

mkdir -p ${LOGDIR}
mkdir -p ${BASE}

echo "================================================"
echo "Two-Stage Cascade Detection Pipeline"
echo "  results : ${BASE}"
echo "  logs    : ${LOGDIR}"
echo "================================================"
echo ""

submit() {
    local desc="$1"; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] sbatch $@"
        echo "12345"  # fake job id
    else
        sbatch --parsable "$@"
    fi
}

# ── Job 1: Train (array 0-7) ──────────────────────────────────────────────────
echo "Submitting Job 1: Train (8 experiments)..."
TRAIN_JID=$(submit "train" ${SCRIPT_DIR}/01_train.slurm)
echo "  Train JID: ${TRAIN_JID}"

# ── Job 2: Geo extract (array 0-7, depends on all train tasks) ───────────────
echo "Submitting Job 2: Geo extract (array 0-7)..."
GEO_JID=$(submit "geo" \
    --dependency=afterok:${TRAIN_JID} \
    ${SCRIPT_DIR}/02_geo_extract.slurm)
echo "  Geo JID: ${GEO_JID}"

# ── Job 3: Ablation (array 0-31, depends on all geo tasks) ───────────────────
echo "Submitting Job 3: Cell-type ablation (array 0-31, 8 exps × 4 shards)..."
ABLATION_JID=$(submit "ablation" \
    --dependency=afterok:${GEO_JID} \
    ${SCRIPT_DIR}/03_ablation.slurm)
echo "  Ablation JID: ${ABLATION_JID}"

# ── Job 4: Analysis (single, depends on all ablation tasks) ──────────────────
echo "Submitting Job 4: Pipeline analysis (single job)..."
ANALYSIS_JID=$(submit "analysis" \
    --dependency=afterok:${ABLATION_JID} \
    ${SCRIPT_DIR}/04_analysis.slurm)
echo "  Analysis JID: ${ANALYSIS_JID}"

echo ""
echo "================================================"
echo "Pipeline submitted."
echo "  Train    : ${TRAIN_JID}    (array 0-7)"
echo "  Geo      : ${GEO_JID}      (array 0-7,  dep: Train)"
echo "  Ablation : ${ABLATION_JID} (array 0-31, dep: Geo)"
echo "  Analysis : ${ANALYSIS_JID} (single,     dep: Ablation)"
echo ""
echo "Monitor: squeue -u \$USER --format='%.12i %.12j %.8t %.12M %.12l %R'"
echo "Logs:    ${LOGDIR}"
echo "Results: ${BASE}"
echo "================================================"
