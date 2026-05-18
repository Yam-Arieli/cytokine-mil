#!/bin/bash
#SBATCH --job-name=refilter_submit
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/refilter_submit_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/refilter_submit_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Runs filter_top_pairs.py on all 8 experiment dirs (per-source top-K),
# then chains ablation (03_ablation.slurm) and analysis (04_analysis.slurm).
# Submit this from a compute node so sbatch is available.

TOP_K=3
MIN_W=0

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil
RESULTS=${BASE}/results/two_stage_pipeline
PYTHON=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
SEEDS=(42 123 7 456 789 314 271 618)

cd ${BASE}
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

echo "========================================================"
echo "Step 1: Per-source top-${TOP_K} filtering (min_w_stat=${MIN_W})"
echo "========================================================"

for IDX in 0 1 2 3 4 5 6 7; do
    SEED=${SEEDS[$IDX]}
    EXP_DIR=${RESULTS}/exp_${IDX}_seed${SEED}
    echo ""
    echo "--- exp_${IDX}_seed${SEED} ---"
    ${PYTHON} scripts/filter_top_pairs.py \
        --exp_dir  ${EXP_DIR} \
        --top_k_per_source ${TOP_K} \
        --min_w_stat ${MIN_W}
done

echo ""
echo "========================================================"
echo "Step 2: Submit ablation array job"
echo "========================================================"

ABL_JID=$(sbatch --parsable ${BASE}/slurm/two_stage/03_ablation.slurm)
echo "Ablation job ID: ${ABL_JID}"

echo ""
echo "========================================================"
echo "Step 3: Submit analysis (depends on ablation)"
echo "========================================================"

ANA_JID=$(sbatch --parsable \
    --dependency=afterok:${ABL_JID} \
    ${BASE}/slurm/two_stage/04_analysis.slurm)
echo "Analysis job ID: ${ANA_JID}"

echo ""
echo "Chain submitted:"
echo "  filter   : done (this job)"
echo "  ablation : ${ABL_JID}  (array 0-31, 8h limit)"
echo "  analysis : ${ANA_JID}  (depends on ${ABL_JID})"

# Save IDs for reference
echo "refilter_submit: ${SLURM_JOB_ID}" > ${BASE}/logs/two_stage/chain_ids2.txt
echo "ablation: ${ABL_JID}"              >> ${BASE}/logs/two_stage/chain_ids2.txt
echo "analysis: ${ANA_JID}"              >> ${BASE}/logs/two_stage/chain_ids2.txt
echo "Saved to logs/two_stage/chain_ids2.txt"
