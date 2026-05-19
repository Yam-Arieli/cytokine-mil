#!/bin/bash
#SBATCH --job-name=union_meta
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/ablation_union/meta_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/ablation_union/meta_%j.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=short

# Meta-submit: chains 00_build_union → 01_run (array 0-31) → 02_report.
# Submit from gateway via `sbatch slurm/ablation_union/meta_submit.slurm`.
# Override ALIGN_VARIANT (e.g. inner_product_6D) via `sbatch --export=ALL,ALIGN_VARIANT=inner_product_6D ...`.

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil

mkdir -p ${BASE}/logs/ablation_union

echo "=== Meta-submit: union pair list → ablation → triple report ==="

BUILD_JID=$(sbatch --parsable \
    --export=ALL${ALIGN_VARIANT:+,ALIGN_VARIANT=$ALIGN_VARIANT} \
    ${BASE}/slurm/ablation_union/00_build_union.slurm)
echo "00_build_union job: ${BUILD_JID}"

RUN_JID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JID} \
    ${BASE}/slurm/ablation_union/01_run.slurm)
echo "01_run array job  : ${RUN_JID}  (depends on ${BUILD_JID})"

REP_JID=$(sbatch --parsable \
    --dependency=afterok:${RUN_JID} \
    ${BASE}/slurm/ablation_union/02_report.slurm)
echo "02_report job     : ${REP_JID}  (depends on ${RUN_JID})"

{
    echo "00_build_union: ${BUILD_JID}"
    echo "01_run:         ${RUN_JID}"
    echo "02_report:      ${REP_JID}"
} > ${BASE}/logs/ablation_union/chain_ids.txt
echo "Saved chain_ids.txt"
