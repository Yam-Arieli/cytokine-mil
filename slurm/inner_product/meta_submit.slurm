#!/bin/bash
#SBATCH --job-name=ip_meta
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/inner_product/meta_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/inner_product/meta_%j.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=short

# Meta-submit: chains 01_run (array 0-7) → 02_analyze.
# Submit this from the gateway via `sbatch slurm/inner_product/meta_submit.slurm`.
BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil

mkdir -p ${BASE}/logs/inner_product

echo "=== Meta-submit: inner-product pair detection chain ==="

RUN_JID=$(sbatch --parsable ${BASE}/slurm/inner_product/01_run.slurm)
echo "01_run array job: ${RUN_JID}"

ANA_JID=$(sbatch --parsable \
    --dependency=afterok:${RUN_JID} \
    ${BASE}/slurm/inner_product/02_analyze.slurm)
echo "02_analyze job: ${ANA_JID}  (depends on ${RUN_JID})"

echo "01_run:    ${RUN_JID}"    > ${BASE}/logs/inner_product/chain_ids.txt
echo "02_analyze: ${ANA_JID}"   >> ${BASE}/logs/inner_product/chain_ids.txt
echo "Saved chain_ids.txt"
