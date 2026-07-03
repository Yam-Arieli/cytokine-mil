#!/bin/bash
#SBATCH --job-name=confdir_control
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/confusion_direction/slurm_control_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/confusion_direction/slurm_control_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=short

# FREE full-gene CONTROL (no training): confusion-direction accuracy on the existing
# 4000-HVG multiclass 3-seed run. Gates the GPU spend on the signature-restricted run.
# dynamics.pkl ~1.3 GB each (softmax trajectories) -> batch node, not gateway.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/confusion_direction reports/confusion_direction
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY -u scripts/score_confusion_direction.py \
    --base_dir results/attention_dynamics --seeds 42 123 7 \
    --label full_gene_control \
    --out reports/confusion_direction/CONTROL_full_gene.md
echo "CONTROL DONE"
