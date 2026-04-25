#!/bin/bash
#SBATCH --job-name=learnability
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/learnability_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/learnability_%j.err
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

# CPU-only job: loads 12 dynamics.pkl files sequentially (~10 GB each)
# and computes per-cytokine learnability AUC + cross-seed Spearman rho.
# No GPU needed.

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

echo "=============================="
echo "Learnability Ordering Analysis"
echo "Started: $(date)"
echo "=============================="

python scripts/analyze_learnability.py

echo "=============================="
echo "Done: $(date)"
echo "=============================="
