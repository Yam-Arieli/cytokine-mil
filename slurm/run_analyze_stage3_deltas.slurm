#!/bin/bash
#SBATCH --job-name=stage3_deltas
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/stage3_deltas.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/stage3_deltas.err
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
python scripts/analyze_stage3_deltas.py
cp /tmp/stage3_ca_deltas.csv results/oesinghaus_stage3_ca/stage3_ca_deltas.csv
