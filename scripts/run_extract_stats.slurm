#!/bin/bash
#SBATCH --job-name=extract_stats
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/extract_stats_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/extract_stats_%j.err
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/extract_real_data_stats.py \
    --encoder_run results/oesinghaus_full/new_seeds_seed1 \
    --n_max_tubes_expr 200 \
    --n_max_tubes_emb  50

echo "Done."
