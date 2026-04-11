#!/bin/bash
#SBATCH --job-name=oesing_bin_s123
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_binary/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_binary/slurm_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_oesinghaus_binary.py --seed 123
