#!/bin/bash
#SBATCH --job-name=oelen_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen_sa_ca/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen_sa_ca/slurm_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=short

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_oelen_stage2sa_stage3ca.py
