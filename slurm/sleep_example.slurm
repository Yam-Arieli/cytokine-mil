#!/bin/bash
#SBATCH --job-name=sleep_example
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=short

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/sleep_example.py
