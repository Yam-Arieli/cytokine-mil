#!/bin/bash
#SBATCH --job-name=abl_inspect
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/abl_inspect_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/abl_inspect_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=short

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
python scripts/ablation_analysis_inspect.py
