#!/bin/bash
#SBATCH --job-name=ig_probe
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_ig_probe_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_ig_probe_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Probe A — static-IG routing sanity check on trained Oesinghaus Stage 2.
# Computes IG of logit_X w.r.t. input genes for 20 marker genes x 91
# cytokines on the final-epoch model state. CPU-only; ~10-30 min wall.
#
# Pass criteria (pre-registered, see scripts/run_static_ig_probe.py):
#   (1) >= 14/20 markers: biological direct inducer in top-3
#   (2) median magnitude gap (winner - runner_up)/|winner| >= 0.20
#   (3) median rho(IG, mean expression) < 0.70

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/gene_dynamics_phase0

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_static_ig_probe.py \
    --run_dir   results/oesinghaus_full_v2/seed_42 \
    --output_dir results/gene_dynamics_phase0/static_ig_seed42 \
    --device cpu
