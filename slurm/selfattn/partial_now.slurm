#!/bin/bash
#SBATCH --job-name=selfattn_partial
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/selfattn/slurm_partial_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/selfattn/slurm_partial_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --partition=short

# §34 PARTIAL result NOW (from current seed_42 checkpoints, while training runs).
# Snapshots the checkpoints into an isolated base dir (no concurrent-write / no
# collision with the running extract/analysis jobs), stride 5 for speed, then runs
# the full readout: G0 interaction gate + heatmaps + direction table vs IG 88%.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

SRC=results/selfattn/seed_42
BASE=results/selfattn_partial
DST=$BASE/seed_42
mkdir -p $DST/checkpoints reports/selfattn_dynamics

# Freeze a snapshot of whatever checkpoints exist right now.
cp $SRC/label_encoder.json $SRC/manifest_train.json $DST/
cp $SRC/checkpoints/epoch_*.pt $DST/checkpoints/
echo "PARTIAL: snapshotted $(ls $DST/checkpoints | wc -l) checkpoints"

$PY -u scripts/extract_selfattn_trajectory.py --run_dir $DST \
    --sab_heads 4 --sab_layers 1 --checkpoint_stride 5 --device cpu

$PY scripts/analyze_attention_dynamics.py  --run_dir $DST \
    --report reports/selfattn_dynamics/PARTIAL_POOLING_seed42.md || true
$PY scripts/analyze_selfattn_interaction.py --run_dir $DST \
    --report reports/selfattn_dynamics/PARTIAL_INTERACTION_seed42.md
$PY scripts/score_selfattn_direction.py --base_dir $BASE --seeds 42 \
    --out reports/selfattn_dynamics/PARTIAL_SELFATTN_RESULTS.md || true
echo "PARTIAL DONE"
