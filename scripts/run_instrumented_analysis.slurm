#!/bin/bash
#SBATCH --job-name=instr_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/instr_analysis_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/instr_analysis_%j.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Runs all 4 method analysis scripts independently.
# Each writes to its own subdirectory; no shared state.
# Dependent on both attn and geo trajectory jobs completing.

BASE=/cs/labs/mornitzan/yam.arieli

echo "========================================================"
echo "Instrumented Dynamics Analysis — all 4 methods"
echo "Started: $(date)"
echo "========================================================"

set -e

source ${BASE}/venvs/biovenv/bin/activate
cd ${BASE}/cytokine-mil

echo ""
echo "--- Method 1: Geo Trajectory (temporal profiles) ---"
python scripts/analyze_method1_geo_trajectory.py

echo ""
echo "--- Method 2: Attention Trajectory (cell-type dynamics) ---"
python scripts/analyze_method2_attention_trajectory.py

echo ""
echo "--- Method 3: Trajectory Shape Stability ---"
python scripts/analyze_method3_trajectory_stability.py

echo ""
echo "--- Method 5: Joint Attention + Geometry ---"
python scripts/analyze_method5_joint.py

echo ""
echo "========================================================"
echo "All analyses done: $(date)"
echo "Results in: results/oesinghaus_full/instrumented_analysis/"
echo "========================================================"
