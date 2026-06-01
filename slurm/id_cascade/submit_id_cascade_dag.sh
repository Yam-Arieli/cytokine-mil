#!/bin/bash
# Submit the full Immune Dictionary single-frame cross_asym DAG with SLURM
# dependencies, so it runs unattended to the bottom line.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/id_cascade/submit_id_cascade_dag.sh'"
#
# DAG (single time point — ID is 4h only, no per-TP loop):
#   build (CPU, skipped if manifest exists)
#     |-> stage12 (GPU array, 3 seeds) -> geo_gate (CPU)    [Path A]
#     |-> binary  (GPU)                -> bridge   (CPU)    [Bridge + Path B + eval]
#
# Bottom line: results/id_cascade/eval/pipeline_accuracy.md
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

DATASETS=/cs/labs/mornitzan/yam.arieli/datasets
PTDIR=$DATASETS/ImmuneDictionary_pseudotubes
MAN=$PTDIR/manifest.json
HVG=$PTDIR/hvg_list.json
RES=results/id_cascade
mkdir -p "$RES"

# Dry-run mode: SUBMIT=echo will print the sbatch commands instead of submitting.
SUBMIT=${SUBMIT:-sbatch}

# Build step (skipped if manifest exists)
BUILD_DEP=""
if [ ! -f "$MAN" ]; then
    BJ=$($SUBMIT --parsable slurm/id_cascade/build.slurm)
    BUILD_DEP="--dependency=afterok:$BJ"
    echo "build=$BJ  (manifest missing -> building $MAN)"
else
    echo "manifest exists -> skip build"
fi

EXP="ALL,MAN=$MAN,HVG=$HVG,RES=$RES"

# Path A: stage12 (3-seed array) -> latent geometry
S12=$($SUBMIT --parsable $BUILD_DEP --export="$EXP" slurm/id_cascade/stage12.slurm)
GEO=$($SUBMIT --parsable --dependency=afterok:$S12 --export="$EXP" slurm/id_cascade/geo_gate.slurm)

# Bridge: binary AB-MIL -> IG -> Path B (cross_asym) -> eval
BIN=$($SUBMIT --parsable $BUILD_DEP --export="$EXP" slurm/id_cascade/binary.slurm)
BRDG=$($SUBMIT --parsable --dependency=afterok:$BIN --export="$EXP" slurm/id_cascade/bridge.slurm)

echo ""
echo "Submitted DAG:"
echo "  build  = ${BJ:-(skipped)}"
echo "  s12    = $S12  (Path A array, 3 seeds)"
echo "  geo    = $GEO  (latent geometry; secondary)"
echo "  binary = $BIN  (Bridge: 12 benchmark cytokines vs PBS)"
echo "  bridge = $BRDG (IG + Path B cross_asym + retally; PRIMARY)"
echo ""
echo "Bottom line will land at:"
echo "  results/id_cascade/eval/pipeline_accuracy.md"
echo "  results/id_cascade/pathB/per_celltype.csv"
echo "  results/id_cascade/binary_ig/binary_ig.parquet"
echo "Monitor: squeue -u yam.arieli"
