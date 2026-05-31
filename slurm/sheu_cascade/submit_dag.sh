#!/bin/bash
# Submit the full Sheu single-frame cascade-direction DAG with SLURM
# dependencies, so it runs unattended to the bottom line
# (reports/sheu_cascade/timepoint_comparison.md).
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/sheu_cascade/submit_dag.sh'"
#
# DAG per time point T in {1hr, 3hr, 5hr} (each fully single-frame):
#   build_T (CPU, skipped if manifest exists)
#     |-> stage12_T (GPU array, 3 seeds) -> geo_gate_T (CPU)        [Path A]
#     |-> binary_T  (GPU)                -> bridge_T  (CPU)         [Bridge+PathB+eval]
#   compare (CPU) <- all three bridge_T
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

DATASETS=/cs/labs/mornitzan/yam.arieli/datasets
RES=results/sheu_cascade
mkdir -p "$RES"

declare -A PTDIR=( [1hr]=Sheu2024_1hr_pseudotubes [3hr]=Sheu2024_pseudotubes [5hr]=Sheu2024_5hr_pseudotubes )
declare -A BUILDSLURM=( [1hr]=slurm/build_pseudotubes_sheu_1hr.slurm \
                        [3hr]=slurm/build_pseudotubes_sheu2024.slurm \
                        [5hr]=slurm/build_pseudotubes_sheu_5hr.slurm )

BRIDGE_JOBS=()
for TP in 1hr 3hr 5hr; do
    MAN=$DATASETS/${PTDIR[$TP]}/manifest.json
    HVG=$DATASETS/${PTDIR[$TP]}/hvg_list.json
    mkdir -p "$RES/$TP"

    BUILD_DEP=""
    if [ ! -f "$MAN" ]; then
        BJ=$(sbatch --parsable "${BUILDSLURM[$TP]}")
        BUILD_DEP="--dependency=afterok:$BJ"
        echo "TP=$TP  build=$BJ  (manifest missing -> building $MAN)"
    else
        echo "TP=$TP  manifest exists -> skip build"
    fi

    EXP="ALL,TP=$TP,MAN=$MAN,HVG=$HVG,RES=$RES"

    # Path A: stage12 (3-seed array) -> geometry + §21 gate
    S12=$(sbatch --parsable $BUILD_DEP --export="$EXP" slurm/sheu_cascade/stage12.slurm)
    GEO=$(sbatch --parsable --dependency=afterok:$S12 --export="$EXP" slurm/sheu_cascade/geo_gate.slurm)

    # Bridge: binary AB-MIL -> IG -> Path B (cross_asym) -> eval
    BIN=$(sbatch --parsable $BUILD_DEP --export="$EXP" slurm/sheu_cascade/binary.slurm)
    BRDG=$(sbatch --parsable --dependency=afterok:$BIN --export="$EXP" slurm/sheu_cascade/bridge.slurm)
    BRIDGE_JOBS+=("$BRDG")

    echo "TP=$TP  s12=$S12  geo=$GEO  binary=$BIN  bridge=$BRDG"
done

DEP=$(IFS=:; echo "${BRIDGE_JOBS[*]}")
CMP=$(sbatch --parsable --dependency=afterok:$DEP --export="ALL,RES=$RES" slurm/sheu_cascade/compare.slurm)
echo "compare=$CMP  (depends on bridges: $DEP)"
echo ""
echo "DAG submitted. Bottom line will land at:"
echo "  reports/sheu_cascade/timepoint_comparison.md"
echo "Monitor: squeue -u yam.arieli"
