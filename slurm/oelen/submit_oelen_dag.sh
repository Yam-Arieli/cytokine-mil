#!/bin/bash
# Submit the Oelen DAG (build -> binary -> coupling) with SLURM dependencies.
# Run ON the cluster (login shell):
#   cluster_cmd "bash -l -c 'bash /cs/labs/mornitzan/yam.arieli/cytokine-mil/slurm/oelen/submit_oelen_dag.sh'"
# Bottom line: results/oelen/donor_coupling/donor_coupling_report.md
#              results/oelen/coupling_cell_degree/cell_degree_report.md
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oelen2022_pseudotubes/manifest.json
mkdir -p results/oelen
SUBMIT=${SUBMIT:-sbatch}

BUILD_DEP=""
if [ ! -f "$MAN" ]; then
    BJ=$($SUBMIT --parsable slurm/oelen/build.slurm)
    BUILD_DEP="--dependency=afterok:$BJ"
    echo "build=$BJ (manifest missing)"
else
    echo "manifest exists -> skip build"
fi

BIN=$($SUBMIT --parsable $BUILD_DEP slurm/oelen/binary.slurm)
COUP=$($SUBMIT --parsable --dependency=afterok:$BIN slurm/oelen/coupling.slurm)
echo "Submitted: build=${BJ:-(skip)}  binary=$BIN  coupling=$COUP"
