#!/bin/bash
# Submit the Cano-Gamez CD4 T-cell coupling DAG (build -> binary -> bridge_coupling)
# with SLURM dependencies. Run ON the cluster (login shell):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/cano_gamez/submit_cano_dag.sh'"
#
# Bottom line: results/cano_gamez/coupling_cell_degree/cell_degree_report.md
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
MAN=/cs/labs/mornitzan/yam.arieli/datasets/CanoGamez_pseudotubes/manifest.json
RES=results/cano_gamez
mkdir -p "$RES"
SUBMIT=${SUBMIT:-sbatch}

BUILD_DEP=""
if [ ! -f "$MAN" ]; then
    BJ=$($SUBMIT --parsable slurm/cano_gamez/build.slurm)
    BUILD_DEP="--dependency=afterok:$BJ"
    echo "build=$BJ (manifest missing)"
else
    echo "manifest exists -> skip build"
fi

BIN=$($SUBMIT --parsable $BUILD_DEP slurm/cano_gamez/binary.slurm)
BRDG=$($SUBMIT --parsable --dependency=afterok:$BIN slurm/cano_gamez/bridge_coupling.slurm)

echo "Submitted: build=${BJ:-(skip)}  binary=$BIN  bridge=$BRDG"
echo "Bottom line: results/cano_gamez/coupling_cell_degree/cell_degree_report.md"
