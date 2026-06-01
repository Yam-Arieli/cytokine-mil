#!/bin/bash
# Submit the 4 downstream ID-cascade jobs chained onto an already-running build
# job.  Used when submit_id_cascade_dag.sh only got as far as the build sbatch
# (e.g., the gateway cluster_cmd window closed mid-script after the slow
# barcode pre-check).  Fast: 4 sbatch calls, no pre-check.
#
# Usage (login shell, on the cluster):
#   bash slurm/id_cascade/submit_downstream.sh <build_jobid>
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
BUILD_JOB=${1:?usage: submit_downstream.sh <build_jobid>}

DATASETS=/cs/labs/mornitzan/yam.arieli/datasets
MAN=$DATASETS/ImmuneDictionary_pseudotubes/manifest.json
HVG=$DATASETS/ImmuneDictionary_pseudotubes/hvg_list.json
RES=results/id_cascade
EXP="ALL,MAN=$MAN,HVG=$HVG,RES=$RES"
mkdir -p "$RES"

S12=$(sbatch --parsable --dependency=afterok:$BUILD_JOB --export="$EXP" slurm/id_cascade/stage12.slurm)
echo "s12=$S12  (Path A array, 3 seeds; afterok:$BUILD_JOB)"
GEO=$(sbatch --parsable --dependency=afterok:$S12 --export="$EXP" slurm/id_cascade/geo_gate.slurm)
echo "geo=$GEO  (latent geometry; afterok:$S12)"
BIN=$(sbatch --parsable --dependency=afterok:$BUILD_JOB --export="$EXP" slurm/id_cascade/binary.slurm)
echo "binary=$BIN  (Bridge: 12 cytokines; afterok:$BUILD_JOB)"
BRDG=$(sbatch --parsable --dependency=afterok:$BIN --export="$EXP" slurm/id_cascade/bridge.slurm)
echo "bridge=$BRDG  (IG + Path B cross_asym + retally; afterok:$BIN)"
echo ""
echo "Downstream chained on build=$BUILD_JOB."
echo "Bottom line -> results/id_cascade/eval/pipeline_accuracy.md"
