#!/bin/bash
# Re-run only the bridge (IG -> Path B -> eval) + compare steps. The GPU
# training (stage12 + binary) already completed for all time points, so we
# reuse those outputs. Fixes the original empty-TARGETS bridge failure.
#
#   cluster_cmd "bash -l -c 'cd cytokine-mil && git pull && bash slurm/sheu_cascade/resubmit_bridge.sh'"
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
DATASETS=/cs/labs/mornitzan/yam.arieli/datasets
RES=results/sheu_cascade
declare -A PTDIR=( [1hr]=Sheu2024_1hr_pseudotubes [3hr]=Sheu2024_pseudotubes [5hr]=Sheu2024_5hr_pseudotubes )

BRIDGE_JOBS=()
for TP in 1hr 3hr 5hr; do
    MAN=$DATASETS/${PTDIR[$TP]}/manifest.json
    HVG=$DATASETS/${PTDIR[$TP]}/hvg_list.json
    EXP="ALL,TP=$TP,MAN=$MAN,HVG=$HVG,RES=$RES"
    BRDG=$(sbatch --parsable --export="$EXP" slurm/sheu_cascade/bridge.slurm)
    BRIDGE_JOBS+=("$BRDG")
    echo "TP=$TP  bridge=$BRDG"
done

DEP=$(IFS=:; echo "${BRIDGE_JOBS[*]}")
CMP=$(sbatch --parsable --dependency=afterok:$DEP --export="ALL,RES=$RES" slurm/sheu_cascade/compare.slurm)
echo "compare=$CMP  (after bridges: $DEP)"
echo "Bottom line -> reports/sheu_cascade/timepoint_comparison.md"
