#!/bin/bash
# Submit the COVID-progression DAG with SLURM dependencies (§30).
#   apparatus (CPU, independent)
#   prepare   (CPU)            ->  fit (GPU)  ->  analysis (CPU, afterok:fit)
#
# Usage (from the cluster lab dir, via the login shell):
#   bash slurm/covid_progression/submit_covid_dag.sh
#   SUBMIT=echo bash slurm/covid_progression/submit_covid_dag.sh    # dry-run
#
# From local Mac:
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/covid_progression/submit_covid_dag.sh'"
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/covid_progression
SUBMIT=${SUBMIT:-sbatch}
S=slurm/covid_progression

APP=$($SUBMIT --parsable $S/apparatus.slurm)
PREP=$($SUBMIT --parsable $S/prepare.slurm)
FIT=$($SUBMIT --parsable --dependency=afterok:$PREP $S/fit.slurm)
ANAL=$($SUBMIT --parsable --dependency=afterok:$FIT $S/analysis.slurm)

echo ""
echo "Submitted COVID-progression DAG:"
echo "  apparatus = $APP   (independent GO/NO-GO gate)"
echo "  prepare   = $PREP"
echo "  fit       = $FIT   (afterok:prepare, GPU)"
echo "  analysis  = $ANAL  (afterok:fit -> figures + COVID_PROGRESSION_RESULTS.md)"
echo ""
echo "Monitor: squeue -u yam.arieli"
