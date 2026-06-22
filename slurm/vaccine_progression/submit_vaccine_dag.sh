#!/bin/bash
# Submit the §32 vaccine T-cell cascade DAG with SLURM dependencies.
#   apparatus (CPU, independent GO/NO-GO gate)
#   download (CPU) -> convert (CPU, R+Py) -> prepare (CPU) -> TWO branches:
#       fit_state(GPU)     -> analysis_state(CPU)       [HEADLINE: naive->eff->mem]
#       fit_timepoint(GPU) -> analysis_timepoint(CPU)   [corroboration: D0<D2<D10<D28]
#
# STAGE control (so the obs schema can be inspected before the GPU fit):
#   STAGE=front : apparatus + download -> convert  (stop; inspect prepare --inspect_only)
#   STAGE=back  : prepare -> {fit_state->analysis_state, fit_timepoint->analysis_timepoint}
#                 (assumes convert already produced vaccine_cite_raw.h5ad)
#   STAGE=all   : the full chain (default)
#
# Usage (from the cluster lab dir, via the login shell):
#   STAGE=front bash slurm/vaccine_progression/submit_vaccine_dag.sh
#   SUBMIT=echo bash slurm/vaccine_progression/submit_vaccine_dag.sh    # dry-run
#
# From local Mac:
#   cluster_cmd "bash -l -c 'cd cytokine-mil && STAGE=front bash slurm/vaccine_progression/submit_vaccine_dag.sh'"
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/vaccine_progression
SUBMIT=${SUBMIT:-sbatch}
STAGE=${STAGE:-all}
S=slurm/vaccine_progression

submit_back() {
    PREP=$($SUBMIT --parsable $S/prepare.slurm)
    FIT_S=$($SUBMIT --parsable --dependency=afterok:$PREP $S/fit_state.slurm)
    ANA_S=$($SUBMIT --parsable --dependency=afterok:$FIT_S $S/analysis_state.slurm)
    FIT_T=$($SUBMIT --parsable --dependency=afterok:$PREP $S/fit_timepoint.slurm)
    ANA_T=$($SUBMIT --parsable --dependency=afterok:$FIT_T $S/analysis_timepoint.slurm)
    echo "  prepare         = $PREP"
    echo "  fit_state       = $FIT_S  (afterok:prepare, GPU)   [HEADLINE]"
    echo "  analysis_state  = $ANA_S  (afterok:fit_state -> VACCINE_PROGRESSION_RESULTS.md)"
    echo "  fit_timepoint   = $FIT_T  (afterok:prepare, GPU)   [corroboration]"
    echo "  analysis_timep. = $ANA_T  (afterok:fit_timepoint -> *_timepoint.md)"
}

echo "Submitted vaccine T-cell cascade DAG (STAGE=$STAGE):"
if [ "$STAGE" = "front" ] || [ "$STAGE" = "all" ]; then
    APP=$($SUBMIT --parsable $S/apparatus.slurm)
    DL=$($SUBMIT --parsable $S/download.slurm)
    CONV=$($SUBMIT --parsable --dependency=afterok:$DL $S/convert.slurm)
    echo "  apparatus       = $APP    (independent GO/NO-GO gate)"
    echo "  download        = $DL"
    echo "  convert         = $CONV   (afterok:download; R .rds -> AnnData)"
fi
if [ "$STAGE" = "all" ]; then
    PREP=$($SUBMIT --parsable --dependency=afterok:$CONV $S/prepare.slurm)
    FIT_S=$($SUBMIT --parsable --dependency=afterok:$PREP $S/fit_state.slurm)
    ANA_S=$($SUBMIT --parsable --dependency=afterok:$FIT_S $S/analysis_state.slurm)
    FIT_T=$($SUBMIT --parsable --dependency=afterok:$PREP $S/fit_timepoint.slurm)
    ANA_T=$($SUBMIT --parsable --dependency=afterok:$FIT_T $S/analysis_timepoint.slurm)
    echo "  prepare         = $PREP   (afterok:convert)"
    echo "  fit_state       = $FIT_S  (afterok:prepare, GPU)   [HEADLINE]"
    echo "  analysis_state  = $ANA_S  (afterok:fit_state -> VACCINE_PROGRESSION_RESULTS.md)"
    echo "  fit_timepoint   = $FIT_T  (afterok:prepare, GPU)   [corroboration]"
    echo "  analysis_timep. = $ANA_T  (afterok:fit_timepoint -> *_timepoint.md)"
fi
if [ "$STAGE" = "back" ]; then
    submit_back
fi
echo ""
echo "Monitor: squeue -u yam.arieli"
