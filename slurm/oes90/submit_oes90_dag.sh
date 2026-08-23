#!/bin/bash
# Submit the full Oesinghaus-90 DAG with SLURM dependencies so it runs unattended.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/oes90/submit_oes90_dag.sh'"
#
# Dry run (prints the sbatch commands instead of submitting; synthetic job ids keep the
# --dependency chain readable):
#   SUBMIT=echo bash slurm/oes90/submit_oes90_dag.sh
#
# DAG:
#   prepare  (CPU 64G)   tubes as (donor,condition) shards + unique-cell Stage-1 AnnData
#     -> encoder  (GPU)  the ONE shared Stage-1 encoder + sha256      [§27.6 guard]
#       -> train  (GPU array 0-8%3)  90 binary AB-MIL + IG, 10 per task
#         -> merge (CPU) merge signatures, verify encoder/tube provenance
#           -> coupling  (CPU 260G) donor-level degree-corrected coupling, 4005 pairs + BH
#             -> direction (CPU 180G) cross_asym for 4005 pairs + audited regression check
#               -> analysis (CPU)  neutral-background enrichment/over-call + RESULTS.md
#
# Plus two health layers:
#   * one afternotok sentinel per stage -> results/oes_full90/STATUS.md + FAILED_<stage>
#   * a self-resubmitting watchdog every 30 min -> results/oes_full90/HEALTH.md
#
# Bottom line: reports/oesinghaus_full90/RESULTS.md
#              results/oes_full90/per_pair_summary.csv
set -e
# REPO override exists so the SUBMIT=echo dry run can be exercised anywhere.
REPO=${REPO:-/cs/labs/mornitzan/yam.arieli/cytokine-mil}
cd "$REPO"
OUT=results/oes_full90
mkdir -p "$OUT"

SUBMIT=${SUBMIT:-sbatch}
DRYRUN=0
[ "$SUBMIT" != "sbatch" ] && DRYRUN=1
# Counter lives in a file: submit() runs inside $( ), i.e. a subshell, so a plain
# variable increment would not survive back to the parent.
FAKE_ID_FILE=$(mktemp)
echo 9000000 > "$FAKE_ID_FILE"
trap 'rm -f "$FAKE_ID_FILE"' EXIT

# submit <sbatch args...> -> prints the job id on stdout.
# In dry-run mode it echoes the command to stderr and returns a synthetic id, so the
# --dependency chain below stays readable instead of nesting whole commands inside itself.
submit () {
    if [ "$DRYRUN" = "1" ]; then
        local id=$(( $(cat "$FAKE_ID_FILE") + 1 ))
        echo "$id" > "$FAKE_ID_FILE"
        echo "  sbatch --parsable $*   ->  $id" >&2
        echo "$id"
    else
        sbatch --parsable "$@"
    fi
}

NOW=$(date -Is 2>/dev/null || date)
[ "$DRYRUN" = "1" ] && echo "DRY RUN (SUBMIT=$SUBMIT) — nothing will be queued:" >&2

# Guards: fail early rather than after a queued stage starts.
for f in scripts/prepare_oesinghaus_full90.py scripts/analyze_oesinghaus_full90.py \
         reports/cascade_pairs/cytokine_axes_audited.csv; do
    [ -f "$f" ] || { echo "FATAL: $f not found — is the cluster clone up to date?" >&2; exit 2; }
done
if [ -f "$OUT/DONE_analysis" ]; then
    echo "WARN: $OUT/DONE_analysis exists — a previous run already completed here."
    echo "      Move results/oes_full90 aside first if you want a clean re-run."
fi
rm -f "$OUT"/FAILED_* 2>/dev/null || true

# sentinel <stage> <jobid> : runs only if that job fails/cancels
sentinel () {
    submit --dependency=afternotok:"$2" --kill-on-invalid-dep=yes \
        --export=ALL,STAGE="$1",WATCH_JOB="$2" \
        slurm/oes90/sentinel.slurm
}

PREP=$(submit slurm/oes90/prepare.slurm)
S_PREP=$(sentinel prepare "$PREP")

ENC=$(submit --dependency=afterok:"$PREP" slurm/oes90/encoder.slurm)
S_ENC=$(sentinel encoder "$ENC")

TRAIN=$(submit --dependency=afterok:"$ENC" slurm/oes90/train.slurm)
S_TRAIN=$(sentinel train "$TRAIN")

MERGE=$(submit --dependency=afterok:"$TRAIN" slurm/oes90/merge.slurm)
S_MERGE=$(sentinel merge "$MERGE")

COUP=$(submit --dependency=afterok:"$MERGE" slurm/oes90/coupling.slurm)
S_COUP=$(sentinel coupling "$COUP")

DIR=$(submit --dependency=afterok:"$COUP" slurm/oes90/direction.slurm)
S_DIR=$(sentinel direction "$DIR")

ANA=$(submit --dependency=afterok:"$DIR" slurm/oes90/analysis.slurm)
S_ANA=$(sentinel analysis "$ANA")

WATCH=$(submit slurm/oes90/watchdog.slurm)

[ "$DRYRUN" = "1" ] || cat > "$OUT/dag_jobs.env" <<ENVEOF
PREPARE=$PREP
ENCODER=$ENC
TRAIN=$TRAIN
MERGE=$MERGE
COUPLING=$COUP
DIRECTION=$DIR
ANALYSIS=$ANA
WATCHDOG=$WATCH
SUBMITTED_AT=$NOW
ENVEOF

echo ""
if [ "$DRYRUN" = "1" ]; then
  echo "Would submit the Oesinghaus-90 DAG (synthetic ids above):"
else
  echo "Submitted Oesinghaus-90 DAG:"
fi
echo "  prepare   = $PREP    (CPU 64G:  tube shards + stage1 cells)"
echo "  encoder   = $ENC    (GPU:       ONE shared Stage-1 encoder + sha256)"
echo "  train     = $TRAIN    (GPU array 0-8%3: 90 binary models + IG)"
echo "  merge     = $MERGE    (CPU:       merge signatures, verify provenance)"
echo "  coupling  = $COUP    (CPU 260G:  donor-level degree-corrected, 4005 pairs + BH)"
echo "  direction = $DIR    (CPU 180G:  cross_asym 4005 pairs + audited regression)"
echo "  analysis  = $ANA    (CPU:       neutral-background scoring + RESULTS.md)"
echo "  watchdog  = $WATCH    (CPU 1G:    HEALTH.md every 30 min, self-resubmitting)"
echo "  sentinels = $S_PREP $S_ENC $S_TRAIN $S_MERGE $S_COUP $S_DIR $S_ANA  (afternotok)"
echo ""
echo "Bottom line will land at:"
echo "  reports/oesinghaus_full90/RESULTS.md"
echo "  results/oes_full90/per_pair_summary.csv"
echo "Monitor: cat results/oes_full90/HEALTH.md ; cat results/oes_full90/STATUS.md"
