#!/bin/bash
# Submit the Oesinghaus 90-cytokine PURE DAG with SLURM dependencies so it runs unattended.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/oes90_pure/submit_oes90_pure_dag.sh'"
#
# Dry run (prints the sbatch commands instead of submitting; synthetic job ids keep the
# --dependency chain readable):
#   SUBMIT=echo bash slurm/oes90_pure/submit_oes90_pure_dag.sh
#
# This run is CYTOKINE-AGNOSTIC by construction: no stage reads the audited pair list or
# the published 24-cytokine panel, every cytokine carries equal weight at every stage, and
# there is deliberately NO analysis stage — scoring waits for a committed pre-registration
# (CLAUDE.md §37).
#
# DAG:
#   prepare  (CPU 64G)    verify the reused §36 tube shards, fix the main/reserve split,
#                         build the equal-weight Stage-1 cell set
#     -> encoder  (GPU)   the ONE shared 2x-wide encoder, early-stopped + sha256  [§27.6]
#       -> encode (GPU)   encode every MAIN tube once and persist the cache
#         -> train  (GPU array 0-8%3)  90 binary AB-MIL, 10 per task; heads + histories
#           -> ig   (GPU array 0-8%3)  IG twice: MAIN signatures + RESERVE stability check
#             -> merge (CPU)   merge signatures, verify provenance, signature stability
#               -> coupling  (CPU 110G) donor-level degree-corrected coupling, 4005 + BH
#                 -> direction (CPU 80G) cross_asym 4005 pairs + per-cell-type engagement
#
# Plus two health layers:
#   * one afternotok sentinel per stage -> results/oes90_pure/STATUS.md + FAILED_<stage>
#   * a self-resubmitting watchdog every 30 min -> results/oes90_pure/HEALTH.md
#
# Bottom line: results/oes90_pure/{signatures_main.parquet, coupling_donor_degree.csv,
#              direction_table.csv, engagement_per_celltype.parquet} + the saved
#              encoder, binary model heads, and training histories.
set -e
# REPO override exists so the SUBMIT=echo dry run can be exercised anywhere.
REPO=${REPO:-/cs/labs/mornitzan/yam.arieli/cytokine-mil}
cd "$REPO"
OUT=results/oes90_pure
mkdir -p "$OUT"

SUBMIT=${SUBMIT:-sbatch}
DRYRUN=0
[ "$SUBMIT" != "sbatch" ] && DRYRUN=1
# Counter lives in a file: submit() runs inside $( ), i.e. a subshell, so a plain
# variable increment would not survive back to the parent.
FAKE_ID_FILE=$(mktemp)
echo 9100000 > "$FAKE_ID_FILE"
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
for f in scripts/prepare_oes90_pure.py scripts/train_oes90_pure_encoder.py \
         scripts/encode_oes90_pure_tubes.py scripts/train_oes90_pure_chunk.py \
         scripts/ig_oes90_pure.py scripts/merge_oes90_pure_signatures.py \
         scripts/run_oes90_pure_coupling.py scripts/run_oes90_pure_direction.py; do
    [ -f "$f" ] || { echo "FATAL: $f not found — is the cluster clone up to date?" >&2; exit 2; }
done
# The tube shards are REUSED, not rebuilt. If they are gone the whole DAG is invalid.
SHARDS=results/oes_full90/tubes/meta.json
if [ "$DRYRUN" != "1" ] && [ ! -f "$SHARDS" ]; then
    echo "FATAL: $SHARDS not found — this run reuses the §36 tube shards." >&2
    exit 2
fi
if [ -f "$OUT/DONE_direction" ]; then
    echo "WARN: $OUT/DONE_direction exists — a previous run already completed here."
    echo "      Move results/oes90_pure aside first if you want a clean re-run."
fi
rm -f "$OUT"/FAILED_* 2>/dev/null || true

# sentinel <stage> <jobid> : runs only if that job fails/cancels
sentinel () {
    submit --dependency=afternotok:"$2" --kill-on-invalid-dep=yes \
        --export=ALL,STAGE="$1",WATCH_JOB="$2" \
        slurm/oes90_pure/sentinel.slurm
}

PREP=$(submit slurm/oes90_pure/prepare.slurm)
S_PREP=$(sentinel prepare "$PREP")

ENC=$(submit --dependency=afterok:"$PREP" slurm/oes90_pure/encoder.slurm)
S_ENC=$(sentinel encoder "$ENC")

EMB=$(submit --dependency=afterok:"$ENC" slurm/oes90_pure/encode.slurm)
S_EMB=$(sentinel encode "$EMB")

TRAIN=$(submit --dependency=afterok:"$EMB" slurm/oes90_pure/train.slurm)
S_TRAIN=$(sentinel train "$TRAIN")

IG=$(submit --dependency=afterok:"$TRAIN" slurm/oes90_pure/ig.slurm)
S_IG=$(sentinel ig "$IG")

MERGE=$(submit --dependency=afterok:"$IG" slurm/oes90_pure/merge.slurm)
S_MERGE=$(sentinel merge "$MERGE")

COUP=$(submit --dependency=afterok:"$MERGE" slurm/oes90_pure/coupling.slurm)
S_COUP=$(sentinel coupling "$COUP")

DIR=$(submit --dependency=afterok:"$COUP" slurm/oes90_pure/direction.slurm)
S_DIR=$(sentinel direction "$DIR")

WATCH=$(submit slurm/oes90_pure/watchdog.slurm)

[ "$DRYRUN" = "1" ] || cat > "$OUT/dag_jobs.env" <<ENVEOF
PREPARE=$PREP
ENCODER=$ENC
ENCODE=$EMB
TRAIN=$TRAIN
IG=$IG
MERGE=$MERGE
COUPLING=$COUP
DIRECTION=$DIR
WATCHDOG=$WATCH
SUBMITTED_AT=$NOW
ENVEOF

echo ""
if [ "$DRYRUN" = "1" ]; then
  echo "Would submit the Oesinghaus-90 PURE DAG (synthetic ids above):"
else
  echo "Submitted Oesinghaus-90 PURE DAG:"
fi
echo "  prepare   = $PREP    (CPU 64G:  verify shards, tube split, stage1 cells)"
echo "  encoder   = $ENC    (GPU:       ONE shared 2x-wide encoder, early-stopped)"
echo "  encode    = $EMB    (GPU:       persist the encoded pseudo-tubes)"
echo "  train     = $TRAIN    (GPU array 0-8%3: 90 binary models, heads + histories)"
echo "  ig        = $IG    (GPU array 0-8%3: top-100 signatures, main + reserve)"
echo "  merge     = $MERGE    (CPU:       merge, verify provenance, stability)"
echo "  coupling  = $COUP    (CPU 110G:  donor-level degree-corrected, 4005 pairs + BH)"
echo "  direction = $DIR    (CPU 80G:   cross_asym 4005 pairs + engagement)"
echo "  watchdog  = $WATCH    (CPU 1G:    HEALTH.md every 30 min, self-resubmitting)"
echo "  sentinels = $S_PREP $S_ENC $S_EMB $S_TRAIN $S_IG $S_MERGE $S_COUP $S_DIR  (afternotok)"
echo ""
echo "No analysis stage by design — this run produces artifacts only; scoring waits for a"
echo "committed pre-registration (CLAUDE.md §37)."
echo ""
echo "Artifacts will land in results/oes90_pure/:"
echo "  encoder.pt + encoder_history.csv   models/*_head.pt + history/*_train.csv"
echo "  signatures_main.parquet + signatures_reserve.parquet + signature_stability.csv"
echo "  coupling_donor_degree.csv   direction_table.csv   engagement_per_celltype.parquet"
echo "Monitor: cat results/oes90_pure/HEALTH.md ; cat results/oes90_pure/STATUS.md"
