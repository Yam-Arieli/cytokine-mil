#!/bin/bash
# Submit the §40 Oesinghaus-90 dropout + curation DAG (CLAUDE.md §40).
#
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/oes90_dc/submit_oes90_dc_dag.sh'"
#
# Dry run (prints the chain with synthetic job ids, queues nothing):
#
#   SUBMIT=echo bash slurm/oes90_dc/submit_oes90_dc_dag.sh
#
# AGNOSTIC BY CONSTRUCTION: no stage reads the audited pair list or the published
# 24-cytokine panel. `_oes90_dc_config.py` imports `_oes90_pure_config` for plumbing only
# (that module deliberately does not define AUDITED_CSV / PUBLISHED_COUPLING_CSV), every
# stage calls `assert_agnostic()`, and `scripts/run_demo_oes90_dc.py` additionally walks
# each stage's AST for benchmark references with a positive control.
#
# THE DAG
#
#   prepare -> encoder -> encode -> train[0-8%3] -> ig[0-8%3] -> merge
#                                                                  |
#                          +---------------------------------------+
#                          |                                       |
#                  coupling(ARM=curated)                   coupling(ARM=raw)
#                          |                                       |
#                  direction[0-3](curated)                 direction[0-3](raw)
#                          |                                       |
#                    dirmerge(curated)                       dirmerge(raw)
#
# The two arms are independent branches off merge, so they overlap in wall-clock; only
# CPU-hours double. "curated" is §40's result; "raw" is the uncurated top-200 control,
# without which there is no way to say whether the curation helped, hurt, or did nothing.
# The arms may cover DIFFERENT condition sets (curation drops conditions it empties), so
# any comparison between them must be made on the intersection.
#
# Stage 0 is §37's `prepare_oes90_pure.py` run VERBATIM into this run's out_dir, so the
# tube split and Stage-1 cell set are provably identical to §37's.
#
# TWO HEALTH LAYERS
#   (a) a sentinel per stage on --dependency=afternotok, appending to STATUS.md and
#       dropping FAILED_<stage>;
#   (b) a self-resubmitting watchdog appending queue + progress to HEALTH.md every 30 min,
#       which exits when both arms finish or any stage fails.
#
# NO ANALYSIS STAGE, BY DESIGN. Like §37 this produces artifacts only; scoring against
# any benchmark waits for a committed pre-registration (CLAUDE.md §25.1).
set -e
# REPO override exists so the SUBMIT=echo dry run can be exercised anywhere.
REPO=${REPO:-/cs/labs/mornitzan/yam.arieli/cytokine-mil}
cd "$REPO"
OUT=results/oes90_dc
mkdir -p "$OUT"

SUBMIT=${SUBMIT:-sbatch}
DRYRUN=0
[ "$SUBMIT" != "sbatch" ] && DRYRUN=1
# Counter lives in a file: submit() runs inside $( ), i.e. a subshell, so a plain
# variable increment would not survive back to the parent.
FAKE_ID_FILE=$(mktemp)
echo 9400000 > "$FAKE_ID_FILE"
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
for f in scripts/prepare_oes90_pure.py scripts/train_oes90_dc_encoder.py \
         scripts/encode_oes90_dc_tubes.py scripts/train_oes90_dc_chunk.py \
         scripts/ig_oes90_dc.py scripts/merge_oes90_dc_signatures.py \
         scripts/run_oes90_dc_coupling.py scripts/run_oes90_dc_direction.py \
         scripts/merge_oes90_dc_direction.py; do
    [ -f "$f" ] || { echo "FATAL: $f not found — is the cluster clone up to date?" >&2; exit 2; }
done
# The tube shards are REUSED, not rebuilt. If they are gone the whole DAG is invalid.
SHARDS=results/oes_full90/tubes/meta.json
if [ "$DRYRUN" != "1" ] && [ ! -f "$SHARDS" ]; then
    echo "FATAL: $SHARDS not found — this run reuses the §36 tube shards." >&2
    exit 2
fi
if [ -f "$OUT/DONE_direction_curated" ] && [ -f "$OUT/DONE_direction_raw" ]; then
    echo "WARN: both DONE_direction_* markers exist — a previous run already completed here."
    echo "      Move results/oes90_dc aside first if you want a clean re-run."
fi
rm -f "$OUT"/FAILED_* 2>/dev/null || true

# sentinel <stage> <jobid> : runs only if that job fails/cancels
sentinel () {
    submit --dependency=afternotok:"$2" --kill-on-invalid-dep=yes \
        --export=ALL,STAGE="$1",WATCH_JOB="$2" \
        slurm/oes90_dc/sentinel.slurm
}

PREP=$(submit slurm/oes90_dc/prepare.slurm)
S_PREP=$(sentinel prepare "$PREP")

ENC=$(submit --dependency=afterok:"$PREP" slurm/oes90_dc/encoder.slurm)
S_ENC=$(sentinel encoder "$ENC")

EMB=$(submit --dependency=afterok:"$ENC" slurm/oes90_dc/encode.slurm)
S_EMB=$(sentinel encode "$EMB")

TRAIN=$(submit --dependency=afterok:"$EMB" slurm/oes90_dc/train.slurm)
S_TRAIN=$(sentinel train "$TRAIN")

IG=$(submit --dependency=afterok:"$TRAIN" slurm/oes90_dc/ig.slurm)
S_IG=$(sentinel ig "$IG")

MERGE=$(submit --dependency=afterok:"$IG" slurm/oes90_dc/merge.slurm)
S_MERGE=$(sentinel merge "$MERGE")

# --- the two signature arms, as independent branches off merge ---------------
for ARM in curated raw; do
    COUP=$(submit --dependency=afterok:"$MERGE" --export=ALL,ARM="$ARM" \
                  slurm/oes90_dc/coupling.slurm)
    sentinel "coupling_$ARM" "$COUP" > /dev/null

    DIR=$(submit --dependency=afterok:"$COUP" --export=ALL,ARM="$ARM" \
                 slurm/oes90_dc/direction.slurm)
    sentinel "direction_$ARM" "$DIR" > /dev/null

    DMRG=$(submit --dependency=afterok:"$DIR" --export=ALL,ARM="$ARM" \
                  slurm/oes90_dc/dirmerge.slurm)
    sentinel "dirmerge_$ARM" "$DMRG" > /dev/null

    if [ "$ARM" = "curated" ]; then
        COUP_CUR=$COUP; DIR_CUR=$DIR; DMRG_CUR=$DMRG
    else
        COUP_RAW=$COUP; DIR_RAW=$DIR; DMRG_RAW=$DMRG
    fi
done

WATCH=$(submit slurm/oes90_dc/watchdog.slurm)

[ "$DRYRUN" = "1" ] || cat > "$OUT/dag_jobs.env" <<ENVEOF
PREPARE=$PREP
ENCODER=$ENC
ENCODE=$EMB
TRAIN=$TRAIN
IG=$IG
MERGE=$MERGE
COUPLING_CURATED=$COUP_CUR
DIRECTION_CURATED=$DIR_CUR
DIRMERGE_CURATED=$DMRG_CUR
COUPLING_RAW=$COUP_RAW
DIRECTION_RAW=$DIR_RAW
DIRMERGE_RAW=$DMRG_RAW
WATCHDOG=$WATCH
SUBMITTED_AT=$NOW
ENVEOF

cat <<SUMMARY

§40 oes90_dc DAG submitted ($NOW)

  prepare    $PREP    CPU 64G   — §37's prepare, verbatim (reuses the §36 shards)
  encoder    $ENC    GPU        — 1024-wide + 50% dropout, FIXED 20 epochs, no best-val restore
  encode     $EMB    GPU        — persist the frozen embedding cache
  train      $TRAIN    GPU 0-8%3  — 90 binary AB-MIL heads
  ig         $IG    GPU 0-8%3  — top-200 signatures, main + reserve, UNCURATED
  merge      $MERGE    CPU        — merge, then null-calibrated curation (both parquets)

  arm=curated (§40 result)          arm=raw (control)
    coupling  $COUP_CUR  110G          coupling  $COUP_RAW  110G
    direction $DIR_CUR  80G 0-3       direction $DIR_RAW  80G 0-3
    dirmerge  $DMRG_CUR              dirmerge  $DMRG_RAW

  watchdog   $WATCH    — HEALTH.md every 30 min; exits when both arms finish or one fails

Each stage also has an afternotok sentinel writing $OUT/STATUS.md + FAILED_<stage>.

NO ANALYSIS STAGE, by design — artifacts only. Scoring waits for a committed
pre-registration (CLAUDE.md §25.1).

Bottom line:
  $OUT/signatures_main.parquet            (top-200, uncurated)
  $OUT/signatures_main_curated.parquet    + curation_report.csv + curation_meta.json
  $OUT/coupling_donor_degree_{curated,raw}.csv
  $OUT/direction_table_{curated,raw}.csv
  $OUT/engagement_per_celltype_{curated,raw}.parquet

The cheapest first read, available as soon as merge lands: curation_meta.json's
excess_removal against its expected_null_removal.

Monitor:  cat $OUT/HEALTH.md   ;   cat $OUT/STATUS.md
SUMMARY
