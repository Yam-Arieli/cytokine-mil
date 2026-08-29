#!/bin/bash
# Re-submit ONLY the direction + dirmerge tail of the §40 DAG (CLAUDE.md §40).
#
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/oes90_dc/resubmit_direction.sh'"
#   SUBMIT=echo bash slurm/oes90_dc/resubmit_direction.sh      # dry run
#
# WHY THIS EXISTS. The first §40 run completed every stage through coupling and then lost
# both arms' direction stage to OOM (see the measurement in direction.slurm). The main
# submitter always starts at `prepare`, so re-running it would retrain the encoder and all
# 90 binary heads to reproduce artifacts that are already on disk and already digest-
# verified. This resumes from the last good stage instead.
#
# It refuses to run unless the upstream artifacts it depends on are present, so a resume
# can never silently score a half-built run.
set -e
REPO=${REPO:-/cs/labs/mornitzan/yam.arieli/cytokine-mil}
cd "$REPO"
OUT=results/oes90_dc

SUBMIT=${SUBMIT:-sbatch}
DRYRUN=0
[ "$SUBMIT" != "sbatch" ] && DRYRUN=1
FAKE_ID_FILE=$(mktemp)
echo 9500000 > "$FAKE_ID_FILE"
trap 'rm -f "$FAKE_ID_FILE"' EXIT

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

sentinel () {
    submit --dependency=afternotok:"$2" --kill-on-invalid-dep=yes \
        --export=ALL,STAGE="$1",WATCH_JOB="$2" \
        slurm/oes90_dc/sentinel.slurm
}

# Upstream must actually be finished. Direction reads the signatures and the tube shards;
# coupling is not an input to it, but a missing coupling table means the arm never ran and
# resuming past it would leave the arm half-scored.
if [ "$DRYRUN" != "1" ]; then
    for f in "$OUT/DONE_merge" "$OUT/signatures_main.parquet" \
             "$OUT/signatures_main_curated.parquet" "$OUT/curation_meta.json" \
             results/oes_full90/tubes/meta.json; do
        [ -e "$f" ] || { echo "FATAL: $f missing — upstream stages have not completed." >&2; exit 2; }
    done
    for ARM in curated raw; do
        [ -e "$OUT/DONE_coupling_$ARM" ] || {
            echo "FATAL: $OUT/DONE_coupling_$ARM missing — resume the coupling stage first." >&2
            exit 2; }
    done
fi

# Clear the previous attempt's failure markers and partial shards, so dirmerge's
# provenance check cannot stitch a stale shard onto a fresh one.
rm -f "$OUT"/FAILED_direction_* "$OUT"/FAILED_dirmerge_* 2>/dev/null || true
rm -f "$OUT"/direction_table_*_shard*.csv 2>/dev/null || true
rm -f "$OUT"/engagement_per_celltype_*_shard*.parquet 2>/dev/null || true
rm -f "$OUT"/direction_meta_*_shard*.json 2>/dev/null || true

NOW=$(date -Is 2>/dev/null || date)
[ "$DRYRUN" = "1" ] && echo "DRY RUN (SUBMIT=$SUBMIT) — nothing will be queued:" >&2

for ARM in curated raw; do
    DIR=$(submit --export=ALL,ARM="$ARM" slurm/oes90_dc/direction.slurm)
    sentinel "direction_$ARM" "$DIR" > /dev/null
    DMRG=$(submit --dependency=afterok:"$DIR" --export=ALL,ARM="$ARM" \
                  slurm/oes90_dc/dirmerge.slurm)
    sentinel "dirmerge_$ARM" "$DMRG" > /dev/null
    if [ "$ARM" = "curated" ]; then DIR_CUR=$DIR; DMRG_CUR=$DMRG
    else DIR_RAW=$DIR; DMRG_RAW=$DMRG; fi
done

WATCH=$(submit slurm/oes90_dc/watchdog.slurm)

cat <<SUMMARY

§40 oes90_dc direction tail RE-submitted ($NOW)

  arm=curated: direction $DIR_CUR (140G, 0-3%1)   dirmerge $DMRG_CUR
  arm=raw:     direction $DIR_RAW (140G, 0-3%1)   dirmerge $DMRG_RAW
  watchdog     $WATCH

Upstream (prepare -> encoder -> encode -> train -> ig -> merge -> coupling) is REUSED
from the first attempt; its digest guards still verify the encoder and tube shards on
every load, so a stale artifact cannot slip through unnoticed.

Monitor:  cat $OUT/HEALTH.md   ;   cat $OUT/STATUS.md
SUMMARY
