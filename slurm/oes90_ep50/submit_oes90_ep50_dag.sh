#!/bin/bash
# Submit the EPOCH-50 re-run of the Oesinghaus 90-cytokine PURE fit.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/oes90_ep50/submit_oes90_ep50_dag.sh'"
#
# Dry run (prints the sbatch commands instead of submitting; synthetic job ids keep the
# --dependency chain readable):
#   SUBMIT=echo bash slurm/oes90_ep50/submit_oes90_ep50_dag.sh
#
# The ONE change vs results/oes90_pure: Stage-2 binary training stops at epoch 50
# instead of 250. In the 250-epoch fit the correlation between a binary model's training
# loss and the validity of the signature it yields INVERTS at epoch ~51 -- before it,
# conditions with a real response train faster; after it, conditions with NO real response
# descend past them by fitting tube-specific noise, and IG (run at 250) then explains that
# noise. 36 of the 90 signatures ended up scoring net-negative in their own cells.
#
# Everything upstream is SHARED, not rebuilt: the encoder, the tube split and the
# persisted embedding cache are symlinked from results/oes90_pure and digest-verified, so
# training length is the only variable. Constant-LR SGD with no scheduler means this is
# the first 50 epochs of that run reproduced exactly.
#
# Still cytokine-agnostic: no stage reads the audited pair list or the published
# 24-cytokine panel, and there is no analysis stage -- artifacts only.
#
# DAG:
#   setup   (CPU 32G)   symlink + verify encoder / tubes / embeddings from the PURE run
#     -> train (GPU array 0-8%3)  90 binary AB-MIL, 10 per task, STOPPED AT EPOCH 50
#       -> ig  (GPU array 0-8%3)  IG twice: MAIN signatures + RESERVE stability check
#         -> merge (CPU)   merge signatures, verify provenance, signature stability
#           -> coupling  (CPU 110G) donor-level degree-corrected coupling, 4005 + BH
#             -> direction (CPU 80G) cross_asym 4005 pairs + per-cell-type engagement
#
# Plus the same two health layers as the parent DAG:
#   * one afternotok sentinel per stage -> results/oes90_pure_ep50/STATUS.md + FAILED_<stage>
#   * a self-resubmitting watchdog every 30 min -> results/oes90_pure_ep50/HEALTH.md
#
# Bottom line: results/oes90_pure_ep50/{signatures_main.parquet, coupling_donor_degree.csv,
#              direction_table.csv, engagement_per_celltype.parquet}, directly comparable
#              to results/oes90_pure/ because only the epoch count differs.
set -e
# REPO override exists so the SUBMIT=echo dry run can be exercised anywhere.
REPO=${REPO:-/cs/labs/mornitzan/yam.arieli/cytokine-mil}
cd "$REPO"
OUT=results/oes90_pure_ep50
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
for f in scripts/prepare_oes90_pure_ep50.py scripts/train_oes90_pure_chunk.py \
         scripts/ig_oes90_pure.py scripts/merge_oes90_pure_signatures.py \
         scripts/run_oes90_pure_coupling.py scripts/run_oes90_pure_direction.py; do
    [ -f "$f" ] || { echo "FATAL: $f not found — is the cluster clone up to date?" >&2; exit 2; }
done
# The tube shards are REUSED, not rebuilt. If they are gone the whole DAG is invalid.
SHARDS=results/oes_full90/tubes/meta.json
PARENT=results/oes90_pure
if [ "$DRYRUN" != "1" ]; then
    [ -f "$SHARDS" ] || { echo "FATAL: $SHARDS not found — this run reuses the §36 tube shards." >&2; exit 2; }
    for f in "$PARENT/encoder.pt" "$PARENT/embeddings_sha256.txt" "$PARENT/tube_split.json" "$PARENT/DONE_encode"; do
        [ -e "$f" ] || { echo "FATAL: $f not found — the epoch-50 fit shares the PURE run's encoder and embeddings." >&2; exit 2; }
    done
fi
if [ -f "$OUT/DONE_direction" ]; then
    echo "WARN: $OUT/DONE_direction exists — a previous run already completed here."
    echo "      Move results/oes90_pure_ep50 aside first if you want a clean re-run."
fi
rm -f "$OUT"/FAILED_* 2>/dev/null || true

# sentinel <stage> <jobid> : runs only if that job fails/cancels
sentinel () {
    submit --dependency=afternotok:"$2" --kill-on-invalid-dep=yes \
        --export=ALL,STAGE="$1",WATCH_JOB="$2" \
        slurm/oes90_ep50/sentinel.slurm
}

PREP=$(submit slurm/oes90_ep50/setup.slurm)
S_PREP=$(sentinel setup "$PREP")

TRAIN=$(submit --dependency=afterok:"$PREP" slurm/oes90_ep50/train.slurm)
S_TRAIN=$(sentinel train "$TRAIN")

IG=$(submit --dependency=afterok:"$TRAIN" slurm/oes90_ep50/ig.slurm)
S_IG=$(sentinel ig "$IG")

MERGE=$(submit --dependency=afterok:"$IG" slurm/oes90_ep50/merge.slurm)
S_MERGE=$(sentinel merge "$MERGE")

COUP=$(submit --dependency=afterok:"$MERGE" slurm/oes90_ep50/coupling.slurm)
S_COUP=$(sentinel coupling "$COUP")

DIR=$(submit --dependency=afterok:"$COUP" slurm/oes90_ep50/direction.slurm)
S_DIR=$(sentinel direction "$DIR")

WATCH=$(submit slurm/oes90_ep50/watchdog.slurm)

[ "$DRYRUN" = "1" ] || cat > "$OUT/dag_jobs.env" <<ENVEOF
SETUP=$PREP
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
  echo "Would submit the Oesinghaus-90 epoch-50 re-run DAG (synthetic ids above):"
else
  echo "Submitted Oesinghaus-90 epoch-50 re-run DAG:"
fi
echo "  setup     = $PREP    (CPU 32G:  share + verify encoder / tubes / embeddings)"
echo "  train     = $TRAIN    (GPU array 0-8%3: 90 binary models, STOPPED AT EPOCH 50)"
echo "  ig        = $IG    (GPU array 0-8%3: top-100 signatures, main + reserve)"
echo "  merge     = $MERGE    (CPU:       merge, verify provenance, stability)"
echo "  coupling  = $COUP    (CPU 110G:  donor-level degree-corrected, 4005 pairs + BH)"
echo "  direction = $DIR    (CPU 80G:   cross_asym 4005 pairs + engagement)"
echo "  watchdog  = $WATCH    (CPU 1G:    HEALTH.md every 30 min, self-resubmitting)"
echo "  sentinels = $S_PREP $S_TRAIN $S_IG $S_MERGE $S_COUP $S_DIR  (afternotok)"
echo ""
echo "No analysis stage by design — artifacts only, directly comparable to results/oes90_pure"
echo "because Stage-2 epoch count is the only difference."
echo ""
echo "Artifacts will land in results/oes90_pure_ep50/:"
echo "  models/*_head.pt + history/*_train.csv   (encoder is shared, not rebuilt)"
echo "  signatures_main.parquet + signatures_reserve.parquet + signature_stability.csv"
echo "  coupling_donor_degree.csv   direction_table.csv   engagement_per_celltype.parquet"
echo "Monitor: cat results/oes90_pure_ep50/HEALTH.md ; cat results/oes90_pure_ep50/STATUS.md"
