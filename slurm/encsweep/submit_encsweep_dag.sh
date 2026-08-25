#!/bin/bash
# Submit the Stage-1 encoder condition-breadth sweep with SLURM dependencies.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/encsweep/submit_encsweep_dag.sh'"
#
# Dry run (prints the sbatch commands instead of submitting; synthetic job ids keep the
# --dependency chain readable):
#   SUBMIT=echo bash slurm/encsweep/submit_encsweep_dag.sh
#
# WHY: on the identical 24 cytokines at the identical top-50 cut, mean between-cytokine
# signature Jaccard runs 0.065 (published anchor) -> 0.178 (§36) -> 0.241 (§37 PURE). The
# largest step is published -> §36, where the ONLY change was the Stage-1 encoder's
# training set (17-18 conditions -> 90). A binary model only ever sees {X, PBS} tubes, so
# condition count cannot reach a signature except through the encoder. Stage 1's cell-type
# objective treats cytokine response as nuisance variance WITHIN a cell type, so the more
# conditions it sees, the more it is trained to discard the signal the method needs.
#
# This sweep varies only that. Everything else is pinned at the published values: 512-wide,
# hidden (512,512), Stage-1 20 epochs with NO early stopping, k=10 tubes, top_n=50.
# Total Stage-1 cells is held fixed across arms, so breadth is not confounded with
# gradient exposure. Encoder arms are nested: rand18 c rand45 c all90.
#
# The readout is signature DIVERSITY only. No coupling, no direction, no benchmark score —
# this decides which encoder to fit with, not what the biology is.
#
# DAG:
#   prepare  (CPU 64G)          one unique-cell bank -> four budget-matched Stage-1 sets
#     -> encoder (GPU array 0-3)      one encoder per arm, sha256-guarded
#       -> train (GPU array 0-15%3)   24-cytokine panel vs PBS, 4 arms x 4 chunks
#         -> ig  (GPU array 0-15%3)   Integrated Gradients signatures, top-50
#           -> sign     (CPU array 0-3%2, SECONDARY)  frac_up per signature
#           -> analysis (CPU 16G)     merge + ARM_COMPARISON.md
#
# analysis waits on ig with afterok but on sign with afterany, so losing the secondary
# sign table never costs the primary diversity readout.
#
# Plus two health layers:
#   * one afternotok sentinel per stage -> results/encsweep/STATUS.md + FAILED_<stage>
#   * a self-resubmitting watchdog every 30 min -> results/encsweep/HEALTH.md
#
# Bottom line: results/encsweep/ARM_COMPARISON.md + arm_diversity.csv
set -e
REPO=${REPO:-/cs/labs/mornitzan/yam.arieli/cytokine-mil}
cd "$REPO"
OUT=results/encsweep
mkdir -p "$OUT"

SUBMIT=${SUBMIT:-sbatch}
DRYRUN=0
[ "$SUBMIT" != "sbatch" ] && DRYRUN=1
FAKE_ID_FILE=$(mktemp)
echo 9200000 > "$FAKE_ID_FILE"
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

NOW=$(date -Is 2>/dev/null || date)
[ "$DRYRUN" = "1" ] && echo "DRY RUN (SUBMIT=$SUBMIT) — nothing will be queued:" >&2

for f in scripts/_encsweep_config.py scripts/prepare_encsweep.py \
         scripts/train_encsweep_encoder.py scripts/train_encsweep_chunk.py \
         scripts/ig_encsweep.py scripts/analyze_encsweep.py; do
    [ -f "$f" ] || { echo "FATAL: $f not found — is the cluster clone up to date?" >&2; exit 2; }
done
# The tube shards are REUSED, not rebuilt (same shards as §36/§37).
SHARDS=results/oes_full90/tubes/meta.json
if [ "$DRYRUN" != "1" ] && [ ! -f "$SHARDS" ]; then
    echo "FATAL: $SHARDS not found — this sweep reuses the §36 tube shards." >&2
    exit 2
fi
if [ -f "$OUT/DONE_analysis" ]; then
    echo "WARN: $OUT/DONE_analysis exists — a previous sweep already completed here."
    echo "      Move results/encsweep aside first if you want a clean re-run."
fi
rm -f "$OUT"/FAILED_* 2>/dev/null || true

sentinel () {
    submit --dependency=afternotok:"$2" --kill-on-invalid-dep=yes \
        --export=ALL,STAGE="$1",WATCH_JOB="$2" \
        slurm/encsweep/sentinel.slurm
}

PREP=$(submit slurm/encsweep/prepare.slurm)
S_PREP=$(sentinel prepare "$PREP")

ENC=$(submit --dependency=afterok:"$PREP" slurm/encsweep/encoder.slurm)
S_ENC=$(sentinel encoder "$ENC")

TRAIN=$(submit --dependency=afterok:"$ENC" slurm/encsweep/train.slurm)
S_TRAIN=$(sentinel train "$TRAIN")

IG=$(submit --dependency=afterok:"$TRAIN" slurm/encsweep/ig.slurm)
S_IG=$(sentinel ig "$IG")

SIGN=$(submit --dependency=afterok:"$IG" slurm/encsweep/sign.slurm)
S_SIGN=$(sentinel sign "$SIGN")

ANA=$(submit --dependency=afterok:"$IG",afterany:"$SIGN" slurm/encsweep/analysis.slurm)
S_ANA=$(sentinel analysis "$ANA")

WATCH=$(submit slurm/encsweep/watchdog.slurm)

[ "$DRYRUN" = "1" ] || cat > "$OUT/dag_jobs.env" <<ENVEOF
PREPARE=$PREP
ENCODER=$ENC
TRAIN=$TRAIN
IG=$IG
SIGN=$SIGN
ANALYSIS=$ANA
WATCHDOG=$WATCH
SUBMITTED_AT=$NOW
ENVEOF

echo ""
if [ "$DRYRUN" = "1" ]; then
  echo "Would submit the encoder-breadth sweep (synthetic ids above):"
else
  echo "Submitted the encoder-breadth sweep:"
fi
echo "  prepare   = $PREP    (CPU 64G:          cell bank -> 4 budget-matched Stage-1 sets)"
echo "  encoder   = $ENC    (GPU array 0-3:    one encoder per arm)"
echo "  train     = $TRAIN    (GPU array 0-15%3: 24-cytokine panel x 4 arms)"
echo "  ig        = $IG    (GPU array 0-15%3: top-50 signatures)"
echo "  sign      = $SIGN    (CPU array 0-3%2:  frac_up, secondary)"
echo "  analysis  = $ANA    (CPU 16G:          ARM_COMPARISON.md)"
echo "  watchdog  = $WATCH    (every 30 min -> $OUT/HEALTH.md)"
echo "  sentinels = $S_PREP $S_ENC $S_TRAIN $S_IG $S_SIGN $S_ANA  (afternotok -> $OUT/STATUS.md)"
echo ""
echo "Watch:  tail -40 $OUT/HEALTH.md   |   cat $OUT/STATUS.md"
echo "Result: $OUT/ARM_COMPARISON.md"
