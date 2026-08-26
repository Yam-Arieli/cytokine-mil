#!/bin/bash
# Encoder gene-space geometry probe (CLAUDE.md §39): probe -> analysis.
#
# Read-only over artifacts that already exist -- no training, no GPU, two short CPU stages.
# Each stage carries an `afternotok` sentinel so a failure lands in STATUS.md without
# anyone opening a log. No watchdog: the whole DAG is under an hour.
#
#   SUBMIT=echo bash slurm/encoder_geometry/submit_encoder_geometry.sh   # dry run
#   bash slurm/encoder_geometry/submit_encoder_geometry.sh               # submit
set -e
cd "$(dirname "$0")/../.."
SUBMIT="${SUBMIT:-sbatch}"
D=slurm/encoder_geometry
OUT=results/encoder_geometry
mkdir -p "$OUT" reports/encoder_geometry

# Command substitution runs `sub` in a subshell, so a shell variable counter would not
# survive; the dry run keeps its fake ids in a file so they stay distinct and readable.
FAKE=$(mktemp); echo 90000000 > "$FAKE"
trap 'rm -f "$FAKE"' EXIT

sub() {  # sub <file> [extra sbatch args...]
    local f="$1"; shift
    if [ "$SUBMIT" = "echo" ]; then
        local n=$(( $(cat "$FAKE") + 1 )); echo "$n" > "$FAKE"
        echo "[dry] sbatch $* $f  -> $n" >&2; echo "$n"
    else
        $SUBMIT --parsable "$@" "$f"
    fi
}

PROBE=$(sub $D/probe.slurm)
echo "probe            $PROBE"
sub $D/sentinel.slurm --dependency=afternotok:$PROBE \
    --export=ALL,STAGE=probe,WATCH_JOB=$PROBE >/dev/null

ANALYSIS=$(sub $D/analysis.slurm --dependency=afterok:$PROBE)
echo "analysis         $ANALYSIS  (afterok:$PROBE)"
sub $D/sentinel.slurm --dependency=afternotok:$ANALYSIS \
    --export=ALL,STAGE=analysis,WATCH_JOB=$ANALYSIS >/dev/null

echo
echo "verdict -> reports/encoder_geometry/ENCODER_GEOMETRY.md"
echo "status  -> $OUT/STATUS.md   (written only on failure)"
