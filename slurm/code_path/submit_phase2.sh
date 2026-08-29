#!/bin/bash
# Submit Phase 1+2 of the controlled code-path comparison (SPEC.md §3-§4).
#
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/code_path/submit_phase2.sh'"
#   SUBMIT=echo bash slurm/code_path/submit_phase2.sh        # dry run
#
#   encoder[0-5%3] --afterok--> arm[0-11%4] --afterok--> analysis
#
# The square is run rather than Phase 2 alone: the transplant arms are uninterpretable
# without both pure references, and the only pure-cytokine_mil fit in existence (run B,
# meanJ 0.079) has no matching pure-cascadir arm on the same protocol. The diagonal IS
# Phase 1, so one run answers both.
set -e
REPO=${REPO:-/cs/labs/mornitzan/yam.arieli/cytokine-mil}
cd "$REPO"
OUT=results/code_path
mkdir -p "$OUT"

SUBMIT=${SUBMIT:-sbatch}
DRYRUN=0
[ "$SUBMIT" != "sbatch" ] && DRYRUN=1
FAKE=$(mktemp); echo 9700000 > "$FAKE"; trap 'rm -f "$FAKE"' EXIT
submit () {
    if [ "$DRYRUN" = "1" ]; then
        local id=$(( $(cat "$FAKE") + 1 )); echo "$id" > "$FAKE"
        echo "  sbatch --parsable $*   ->  $id" >&2; echo "$id"
    else
        sbatch --parsable "$@"
    fi
}

for f in scripts/phase2_train_encoder.py scripts/phase2_train_arm.py \
         scripts/analyze_phase2.py; do
    [ -f "$f" ] || { echo "FATAL: $f not found — is the cluster clone up to date?" >&2; exit 2; }
done
if [ "$DRYRUN" != "1" ]; then
    # Phase 0a is what licenses reading this as a CODE comparison rather than a data one.
    G=results/code_path/phase0/phase0a_tube_identity.json
    [ -f "$G" ] || { echo "FATAL: $G missing — run Phase 0 first." >&2; exit 2; }
    grep -q '"passed": true' "$G" || {
        echo "FATAL: Phase 0a did not pass — the two paths do not read identical tubes," >&2
        echo "       so this comparison would be measuring data, not code." >&2; exit 2; }
    for f in results/oes_full90/tubes/meta.json results/oes90_dc/stage1_cells.h5ad; do
        [ -e "$f" ] || { echo "FATAL: $f missing." >&2; exit 2; }
    done
fi

NOW=$(date -Is 2>/dev/null || date)
[ "$DRYRUN" = "1" ] && echo "DRY RUN (SUBMIT=$SUBMIT) — nothing will be queued:" >&2

ENC=$(submit slurm/code_path/phase2_encoder.slurm)
ARM=$(submit --dependency=afterok:"$ENC" slurm/code_path/phase2_arm.slurm)
ANA=$(submit --dependency=afterok:"$ARM" slurm/code_path/phase2_analysis.slurm)

cat <<SUMMARY

Phase 1+2 submitted ($NOW)

  encoder   $ENC  GPU array 0-5%3    — 2 paths x 3 seeds, same Stage-1 cells
  arm       $ARM  GPU array 0-11%4   — 4 arms x 3 seeds, published-24 panel
  analysis  $ANA  CPU                — pre-registered rule + verdict

  arms:  cm_cm = pure cytokine_mil (P)    cd_cd = pure cascadir (C)
         cm_cd = T1 (cm encoder)          cd_cm = T2 (cd encoder)

Attribution is held FIXED across all four arms — Phase 0 showed the two IG
implementations agree exactly, so the arms differ only in which code made the weights.

Bottom line:  $OUT/phase2/PHASE2_VERDICT.md
              $OUT/phase2/phase2_arm_summary.csv   (means + across-seed spread)

Read P2 (the seed control) FIRST. If within-arm seed spread is not small against the
between-arm gap, no arm comparison is a path effect and the rest does not signify.
SUMMARY
