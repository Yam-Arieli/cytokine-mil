#!/bin/bash
# Submit Phase 0 of the controlled code-path comparison
# (reports/code_path_comparison/SPEC.md §2).
#
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/code_path/submit_phase0.sh'"
#   SUBMIT=echo bash slurm/code_path/submit_phase0.sh        # dry run
#
#   gate0a (CPU) --afterok--> gate0b (GPU)
#
# 0b runs only if 0a passes, because that is what "hard gate" means: if the two paths do
# not read identical tubes then the as-run half of 0b is comparing data, not code, and the
# honest move is to stop and report that instead of producing a number that looks like an
# answer. 0a exits nonzero on mismatch so the dependency does the enforcing.
#
# Neither stage trains anything. Both read artifacts that already exist: the committed
# pseudo-tubes, the §36 shards, and the published run's saved weights.
set -e
REPO=${REPO:-/cs/labs/mornitzan/yam.arieli/cytokine-mil}
cd "$REPO"
OUT=results/code_path
mkdir -p "$OUT"

SUBMIT=${SUBMIT:-sbatch}
DRYRUN=0
[ "$SUBMIT" != "sbatch" ] && DRYRUN=1
FAKE=$(mktemp); echo 9600000 > "$FAKE"; trap 'rm -f "$FAKE"' EXIT

submit () {
    if [ "$DRYRUN" = "1" ]; then
        local id=$(( $(cat "$FAKE") + 1 )); echo "$id" > "$FAKE"
        echo "  sbatch --parsable $*   ->  $id" >&2; echo "$id"
    else
        sbatch --parsable "$@"
    fi
}

for f in scripts/phase0_tube_identity.py scripts/phase0_ig_transplant.py; do
    [ -f "$f" ] || { echo "FATAL: $f not found — is the cluster clone up to date?" >&2; exit 2; }
done
if [ "$DRYRUN" != "1" ]; then
    [ -f results/oes_full90/tubes/meta.json ] || {
        echo "FATAL: §36 tube shards missing — 0a has nothing to compare against." >&2; exit 2; }
    ls results/oesinghaus_binary_missing16/run_*/model_*.pt >/dev/null 2>&1 || {
        echo "FATAL: no published run-B model weights found — 0b has nothing to transplant." >&2
        exit 2; }
fi

NOW=$(date -Is 2>/dev/null || date)
[ "$DRYRUN" = "1" ] && echo "DRY RUN (SUBMIT=$SUBMIT) — nothing will be queued:" >&2

A=$(submit slurm/code_path/gate0a.slurm)
B=$(submit --dependency=afterok:"$A" slurm/code_path/gate0b.slurm)

cat <<SUMMARY

Phase 0 submitted ($NOW)

  gate0a  $A  CPU 48G  — tube identity: do both paths read bit-identical X?
  gate0b  $B  GPU      — IG transplant on the published anchor (runs only if 0a passes)

Neither stage trains. Read in this order when they land:

  $OUT/phase0/phase0a_tube_identity.json     passed: true/false
  $OUT/phase0/phase0b_config_agreement.csv   cm_prod vs cd_cond.cm_base.cm = the algorithm test
  $OUT/phase0/phase0b_diversity.csv          meanJ per configuration

If 0a fails, the code-path gap is a DATA difference and Phase 1 should not run.
If 0a passes and the algorithm test is >= 0.95, the attribution code is exonerated and
the 2x2 says whether tube/baseline SELECTION explains the rest.
SUMMARY
