#!/bin/bash
#SBATCH --job-name=dc_sent
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_sentinel_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_sentinel_%j.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=short

# Health layer (a): submitted with --dependency=afternotok:<stage job>, so it runs ONLY
# when that stage fails or is cancelled. Records a one-line reason in STATUS.md and drops
# a FAILED_<stage> marker, so a dead stage is visible without opening any log.
#
# Env: STAGE (name), WATCH_JOB (the job id it is watching).
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
OUT=results/oes90_dc
mkdir -p "$OUT"

STAGE="${STAGE:-unknown}"
WATCH_JOB="${WATCH_JOB:-unknown}"

INFO=$(sacct -j "$WATCH_JOB" --format=JobID,State,ExitCode,Elapsed --noheader -P 2>/dev/null | head -5 | tr '\n' ';')
[ -z "$INFO" ] && INFO="sacct returned nothing (job may have been purged)"

echo "FAILED stage=$STAGE job=$WATCH_JOB at=$(date -Is) :: $INFO" >> "$OUT/STATUS.md"
touch "$OUT/FAILED_$STAGE"
echo "sentinel: recorded failure of stage=$STAGE job=$WATCH_JOB"
