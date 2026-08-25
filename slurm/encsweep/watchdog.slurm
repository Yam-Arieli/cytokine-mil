#!/bin/bash
#SBATCH --job-name=es_watch
#SBATCH --output=/dev/null
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encsweep/slurm_watchdog_%j.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=short

# Health layer (b): appends queue state + per-arm progress to HEALTH.md, then re-submits
# itself 30 minutes later. Terminates when the analysis stage finishes or any stage fails,
# so it never outlives the DAG.
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
OUT=results/encsweep
mkdir -p "$OUT"
H="$OUT/HEALTH.md"
[ -f "$H" ] || echo "# encsweep — DAG health log (appended every 30 min by slurm/encsweep/watchdog.slurm)" > "$H"

DONE=$(ls "$OUT"/DONE_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
FAILED=$(ls "$OUT"/FAILED_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')

{
  echo ""
  echo "## $(date -Is)"
  echo ""
  echo '```'
  squeue -u "$USER" -o "%.10i %.12j %.9P %.2t %.10M %.6D %R" 2>/dev/null | head -30
  echo '```'
  echo ""
  for ARM in pbs_only rand18 rand45 all90; do
    A="$OUT/$ARM"
    ENC="-"; [ -f "$A/encoder.pt" ] && ENC="yes"
    NM=$(ls "$A"/models/*_head.pt 2>/dev/null | wc -l | tr -d ' ')
    NS=$(ls "$A"/signatures_chunk_*.parquet 2>/dev/null | wc -l | tr -d ' ')
    SIG="-"; [ -f "$A/signatures.parquet" ] && SIG="merged"
    echo "- \`$ARM\`: encoder=$ENC  heads=$NM/24  sig_chunks=$NS/4  $SIG"
  done
  echo "- done markers: ${DONE:-none}"
  echo "- failure markers: ${FAILED:-none}"
} >> "$H"

if [ -f "$OUT/DONE_analysis" ]; then
    echo "" >> "$H"; echo "**DAG complete — watchdog exiting.**" >> "$H"; exit 0
fi
if [ -n "$FAILED" ]; then
    echo "" >> "$H"; echo "**Stage failure detected ($FAILED) — watchdog exiting; see STATUS.md.**" >> "$H"; exit 0
fi

sbatch --begin=now+30minutes slurm/encsweep/watchdog.slurm > /dev/null
