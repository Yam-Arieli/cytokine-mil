#!/bin/bash
#SBATCH --job-name=dc_watch
#SBATCH --output=/dev/null
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_watchdog_%j.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=short

# Health layer (b): appends queue state + per-stage progress to HEALTH.md, then
# re-submits itself 30 minutes later. Terminates when BOTH arms' terminal stages finish
# or any stage fails, so it never outlives the DAG.
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
OUT=results/oes90_dc
mkdir -p "$OUT"
H="$OUT/HEALTH.md"
[ -f "$H" ] || echo "# oes90_dc (§40) — DAG health log (appended every 30 min by slurm/oes90_dc/watchdog.slurm)" > "$H"

N_MODEL=$(ls "$OUT"/models/*_head.pt 2>/dev/null | wc -l | tr -d ' ')
N_HIST=$(ls "$OUT"/history/*_train.csv 2>/dev/null | wc -l | tr -d ' ')
N_EMB=$(ls "$OUT"/embeddings/*.npy 2>/dev/null | wc -l | tr -d ' ')
N_SIGM=$(ls "$OUT"/signatures_main_chunk_*.parquet 2>/dev/null | wc -l | tr -d ' ')
N_SIGR=$(ls "$OUT"/signatures_reserve_chunk_*.parquet 2>/dev/null | wc -l | tr -d ' ')
N_DSHC=$(ls "$OUT"/direction_table_curated_shard*.csv 2>/dev/null | wc -l | tr -d ' ')
N_DSHR=$(ls "$OUT"/direction_table_raw_shard*.csv 2>/dev/null | wc -l | tr -d ' ')
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
  echo "- encoded tube shards: $N_EMB"
  echo "- binary model heads: $N_MODEL / 90"
  echo "- training histories: $N_HIST / 90"
  echo "- signature chunks: main $N_SIGM / 9, reserve $N_SIGR / 9"
  echo "- direction shards: curated $N_DSHC / 4, raw $N_DSHR / 4"
  for f in encoder.pt encoder_history.csv signatures_main.parquet signatures_reserve.parquet \
           signatures_main_curated.parquet signatures_reserve_curated.parquet \
           curation_report.csv curation_meta.json signature_stability.csv \
           coupling_donor_degree_curated.csv coupling_donor_degree_raw.csv \
           direction_table_curated.csv direction_table_raw.csv \
           engagement_per_celltype_curated.parquet engagement_per_celltype_raw.parquet; do
    if [ -f "$OUT/$f" ]; then echo "- $f: present ($(du -h "$OUT/$f" | cut -f1))"; else echo "- $f: -"; fi
  done
  # The single cheapest scientific read of the run, available the moment merge lands.
  if [ -f "$OUT/curation_meta.json" ]; then
    echo ""
    echo '```'
    grep -E '"(cap|expected_null_removal|observed_removal_frac|excess_removal|n_conditions_dropped)"' \
         "$OUT/curation_meta.json" 2>/dev/null
    echo '```'
  fi
  echo "- done markers: ${DONE:-none}"
  echo "- failure markers: ${FAILED:-none}"
} >> "$H"

if [ -f "$OUT/DONE_direction_curated" ] && [ -f "$OUT/DONE_direction_raw" ]; then
    echo "" >> "$H"; echo "**DAG complete (both arms) — watchdog exiting.**" >> "$H"; exit 0
fi
if [ -n "$FAILED" ]; then
    echo "" >> "$H"; echo "**Stage failure detected ($FAILED) — watchdog exiting; see STATUS.md.**" >> "$H"; exit 0
fi

sbatch --begin=now+30minutes slurm/oes90_dc/watchdog.slurm > /dev/null
