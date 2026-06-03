#!/bin/bash
#SBATCH --job-name=gu_coup_sheu
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_coup_sheu_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_coup_sheu_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Signature-space coupling on Sheu (CLAUDE.md §28) — the decisive 'irrelevant
# features' test. Does coupling in cytokine-SPECIFIC dimensions recover the
# textbook TLR pairs (LPS-TNF, polyIC-IFNb) that the latent-geometry Path A gate
# FAILED (q=1 everywhere, reports/sheu2024/AXIS_GATE_VERDICT.md)?
# Run at 3hr (the frame the gate failed on) AND 5hr (best cross_asym frame).

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

DATASETS=/cs/labs/mornitzan/yam.arieli/datasets
AXES=reports/sheu_cascade/sheu_axes_labeled.csv

# Sheu pseudotube manifests are PER-TIME-POINT (the standard manifest is 0hr+3hr
# only, §2.5; 5hr was built separately). Signatures are also per-TP. Use the
# frame-matched manifest AND signature for each TP (single-frame, no leakage).
declare -A PTDIR=( [3hr]=Sheu2024_pseudotubes [5hr]=Sheu2024_5hr_pseudotubes )
for TP in 3hr 5hr; do
  MAN=$DATASETS/${PTDIR[$TP]}/manifest.json
  HVG=$DATASETS/${PTDIR[$TP]}/hvg_list.json
  echo "=== Sheu signature coupling @ $TP (manifest ${PTDIR[$TP]}) ==="
  $PY scripts/run_signature_coupling.py \
      --dataset sheu \
      --binary_ig_parquet "results/sheu_cascade/${TP}/binary_ig/binary_ig.parquet" \
      --manifest_path "$MAN" \
      --hvg_path "$HVG" \
      --axes_csv "$AXES" \
      --time_filter "$TP" \
      --top_n 50 --min_cells 10 --n_perm 1000 \
      --output_dir results/group_u/coupling_sheu_${TP}
done

echo "DONE. reports at results/group_u/coupling_sheu_{3hr,5hr}/coupling_report.md"
