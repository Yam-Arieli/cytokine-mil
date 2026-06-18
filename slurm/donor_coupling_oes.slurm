#!/bin/bash
#SBATCH --job-name=donor_coupling_oes
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_donorcoup_oes_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_donorcoup_oes_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=short

# Donor-level coupling gate validation (CLAUDE.md §16/§28). Tests whether a
# donor-level sign-flip null discriminates coupling (vs the over-powered cell-level
# ~77%), and whether IG_vsPanel beats IG_vsPBS once the null is honest. Reads the
# ablation summary for the cell-level reference. Research code only (not cascadir).

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

IG=results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json
AUDIT=reports/cascade_pairs/cytokine_axes_audited.csv
ABL=results/sig_ablation/oes/ablation_summary.csv

$PY scripts/run_donor_coupling_null.py \
    --dataset oesinghaus \
    --binary_ig_parquet "$IG" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --audit_csv "$AUDIT" \
    --ablation_summary "$ABL" \
    --exclude_donors Donor2 Donor3 \
    --top_n 50 --min_cells 10 --n_perm 200 --min_donors 5 \
    --output_dir results/sig_ablation/oes/donor_coupling

echo "DONE. report at results/sig_ablation/oes/donor_coupling/donor_coupling_report.md"
