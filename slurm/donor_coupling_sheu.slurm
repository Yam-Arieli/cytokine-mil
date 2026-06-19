#!/bin/bash
#SBATCH --job-name=donor_coupling_sheu
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_donorcoup_sheu_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_donorcoup_sheu_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Generalization test (2nd dataset): donor-level + degree-corrected coupling gate on
# Sheu 5hr (targeted 500-gene panel, mouse BMDM). 6 pseudo-donors -> donor null is
# small but valid (unanimous-6 -> p=1/64). Benchmark = pre-registered MUST (LPS-TNF,
# PIC-IFNb) / MUST-NOT (P3CSK-IFNb, CpG-IFNb, TNF-IFNb) coupling labels. Key question:
# does the degree correction keep the textbook IFN cascades coupled while suppressing
# the MUST-NOTs, on a panel where EVERY gene is shared-activation? Research code only.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

IG=results/sheu_cascade/5hr/binary_ig/binary_ig.parquet
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json

$PY scripts/run_donor_coupling_null.py \
    --dataset sheu \
    --binary_ig_parquet "$IG" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --time_filter 5hr \
    --top_n 50 --min_cells 10 --n_perm 200 --min_donors 4 \
    --output_dir results/sig_ablation/sheu5hr/donor_coupling

echo "DONE. report at results/sig_ablation/sheu5hr/donor_coupling/donor_coupling_report.md"
