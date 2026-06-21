#!/bin/bash
#SBATCH --job-name=enc_abl_dn
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encoder_ablation/slurm_down_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encoder_ablation/slurm_down_%j.err
#
# Downstream for both encoder-ablation arms: IG signatures -> cross_asym (direction)
# + donor-level coupling, then compare two_stage vs single_stage.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json
AXES=reports/cascade_pairs/cytokine_axes_audited.csv

for ARM in two_stage single_stage; do
    BIN=results/encoder_ablation/$ARM/binary
    IGDIR=results/encoder_ablation/$ARM/binary_ig
    mkdir -p "$IGDIR"
    TARGETS=$(ls "$BIN"/model_*.pt 2>/dev/null | sed -E 's#.*/model_(.*)\.pt#\1#' | tr '\n' ' ')
    echo "=== [$ARM] IG  targets=[$TARGETS] ==="
    if [ -z "$TARGETS" ]; then echo "FATAL: no models for $ARM"; exit 3; fi
    $PY scripts/run_binary_ig_probe.py \
        --binary_run_dir "$BIN" --output_dir "$IGDIR" \
        --manifest_path "$MAN" --hvg_path "$HVG" --targets $TARGETS \
        --top_n 50 --n_ig_steps 20 --max_tubes_per_cytokine 10 --device cpu

    echo "=== [$ARM] cross_asym (direction) + cell coupling ==="
    $PY scripts/run_cell_degree_coupling.py --dataset oesinghaus \
        --binary_ig_parquet "$IGDIR/binary_ig.parquet" \
        --manifest_path "$MAN" --hvg_path "$HVG" --axes_csv "$AXES" \
        --exclude_donors Donor2 Donor3 \
        --top_n 50 --min_cells 10 --n_perm 500 \
        --output_dir results/encoder_ablation/$ARM/coupling_cell

    echo "=== [$ARM] donor-level coupling gate ==="
    $PY scripts/run_donor_coupling_null.py --dataset oesinghaus \
        --binary_ig_parquet "$IGDIR/binary_ig.parquet" \
        --manifest_path "$MAN" --hvg_path "$HVG" --audit_csv "$AXES" \
        --exclude_donors Donor2 Donor3 \
        --min_donors 8 --top_n 50 --min_cells 10 --n_perm 200 \
        --output_dir results/encoder_ablation/$ARM/donor_coupling
done

echo "=== COMPARE two_stage vs single_stage ==="
$PY scripts/compare_encoder_ablation.py \
    --base results/encoder_ablation --audit_csv "$AXES" \
    --out reports/method_summary/encoder_ablation_result.md
echo "=== DONE ==="
cat reports/method_summary/encoder_ablation_result.md
