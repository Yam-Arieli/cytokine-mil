#!/bin/bash
# One-shot submission: (1) re-run the Oesinghaus pipeline with the cross_asym
# driver (reuses existing all-24 binary IG; ~30-90 min CPU, adds the cross_asym
# null control), and (2) submit the full Sheu single-frame DAG.
#
# Run ON the cluster in a login shell (sbatch in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/sheu_cascade/submit_all.sh'"
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
DS=/cs/labs/mornitzan/yam.arieli/datasets

# ---- (1) Oesinghaus cross_asym re-run (same inputs as full19, new driver) ----
OUT=results/gene_dynamics_phase0/pipeline_a_b_full19_crossasym
mkdir -p "$OUT"
OES=$(sbatch --parsable --job-name=oes_xa --mem=24G --cpus-per-task=4 \
    --time=01:30:00 --partition=short \
    --output=results/gene_dynamics_phase0/oes_xa_%j.out \
    --wrap="$PY scripts/run_pipeline_a_bridge_b.py \
        --axes_csv reports/cascade_pairs/cytokine_axes.csv \
        --binary_ig_parquet results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet \
        --manifest_path $DS/Oesinghaus_pseudotubes/manifest.json \
        --hvg_path $DS/Oesinghaus_pseudotubes/hvg_list.json \
        --output_dir $OUT --top_n 50 --min_cells 10 --n_null_perms 100")
echo "Oesinghaus cross_asym re-run: job $OES -> $OUT"

# ---- (2) Sheu single-frame multi-time DAG ----
bash slurm/sheu_cascade/submit_dag.sh

echo ""
echo "All submitted. Bottom lines:"
echo "  Oesinghaus: $OUT/verdict.md  (cross_asym primary + null)"
echo "  Sheu:       reports/sheu_cascade/timepoint_comparison.md"
