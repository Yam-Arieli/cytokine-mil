#!/bin/bash
#SBATCH --job-name=ab_pipe_td
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_ab_pipe_td_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_ab_pipe_td_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Train-donors-only variant of the A->Bridge->B pipeline.
# §24 cells_by_pair is restricted to the 10 training donors (D1, D4..D12),
# matching the train/val split used by the binary AB-MIL models in
# results/oesinghaus_binary/run_20260412_114413_pid3728447/.
# Output dir is distinct from the pooled-all-donors run so both verdicts
# can be compared as a sensitivity check.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/gene_dynamics_phase0/pipeline_a_b_train_only

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_pipeline_a_bridge_b.py \
    --axes_csv          reports/cascade_pairs/cytokine_axes.csv \
    --binary_ig_parquet results/gene_dynamics_phase0/binary_ig/binary_ig.parquet \
    --manifest_path     /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json \
    --hvg_path          /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json \
    --output_dir        results/gene_dynamics_phase0/pipeline_a_b_train_only \
    --top_n             50 \
    --min_cells         10 \
    --exclude_donors    Donor2 Donor3
