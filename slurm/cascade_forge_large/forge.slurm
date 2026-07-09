#!/bin/bash
#SBATCH --job-name=cfl_forge
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_large/slurm_forge_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_large/slurm_forge_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Forge all large-cascade configs (~1.05M cells each) on CPU. Produces 7 h5ads:
#   all_eff0.30/{snapshot_t3,snapshot_t6}, receptor_eff0.30/{snapshot_t3,snapshot_t6},
#   all_eff{0.15,0.20,0.40}/snapshot_t6  -> plus h5ad_manifest.txt for the benchmark array.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUT=results/cascade_forge_large
mkdir -p "$OUT"

run() { $PY scripts/forge_large_cascade.py "$@"; }

run --responder_mode all      --effect_size 0.30 --snapshot_times 3,6 --out "$OUT/all_eff0.30"
run --responder_mode receptor --effect_size 0.30 --snapshot_times 3,6 --out "$OUT/receptor_eff0.30"
run --responder_mode all      --effect_size 0.15 --snapshot_times 6   --out "$OUT/all_eff0.15"
run --responder_mode all      --effect_size 0.20 --snapshot_times 6   --out "$OUT/all_eff0.20"
run --responder_mode all      --effect_size 0.40 --snapshot_times 6   --out "$OUT/all_eff0.40"

find "$OUT" -name 'snapshot_t*.h5ad' | sort > "$OUT/h5ad_manifest.txt"
echo "=== forged h5ads ($(wc -l < "$OUT/h5ad_manifest.txt")) ==="
cat "$OUT/h5ad_manifest.txt"
