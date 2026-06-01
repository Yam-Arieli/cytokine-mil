#!/bin/bash
#SBATCH --job-name=id_build
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_build_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_build_%j.err
#
# Immune Dictionary build step:
#   1. Run download script (idempotent — wget -c resumes; tar -xf no-op if done).
#   2. Run build_pseudotubes_immune_dictionary.py adapter.
#   3. Produce ImmuneDictionary_pseudotubes/{manifest.json, hvg_list.json,
#      build_metadata.json}.
#
# Env: RES (default results/id_cascade)
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

# Defensive default in case --export didn't propagate
RES=${RES:-results/id_cascade}
MANIFEST=/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/manifest.json

echo "=== Immune Dictionary build TP=4h ==="

# 1. Resume / extract download (idempotent)
if [ ! -f /cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary/raw/extracted/.done ]; then
    echo "[$(date -Iseconds)] Running download_immune_dictionary.sh"
    bash scripts/download_immune_dictionary.sh
    touch /cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary/raw/extracted/.done
else
    echo "[$(date -Iseconds)] Download already extracted; skipping"
fi

# 2. Run adapter (skipped if manifest already exists; the adapter itself is also
# idempotent via --skip_existing for individual tubes but we early-return here)
if [ -f "$MANIFEST" ]; then
    echo "[$(date -Iseconds)] Manifest already exists at $MANIFEST; skipping adapter"
else
    echo "[$(date -Iseconds)] Running ID adapter (4000 HVG, Leiden id_c*, val mouse)"
    $PY scripts/build_pseudotubes_immune_dictionary.py
fi

# 3. Verify
echo "[$(date -Iseconds)] Verifying manifest..."
$PY -c "
import json, sys
with open('$MANIFEST') as f:
    m = json.load(f)
cytokines = sorted({e['cytokine'] for e in m})
mice = sorted({e['donor'] for e in m})
cell_types = sorted({ct for e in m for ct in e.get('cell_types_included', [])})
print(f'  tubes:      {len(m)}')
print(f'  cytokines:  {len(cytokines)} (PBS in: {\"PBS\" in cytokines})')
print(f'  mice:       {mice}')
print(f'  cell_types: {cell_types}')
if 'PBS' not in cytokines:
    print('FATAL: PBS not in manifest cytokines'); sys.exit(3)
if len(cytokines) < 50:
    print(f'WARN: only {len(cytokines)} cytokines (expected ~87)')
"
echo "=== DONE build ==="
