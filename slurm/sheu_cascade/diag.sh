#!/bin/bash
# One-shot diagnostic for the Sheu DAG + Oesinghaus cross_asym re-run.
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

echo "=== job states (oes + sheu DAG) ==="
sacct -j 30702116,30702117,30702121,30702122,30702123,30702125,30702126,30702127,30702129,30702130,30702131 \
    --format=JobID%15,JobName%12,State,ExitCode,Elapsed -X 2>&1 | head -50

echo ""
echo "=== sheu_cascade logs present ==="
ls -t results/sheu_cascade/slurm_*.out results/sheu_cascade/slurm_*.err 2>/dev/null | head -12

echo ""
echo "=== latest stage12 .err tail ==="
ls -t results/sheu_cascade/slurm_s12_*.err 2>/dev/null | head -1 | xargs tail -20 2>&1

echo ""
echo "=== latest binary .err tail ==="
ls -t results/sheu_cascade/slurm_bin_*.err 2>/dev/null | head -1 | xargs tail -20 2>&1

echo ""
echo "=== latest binary .out tail ==="
ls -t results/sheu_cascade/slurm_bin_*.out 2>/dev/null | head -1 | xargs tail -15 2>&1

echo ""
echo "=== Oesinghaus cross_asym vs AUDITED labels (the real benchmark) ==="
$PY scripts/retally_pipeline_against_audit.py \
    --pipeline_csv results/gene_dynamics_phase0/pipeline_a_b_full19_crossasym/per_celltype.csv \
    --audit_csv reports/cascade_pairs/cytokine_axes_audited.csv \
    --out reports/cascade_pairs/oes_crossasym_audited.md --metric cross_asym 2>&1 | tail -6
