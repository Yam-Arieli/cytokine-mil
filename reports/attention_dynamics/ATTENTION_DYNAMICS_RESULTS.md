# Attention training-dynamics — cell-type-resolved cascade (CLAUDE.md §33)

**Placeholder — not yet run on real data.** This file is overwritten by
`scripts/analyze_attention_dynamics.py` once the checkpointed **multiclass** Oesinghaus
Stage-2 run (3 seeds, checkpoints every 10 epochs) exists on the cluster and
`scripts/extract_attention_trajectory.py` has produced `attention_trajectory.pkl`.

Status:
- Code + pre-registration + unit tests + SLURM DAG: **built**.
- Local demo de-risk (harness correctness on the synthetic fixture, not biology): see
  `scripts/run_demo_attention_dynamics.py`.
- Real cluster run + verdict: **pending** (separate approved step).

Pre-registered gates (P1–P4) and operationalizations are locked in
`reports/attention_dynamics/PRE_REGISTRATION.md`.
