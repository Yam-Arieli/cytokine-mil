# Pre-registration — Directed Cross-Cell Relay Cascades (gene-level), Immune Dictionary

**Locked before any audit/real-data run (per CLAUDE.md §25.1). Commit to `main` first.**
Date: 2026-06-11.

## Hypothesis
In a pseudo-tube ("soup"), a held-out cell's per-cell-type expression is predicted from the OTHER
cells, with each gene predicted only from *other* genes (hollow `(cell_type, gene)` diagonal). Direction
is read from the asymmetry of the learned relay-influence tensor `(source_type, gene) → (target_type,
gene)`. The leave-one-out (target excluded from the soup aggregate) is the source of the population→cell
asymmetry.

**Scope (from the toy):** this recovers a **one-hop** cell-autonomous-source → population-responder
relation. It is NOT expected to order multi-step cascades, and direction collapses when both nodes are
purely tube-level (stimulus-driven). Stage 1 is an honest test of whether ID snapshots carry any
recoverable *within-condition cross-cell* directional signal at all.

## Conditions (tubes built)
IL-12, IL-18, IL-15, IFN-γ, IL-4 (Th2 negative), PBS. Source/target cell types focus on:
`NK_cell, Macrophage, cDC1, B_cell, T_cell_CD8, T_cell_CD4` (use any present with ≥ min_cells).

## Gene panel
Curated relay panel (~80–120 mouse genes) read from RAW ID tubes (full transcriptome), guaranteeing the
IFN-γ axis is present: producers/transactivators `Ifng, Stat1, Irf1, Irf7, Il12a, Il12b`; ISG targets
`Isg15, Mx1, Mx2, Oasl2, Oas1a, Gbp2, Usp18, Rsad2, Ifit1, Ifit3`; lineage markers (NK/mac/T/B); Th2
negatives `Gata3, Il4, Il13`. Exact list committed in `cytokine_mil/analysis/relay_cascade.py`.

## Pre-registered POSITIVES (expected recoverable IF the relay carries cell-autonomous source signal)
| # | Relay edge | Source type | Target type(s) | In which tubes | Predicted |
|---|---|---|---|---|---|
| 1 | NK IFN-γ axis → responder ISG | NK_cell | Macrophage, cDC1, B_cell | IL-12, IL-18, IL-15 | direction NK → responder (source asymmetry > 0), absent in PBS |

Operationalised: the relay-influence from `(NK, {Ifng,Stat1,Irf1})` to `(Macrophage/cDC1/B, {Mx1,Isg15,
Oasl2,Gbp2,...})` exceeds the reverse, and clears the direction permutation null (q ≤ 0.10).

## Pre-registered NEGATIVES (must NOT produce a directional relay call)
| # | Pair | Reason |
|---|---|---|
| N1 | IL-4 condition, any NK→responder IFN edge | Th2; no IFN-γ relay |
| N2 | masked self edges `(T,g) → (T,g)` | structurally zero by the hollow mask (sanity) |
| N3 | reverse `(responder ISG) → (NK IFN-axis)` | wrong direction; must be weaker than forward |

## Metric & gates (locked)
Per-tube leave-one-out samples; ridge hollow influence + cMLP-hollow; 3 seeds (42/123/7).
- **G1 signal exists:** held-out per-type R² > permutation-null 95th pctile (else premise fails on ID).
- **G2 known relay:** positive #1 forward > reverse, clears direction null (q ≤ 0.10), present in
  IL-12/15/18 and absent in PBS.
- **G3 negatives clean:** N1 not called; N3 forward > reverse; N2 ≈ 0.
- **G4 seed-stable:** sign of the positive relay agrees across the 3 seeds.

**Verdict:** GREEN = G1 + G2 + (G3 mostly) + G4 → proceed to Stage 2 (attention) / Stage 3 (modules).
RED = G1 fails (no signal) OR G2 collapses to symmetric (the tube-level ceiling) → honest stop on ID
snapshots; pivot direction inference to time-resolved data (Sheu).

## Discipline
No tuning of the panel, conditions, edges, or gates after seeing results. Synthetic self-test
(`--synthetic`) validates the pipeline against a planted relay BEFORE the real run; it is a code check,
not part of the registered claim.
