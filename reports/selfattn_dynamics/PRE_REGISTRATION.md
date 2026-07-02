# Pre-registration — Self-attention over cells: cell↔cell interaction (§34)

**Locked before the cluster analysis job runs on real data (§25.1 discipline).** Method/spec:
CLAUDE.md §34. This fixes the operationalizations and GREEN/AMBER/RED gates so the verdict is
not chosen after seeing the result. Direction-not-existence and not-causation caveats (§26.4)
carry over. This experiment is **additive** — it does NOT replace IG `cross_asym` (§26).

## Experiment

- **Data/scope:** Oesinghaus 24h PBMC, 91-class **multiclass**, frozen encoder. One checkpointed
  Stage-2 run per seed; **seeds 42, 123, 7**. Identical to the §33 baseline
  (`results/attention_dynamics`) except the attention layer.
- **Architecture (locked):** `encoder(frozen) → SAB(1 layer, 4 heads) → AB-MIL AttentionModule
  pooling → BagClassifier`. Optimizer **SGD** (momentum 0.9, lr 0.001, warmup 5) — matches §33.
- **Checkpoints:** every epoch, 1..250 (finer than needed; gates are grid-agnostic).
- **Readouts:** (1) pooling weights → the §33 P1–P4 machinery **unchanged**; (2) cell-type ×
  cell-type interaction matrix `M[τ,σ]` and its off-diagonal (cross-type) mass; (3) a per-pair
  direction statistic scored on the audited benchmark.

## Locked operationalizations

- **Interaction matrix** `M[τ,σ]` = mean over τ-cells of the total self-attention they place on
  σ-cells (row-normalised, Σ_σ M[τ,σ]=1), head-averaged over the single SAB layer, donor-mean.
- **Off-diagonal mass** = mean over cells of the fraction of a cell's attention on OTHER cell
  types (1 = fully cross-type; ~0 = self/diagonal).
- **Interaction asymmetry** `Asym[τ,σ] = M[τ,σ] − M[σ,τ]` (antisymmetric).
- **Relay direction** for a pair (A,B): `D(A,B) = M^A[T_B,T_A] − M^B[T_A,T_B]`, T_A/T_B = pooling
  attention-primary cell types; `D>0 ⇒ A→B` (a_to_b). Sign scored vs `expected_sign`.
- **relay-lag** (§33): sign of `relay_recruitment_lag` mean_lag (`>0 ⇒ A→B`).
- **Benchmark:** `reports/cascade_pairs/cytokine_axes_audited.csv`, `counts_in_benchmark=True`
  rows (the 15/17 = 88% denominator), sign vs `expected_sign` (+1 ⇒ a_to_b). Same as `cross_asym`.
- **Known cascades:** IL-12/IL-2/IL-15→IFN-γ (relay=NK/mono, Fig 4f/i). **Negative control:**
  IL-6 / TNF-α (coupled only via shared activation, §28 negative — no clean relay expected).

## Pre-registered predictions and gates

| ID | prediction | GREEN | AMBER | RED |
|---|---|---|---|---|
| **G0** (interaction exists) | self-attention uses cross-type info | mean final off-diagonal fraction ≥ 0.20 | ≥ 0.05 | < 0.05 (SAB inert → premise broken; STOP) |
| **S1** (pooling head sane) | SAB-pooling attention-primary recovers known responders (§33 P1) | frac_match (top-3) ≥ 0.6 | ≥ 0.4 | else |
| **S2** (direction beats/matches 88%) | a self-attention direction stat is competitive on the labeled benchmark | either stat ≥ 0.80 accuracy | ≥ 0.65 | < 0.65 |
| **S3** (relay biology) | known cascades' interaction points A→B; control ambiguous | ≥ 2/3 cascades a_to_b AND control not a clean a_to_b | ≥ 1/3 | 0/3 |

**Overall:** GREEN iff **G0 passes** AND (**S2 GREEN** OR **S3 GREEN**) — i.e. cells demonstrably
interact and the interaction adds a competitive or biologically-right direction signal beyond
AB-MIL. AMBER if G0 passes but S2/S3 only partial. RED if G0 fails (self-attention collapsed to
self-attention-per-cell → no interaction → no added value over §33).

## Reference (not a prediction)

The IG `cross_asym` Oesinghaus direction accuracy (**88% = 15/17**, §26) is the reference row in
`SELFATTN_RESULTS.md`. It is unchanged by the §28.2 panel/degree updates (those touch only the
*symmetric* coupling gate). IG `cross_asym` remains the project's primary coupling+direction
method regardless of this experiment's outcome.

## Honest caveats

Attention is task-driven (discriminative), NOT biology — validate on held-out donors. A relay is
visible only if it lives in the frozen cell-type-pretrained embedding subspace (representability
risk). Direction ≠ existence (Path A's job) ≠ causation. Small benchmark n and donor N; multi-seed
before trusting ordering (the dynamics pipeline is seed-noisy). SGD may train the self-attention
slowly — if G0/S1 fail with clearly-untrained attention, LR-warmup / Adam is the documented retry,
not a post-hoc gate change.
