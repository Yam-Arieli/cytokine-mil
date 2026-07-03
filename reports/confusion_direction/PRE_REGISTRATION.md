# Pre-registration — Directed cascade graph from confusion dynamics (signature-restricted)

**Locked before the restricted-model analysis (§25.1).** Method: the approved plan + CLAUDE.md
§19. This is a **direction** claim (who is upstream), not existence/causation. Additive to
`cross_asym` (§26), which stays primary.

## Hypothesis / mechanism
A multiclass model learns strong primary patterns first, weak secondary late. For cascade A→B,
A-tubes carry `S_A` + weak `S_B` (autocrine product); B-tubes carry `S_B` only. So the classifier
asymmetrically confuses A as B. **Direction (locked):** `Asym[A,B] = mean_late(C[A,B] − C[B,A])`,
`C[A,B]` = A-tubes' softmax mass on B; **`Asym[A,B] > 0 ⇒ A→B`** (same sign convention as
`cross_asym`; sanity IL-12→IFN-γ ⇒ Asym>0). Restricting features to `U = ∪ S_X` (§28) removes the
shared-activation program so cascade leakage dominates the confusion.

## Locked settings
- Data: Oesinghaus multiclass, seeds 42/123/7; `late_epoch_fraction = 0.3`.
- Features: `U` = union of top-50 binary-IG genes per cytokine ∩ 4000-HVG.
- Benchmark: `sign(Asym[axis_a,axis_b])` vs `expected_sign` on `counts_in_benchmark=True` rows of
  `cytokine_axes_audited.csv` (the 15/17 = 88% denominator).
- Temporal: `compute_temporal_profile` peak-fraction; cascade set = benchmark positives (scored on
  the expected upstream→downstream direction); shared set = `KNOWN_COREGULATED` pairs.
- Control: identical scorer on the existing FULL-gene 4000-HVG 3-seed run (no training).

## Gates
| ID | prediction | GREEN | AMBER | RED |
|---|---|---|---|---|
| **P1** direction works | restricted accuracy > chance | ≥ 0.65 on the 17 pairs (binomial p<0.05) | > 0.5 | ≤ 0.5 |
| **P2** restriction is the key (§28) | restricted > full-gene control | restricted − control ≥ +0.1 | > control | ≤ control |
| **P3** temporal (novel) | cascade peaks later than coregulated | one-sided perm p < 0.05 | Δ > 0 | Δ ≤ 0 |
| **P4** regression | §19 positives called A→B | IL-12/IL-2/IL-15→IFN-γ all A→B | ≥ 1 | 0 |

**Overall GREEN** iff **P1 GREEN and (P2 or P3 GREEN)** — i.e. the training-dynamics route recovers
direction AND either the signature-restriction or the temporal signal adds value beyond the full-gene
baseline. AMBER if P1 only. RED if P1 fails (dynamics confusion carries no direction here).

## Honest caveats
Confusion-asymmetry and `cross_asym` are both driven by "A carries `S_B`", so matching 88% is a
robustness result, not new direction info — the genuinely new claim is **P3 (temporal)**. If
restricted ≈ control ≈ chance and P3 fails, the dynamics route adds nothing; report that. Sign-based
scoring uses tube-level confusion; the donor-level FDR graph (`build_cascade_graph`) is the
significance refinement. Signatures are cytokine-specific-by-construction → confusion may be small;
mitigated by 3 seeds. `S_a ≈ S_b` overlap pairs (e.g. ISG-dominated) can flip — a known boundary.
