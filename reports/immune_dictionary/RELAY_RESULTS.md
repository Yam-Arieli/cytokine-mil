# Relay Cascade — Stage 1 Results (Immune Dictionary, 4h snapshots)

**Verdict: RED — no reliable cross-cell relay signal on ID snapshots with this design.**
Run: job 30813990, seeds 42/123/7, ~2.5 min each. Driver `scripts/relay_cascade_id.py`,
apparatus `cytokine_mil/analysis/relay_cascade.py`. Pre-registration:
`reports/immune_dictionary/RELAY_PREREGISTRATION.md`.

## Setup actually run
- Conditions IL-12, IL-18, IL-15, IFN-γ, IL-4, PBS; fit on IL-12/18/15 + PBS (140 tubes).
- Curated relay panel: **52/67** panel genes present in ID; IFN-axis source = 5 genes, ISG target = 10.
- Up to 7 cell types; P = C·G ≈ 364; per-tube leave-one-out, PBS-residualised, hollow ridge (α=1).
- Headline edge: NK_cell → {Macrophage, cDC1, B_cell} on IFN-axis → ISG.

## Result (decisive, three independent signatures of no signal)
1. **Negative predictive power.** `R²_test = −0.44`; per-node predictability (`pred_tgt`, `pred_src`)
   **negative** for all targets. The held-out per-type residual is NOT predictable from the soup
   (worse than the mean). → the population→cell map carries no usable signal here.
2. **Direction not seed-stable.** G2 (relay direction) = AMBER / GREEN / GREEN across seeds 42/123/7
   — it flips with the RNG. The per-seed "GREEN"s are tiny asymmetries (~0.07) clearing the 0.1
   threshold by chance; not reproducible.
3. **Negative control fails everywhere.** G3 fails in all seeds: the IL-4 (Th2) negative fires the
   *same* "reliable" asymmetry (+0.068) as the Macrophage positive → positives and negatives are
   indistinguishable. The bootstrap "reliable" flag is detecting noise (CI excludes 0 by a hair on an
   unpredictive model).

**Gate-logic note (flaw to fix):** G1 as coded (`signal_p < 0.05` vs a row-shuffle permutation null)
PASSED despite `R²_test < 0`, because permuted R² is even more negative. The honest signal gate must
require `R²_test > 0`. With that, G1 fails — consistent with G2/G3/G4.

## Interpretation (matches the toy's prediction exactly)
The toy established the **one-hop ceiling**: direction is recoverable only when the upstream node has
**cell-autonomous** variation driving a population-aggregate response; it **collapses** when both nodes
are **tube-level** (stimulus-driven). On ID, IL-12/15/18 drive IFN-γ in ~all NK cells → NK IFN-γ is a
**tube-level (stimulus-driven)** quantity, not cell-autonomous → the population→cell asymmetry collapses.
Additionally, the within-condition cross-cell residual (cell − type/PBS mean) is essentially unpredictable
at 4h → negative R². Both point to the same conclusion: **ID 4h snapshots do not carry a recoverable
within-condition cross-cell relay signal** by this method.

This does NOT contradict the published §26 ID `cross_asym` 83% — that measures *cytokine-signature*
direction across hand-picked labeled pairs, a different quantity from a *within-condition cross-cell-type*
gene relay.

## Recommendation
- **Do not proceed to Stage 2 (attention)** — there is no signal here to model; the apparatus is correct
  (synthetic self-test passed), the data lacks the structure.
- **Pivot direction inference to time-resolved data (Sheu 1h/3h/5h)** where direction has a real temporal
  source (early vs late), instead of relying on a cell-autonomy asymmetry that stimulus-driven snapshots
  don't provide. The relay apparatus (`relay_cascade.py`) transfers directly; the direction readout would
  use the time axis rather than population→cell predictability.
- Alternatively, accept the honest scope: the published axis/`cross_asym` cytokine results stand;
  gene-level *directed* cascades are not recoverable from these snapshots.
