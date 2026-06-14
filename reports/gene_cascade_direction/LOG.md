# Gene-Cascade Direction — Exploration Log

**Standalone log for the "extend cascade detection from cytokines to genes" line of work
(PI's suggestion). Kept separate from CLAUDE.md so the cytokine experiment record is
untouched.** Last updated: 2026-06-14.

## The goal
Detect **directional gene→gene cascades from raw single-cell data** by predicting a held-out
cell from its "soup" (the other cells of its pseudo-tube), with each gene predicted only from
*other* genes (hollow diagonal), reading direction from the **population→cell asymmetry**, with
the output organized as **cell_type × genes** so influence is relay-resolved
`(source_type, gene) → (target_type, gene)`. Training-dynamics framing retained.

## Phase 0 — Toy (simulated), DONE
Files: `scripts/toy_complex_gene_direction.py`. Verdicts (3 seeds, cluster GPU):
- **The reframe works.** Predict-cell-from-soup + hollow diagonal + leave-one-out recovers planted
  cross-cell **direction at AUC 1.00**; mechanism = *downstream genes are predictable from the soup,
  upstream (cell-autonomous) genes are not*.
- **Copy-trap defeated by the hollow diagonal.** A diagonal-only "copy" baseline gets the highest R²
  but **zero** direction — high prediction, no biology.
- **Complex numbers unjustified.** Real full / low-rank match the complex (ComplEx/RotatE) operator on
  recovery and composition; phase-additivity inconsistent → **use real models, drop complex.**
- **One-hop ceiling (key limit).** Direction is recoverable only for a **cell-autonomous source →
  population responder** hop. It **collapses (symmetric)** when both nodes are tube-level
  (stimulus-driven) — the same shared-activation confound as the cytokine work.
- **Training-dynamics-as-readout** (emergence order = cascade depth) does NOT hold for the linear model
  (SNR-driven, not curriculum); a nonlinear cMLP showed weak/ambiguous depth ordering — not pursued as
  primary. Direction lives in the structural asymmetry, not the emergence order.

## Method decisions locked (from the design discussion)
- Output = per-target-type **held-out mean** (single-cell targets are mostly noise).
- Input = per-**source**-type soup summary (mean; distribution-enrichment deferred).
- Hollow mask = exact `(type,gene)` diagonal only; **keep** cross-type-same-gene (the relay).
- Direction metric = **predictability asymmetry** (variance-normalized R²), NOT coefficient magnitude
  (which is variance-ratio-confounded — the original cross_asym trap; re-derived and avoided here).
- Primary model = closed-form **hollow ridge** (exact per-output self-exclusion); cMLP secondary.
- Attention is the eventual home (QK^T = relay tensor, diagonal-masked, Q/K-asymmetric) — only after
  a real-data signal gate is green.

## Phase 1 — Relay Stage 1 on Immune Dictionary (4h snapshots), DONE → **RED**
Files: `cytokine_mil/analysis/relay_cascade.py`, `scripts/relay_cascade_id.py`,
pre-reg `reports/immune_dictionary/RELAY_PREREGISTRATION.md`, results
`reports/immune_dictionary/RELAY_RESULTS.md`. Job 30813990 (seeds 42/123/7).
Conditions IL-12/18/15/IFN-γ/IL-4/PBS; curated relay panel (52/67 genes present); NK→{Mac,cDC1,B} IFN→ISG.

- Apparatus **validated** on synthetic (recovers cell-autonomous relay, collapses on tube-level).
- On real ID: **no reliable signal**, three independent signatures —
  1. `R²_test = −0.44`, per-node predictability negative → soup does not predict held-out residuals.
  2. Direction **not seed-stable** (G2 AMBER/GREEN/GREEN).
  3. Negative control (IL-4) **indistinguishable** from positives (G3 fails all seeds).
- Gate-logic flaw found: G1 (`signal_p<0.05` vs row-shuffle null) passes despite R²<0 — the honest
  gate must require R²>0. Fixed understanding; G1 should fail.
- **Interpretation:** exactly the toy-predicted collapse — IL-12/15/18 drive IFN-γ in ~all NK cells →
  NK IFN-γ is tube-level (stimulus-driven), not cell-autonomous → population→cell asymmetry collapses;
  within-condition cross-cell residual unpredictable at 4h. (Does not affect the published §26 ID
  cross_asym 83% — different quantity.)

## Sanity-check of the RED — IN PROGRESS (job 30830112)
Sweep α∈{1,10,100} × cell-types{NK+Mac, 4-type} × panel{IFN+ISG, full} × seeds{42,123} on the
NK→Macrophage relay. Pass = a config with R²_test>0 AND asym>0 reliable in BOTH seeds AND IL-4
negative NOT reliable. Result: _pending — to be filled in._

## Decisions / next
- If sanity **confirms RED**: do NOT build Stage 2 (attention) — no signal to model. **Pivot direction
  inference to time-resolved Sheu (1h/3h/5h)**, where direction has a real temporal source; the relay
  apparatus transfers, only the direction readout changes (time axis vs population→cell predictability).
- If sanity finds a **clean stable config**: revisit that configuration before any pivot.
- Standing results unaffected: published cytokine axes (121) and cross_asym (88/86/83%).
