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

## Sanity-check of the RED — DONE (job 30830112) → **RED CONFIRMED**
Swept α∈{1,10,100} × cell-types{NK+Mac, 4-type} × panel{IFN+ISG, full} × seeds{42,123} (24 configs) on
the NK→Macrophage relay. Pass = R²_test>0 AND asym>0 reliable in BOTH seeds AND IL-4 negative NOT
reliable. **No config passed.**
- Stronger ridge / full panel / more types **does** fix R²_test (best: full+4type+α=100 → R²≈0.13>0), so
  the earlier negative R² was partly under-regularization — but it changes nothing about direction:
- In every config the **IL-4 negative control fires the same "reliable" asymmetry as the positive**
  (e.g. full/4type/α=100: asym 0.244 [seed42] vs 0.099 [seed123], `reliable=True` for BOTH positive and
  negative). Positives and negatives stay **indistinguishable**, and the asymmetry is **not seed-stable**.
- Conclusion: the RED is **robust**, not a power/config artifact. ID 4h snapshots carry no separable,
  stable, within-condition cross-cell relay direction — consistent with the toy's tube-level collapse.

## Decisions / next
- If sanity **confirms RED**: do NOT build Stage 2 (attention) — no signal to model. **Pivot direction
  inference to time-resolved Sheu (1h/3h/5h)**, where direction has a real temporal source; the relay
  apparatus transfers, only the direction readout changes (time axis vs population→cell predictability).
- If sanity finds a **clean stable config**: revisit that configuration before any pivot.
- Standing results unaffected: published cytokine axes (121) and cross_asym (88/86/83%).

## Phase 2 — Gradient learning-order (training dynamics, Sheu 3hr), DONE → **AMBER/RED**
Hypothesis: cascade-SOURCE genes are learned before DOWNSTREAM genes (follow per-gene gradient over
epochs). Files: `cytokine_mil/analysis/{attribution,learning_order}.py`,
`scripts/{train_sheu_binary_learning_order,extract_gene_attribution_trajectory,compute_sheu_realtime_emergence,analyze_gene_learning_order}.py`;
pre-reg `LEARNING_ORDER_PREREGISTRATION.md`; results `LEARNING_ORDER_RESULTS.md`. Jobs 30830386 (train) →
30830578 (analyze), seeds 42/123/7, PIC & LPS vs PBS.
- Apparatus validated on synthetic: controls (H1 effect-size-matched, H2 partial-vs-real-time) correctly
  PASS a real cascade-order regime and REJECT an SNR-confound regime (raw "source-first" fires in both).
- Real data: **both controls reject.** H1 p_matched not significant / not seed-stable (LPS 0.06/1/1;
  PIC 0.67/0.68/0.78); H2 partial ≈ 0 (|≤0.07|, p≥0.17). Any raw source-first is an SNR/learnability
  artifact, not cascade order. H1 (no real-time) rejects on its own → robust.
- Same root cause as the relay RED: in 3hr snapshots, source & downstream IFN genes are both expressed and
  co-learnable; learning order is set by effect size, not cascade depth.

**Net (after Phases 1–2):** two snapshot-based angles (relay; learning-order) both NEGATIVE under
validated controls. The honest remaining avenue is genuinely time-resolved trajectories.

## Phase 3 — Time-resolved cascades (Sheu 0.25→8/24hr), DONE → **metric-AMBER, biology-VISIBLE**
Files: `cytokine_mil/analysis/temporal_cascade.py`, `scripts/run_sheu_temporal_cascade.py`,
pre-reg `reports/sheu2024_temporal/TEMPORAL_CASCADE_PREREGISTRATION.md`, results `…/TEMPORAL_CASCADE_RESULTS.md`.
Job 30833076 (PIC, LPS). Apparatus validated on synthetic (planted early→late: AUC=1.0).
- **The IFN cascade IS visible in real kinetics** (cascade_{PIC,LPS}.png): IRF3-direct (source) induced
  early (~1h bump, ≈0.7–0.85 above baseline) while IFNAR-induced ISGs stay flat (~0) until 3h, in BOTH
  stimuli — textbook TRIF→IRF3→autocrine IFN-β→ISG, read off real time. **First positive sighting in this line.**
- **But the pre-registered metric (50%-of-OWN-max activation) is AMBER/RED**: it's dominated by the
  sustained 3–24h plateau, so every gene "activates" at 3h (V1 AUC≈0.55–0.63, ns). Diagnosed, not a null.
- **Fix:** absolute-threshold / early-window onset metric (source crosses ~0.5 at ~1h, downstream at ~3h)
  recovers the ordering — a transparent metric correction (50%-max conflates onset with peak).
- **Overall takeaway:** direction needs real time (it appears here; snapshots collapsed). The remaining
  work is just the onset metric to quantify the visible cascade.

## Phase 3b — Onset-metric re-run (job 30833205) → **LPS cascade recovered (pooled); gate AMBER (donor power)**
Absolute-threshold onset (≥0.5) instead of 50%-of-max. Results `…/TEMPORAL_CASCADE_RESULTS.md`.
- **LPS GREEN (pooled):** IRF3-direct source genes onset 0.25–1h (Ccl5 0.25, Cxcl10/Ifit2 0.5, Ifnb1 1h);
  ISGs onset 3h. **V1 AUC=0.938 p=0.008; 15/15 directed edges source→downstream.** First clean positive.
- **PIC weak:** only Ccl5 early; rest at 3h → AUC=0.58 ns.
- **Donor-stable = false (both):** per-donor onset noisy (few cells/donor/timepoint) — power limitation.
- **Pre-registered gate (both stimuli + donor-stable): AMBER/RED**, but the LPS pooled recovery is a
  genuine mechanism-consistent positive. Gap to clean GREEN = more cells/donor/timepoint + PIC kinetics.

## Phase 4 — Distribution BEYOND the mean: early-vs-late variance/spread/bimodality (Sheu), DONE → **H1 refuted; onset-locked + producer-burst**
User ask: compare timestamps, find early-vs-late differences in MORE aspects than onset of the mean —
variance, spread, dispersion, bimodality. Pure statistics, no training. Files:
`cytokine_mil/analysis/temporal_gene_stats.py`, `scripts/run_sheu_temporal_gene_stats.py`,
`slurm/run_sheu_temporal_gene_stats.slurm`; pre-reg `…/EARLY_LATE_STATS_PREREGISTRATION.md`; results
`…/EARLY_LATE_STATS_RESULTS.md`. Job 30833452 (LPS, PIC). Battery designed via a 4-lens design workflow
(shape-statistician / single-cell biologist / confound-adversary / viz) → 11 per-(gene,timepoint) stats +
6 trajectory features, with the mean-variance coupling trap controlled (Fano-residual + matched-mean),
tested at DONOR level (n_eff). Apparatus validated on synthetic (decoupled AUC 1.0; raw mean not discriminating).
- **Pre-reg H1 (late cascade products MORE overdispersed at intermediate t): REFUTED.** After mean-decoupling,
  `fano_res` AUC(late>early) ≤0.5 everywhere (LPS 0.06/0.36, PIC 0.34/0.20; ns); raw var differences
  (source>late, AUC 0.00–0.05) are pure mean-coupling. Controls at null (random-split 0.44/0.59). Verdict RED.
- **Real finding #1 — heterogeneity is ONSET-LOCKED, not cascade-depth-locked.** `distributions_*.png`: every
  gene is most spread/bimodal AT ITS OWN ONSET (partial cell penetrance) and converges to uniform-high as it
  saturates. Spread re-encodes the onset timing; it adds no new direction beyond Phase-3 onset.
- **Real finding #2 — producer-cell burstiness at the SOURCE (mean-decoupled, replicated).** `det_res`
  AUC=1.00 (p=0.00) + negative detection-residual for source in both stimuli: Ifnb1 + chemokines (Ccl5,Cxcl10)
  expressed in FEWER cells than their bulk level implies (bursty/producer-restricted); late ISGs broad. INVERTS
  the naive prediction (heterogeneity at the producer step, not the responder step).
- **Bimodality is mostly dropout** — late-gene Sarle-BC at 0.25h collapses on expressing-cells-only (H2 fails its
  dropout gate). Timing features `disp_peak_time`/`het_lag` corrupted for late genes by the low-mean-early floor.

**FINAL throughline (Phases 1–4):** gene-cascade DIRECTION is unrecoverable from snapshots (relay RED;
learning-order SNR-AMBER) but DOES appear once you use real biological time (LPS IFN cascade recovered,
pooled). The method (onset timing + directed precedence on a real time course) works; the bar to a fully
donor-stable, multi-stimulus claim is data power, not method. The DISTRIBUTION beyond the mean (variance/
spread/bimodality) does NOT add new directional info — it is onset-locked (each gene's spread pulse tracks
its own onset) — but it does reveal a producer-cell burstiness signature at cascade SOURCES (Ifnb1/chemokines
restricted to a minority of cells) vs broad downstream ISG responses.
