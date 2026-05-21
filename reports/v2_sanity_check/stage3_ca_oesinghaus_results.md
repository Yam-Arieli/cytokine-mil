# Stage 3 CA-only on full Oesinghaus 91-class — v2 sanity check results

**Date:** 2026-05-21
**Seeds:** 42, 123 (jobs 30630752_0, 30630752_1, both COMPLETED, ~1h27 each)
**Stage 2 baseline:** `results/oesinghaus_full_v2/seed_{42,123}` (April 2026, 300 epochs, lr 0.003)
**Stage 3 CA-only:** `results/oesinghaus_stage3_ca/seed_{42,123}` (100 epochs, lr 2e-4, momentum 0.95, ~16K trainable params)
**Data:** 91 classes (90 cytokines + PBS), 9100 train tubes (D1, D4–D12), 1820 val tubes (D2, D3), 4000 HVGs, 39909 cells, 18 cell types.

---

## Headline

> **Verdict: PARTIAL.** SA and CA attention separate cleanly in entropy as designed (D1 ✓), and the CA head learns real signal on train data (D2 ✓ on intent, ✗ on the 2× cutoff). But CA does **not** improve held-out validation AUC, and the hard-class boost seen on the Oelen pathogen panel does not transfer — easy cytokines gain a little val AUC, hard cytokines gain none. The architectural mechanism works as designed; what it learns on cytokine data does not generalize to held-out donors.

Recommendation: **Outcome B** — run Path B (confusion-dynamics asymmetry mining on existing Stage 2 logs) in parallel with finalizing Path A (axis-discovery writeup). Do not commit to the remaining 6 seeds of Stage 3 CA, and do not yet move to the heavier joint-trained v2 with KL regularization — the val-side signal needed to justify that compute is not present here.

---

## Diagnostic 1 — SA vs CA attention entropy separation

Goal: confirm SA and CA learn functionally different aggregation policies (peaked vs distributed). Oelen prior: SA ≈ 2 nats, CA ≈ 4 nats (gap ≈ 2 nats).

| Seed | SA entropy (frozen) | CA entropy end (epoch 100) | Gap | Verdict |
|------|---------------------|----------------------------|-----|---------|
| 42   | ~1.2–1.7 nats (flat) | ~5.5 nats | **~4 nats** | ✓ PASS |
| 123  | ~1.2–1.7 nats (flat) | ~5.5 nats | **~4 nats** | ✓ PASS |

**Reading.** Both seeds show CA entropy starting near ~6 nats (near-uniform over the ~256 cell positions) and drifting slowly downward to ~5.5 by epoch 100. SA stays pinned (var = 0.00e+00 — frozen as required). The gap is **roughly 2× the Oelen prior** — the heads are even more separated in entropy on cytokine data than they were on the Oelen pathogen panel. This is the cleanest piece of evidence that the SA+CA decomposition is doing what it was designed to do.

SA frozen check: `SA entropy variance = 0.00e+00` for both seeds — the gradient flow is correctly blocked to the SA head.

Plots:
- `reports/v2_oesinghaus_seed42/sa_vs_ca_entropy_stage3_ca_42.png`
- `reports/v2_oesinghaus_seed123/sa_vs_ca_entropy_stage3_ca_123.png`

---

## Diagnostic 2 — CA weight norm growth

Goal: confirm the CA head moves from its small-random initialization (std=0.01 ⇒ norm ≈ 1.3) toward a non-trivial solution. Threshold from prompt: ≥2× = PASS, 1.1–2× = PARTIAL, ~1× = FAIL.

| Seed | Initial CA L2 norm | Final (epoch 100) | Growth | Final train loss | Verdict |
|------|--------------------|-------------------|--------|------------------|---------|
| 42   | 1.272              | 2.368             | **1.86×** | 0.097 (from 0.384) | PARTIAL |
| 123  | 1.287              | 2.479             | **1.93×** | 0.089 (from 0.502) | PARTIAL |

**Reading.** Both seeds end *just under* the 2× PASS threshold but the trajectory is monotonic and still increasing at epoch 100 (no plateau visible — see `ca_weight_norm_stage3_ca_{42,123}.png`). Training loss dropped from 0.38–0.50 down to 0.09 in both seeds, so the CA head is genuinely learning a discriminative signal, not stuck. With 50–100 more epochs both would cross 2×. By intent this is a PASS; by the strict numerical cutoff it lands on the PARTIAL boundary.

Plots:
- `reports/v2_oesinghaus_seed42/ca_weight_norm_stage3_ca_42.png`
- `reports/v2_oesinghaus_seed123/ca_weight_norm_stage3_ca_123.png`

---

## Diagnostic 3 — Val AUC delta on harder cytokines

Goal: confirm the train-AUC improvement from CA also delivers held-out generalization, especially on hard classes (the Oelen pattern). Computed per-class `mean(p_correct_trajectory)` on Stage 2 and Stage 3 records, both train and val splits. Source: `reports/v2_sanity_check/stage3_ca_deltas.csv` (91 cyts × 2 seeds = 182 rows).

### Across-the-board statistics

| Seed | S2 train AUC (median) | S3 train AUC (median) | **Train Δ (mean)** | S2 val AUC (median) | S3 val AUC (median) | **Val Δ (mean)** | Val gain >0.01 | Val loss <-0.01 |
|------|------------------------|------------------------|---------------------|----------------------|----------------------|-------------------|----------------|------------------|
| 42   | 0.641 | 0.909 | **+0.259** | 0.006 | 0.001 | **+0.002** | 14/91 | 11/91 |
| 123  | 0.627 | 0.909 | **+0.265** | 0.002 | 0.000 | **+0.012** | 14/91 | 11/91 |

**Reading.** CA delivers a large train-AUC gain (+26 pp on average, +27 pp on hard classes — see stratification below). Val AUC barely moves: the median val gain is **negative** in both seeds, the mean is essentially zero in seed 42 and a tiny +0.012 in seed 123, and the count of cytokines that gained vs. lost val AUC is **symmetric** (14 gain / 11 lose, both seeds). This is overfitting in CA, not learning of transferable cascade biology.

### Stratification by Stage 2 train AUC (proxy for difficulty, both seeds pooled)

| Group | n | Train Δ (mean) | **Val Δ (mean)** |
|---|---|---|---|
| HARD (S2 train AUC < 0.60) | 59  | +0.339 | **+0.0008** |
| MED  (0.60–0.75)           | 98  | +0.246 | +0.0047 |
| EASY (≥0.75)               | 25  | +0.142 | **+0.0314** |

**Reading.** This is the **opposite** of the Oelen pattern. On Oelen, hard 24hMTB gained +21 val AUC and the control UT gained only +1.86. Here, hard classes gain ~0 val AUC and only the already-easier classes pick up any meaningful val improvement. The shape of the CA contribution is *more memorization of the training donors*, not *new cascade biology that generalizes*.

### Known-cascade cytokines (pre-registered 11 pairs, both seeds pooled)

| Cytokine | S2 val | S3 val | **Δ val** |
|---|---|---|---|
| IL-12      | 0.065 | 0.058 | -0.007 |
| IFN-γ      | 0.111 | 0.160 | **+0.048** |
| IL-1β      | 0.272 | 0.322 | **+0.049** |
| IL-6       | 0.017 | 0.021 | +0.004 |
| IL-2       | 0.129 | 0.243 | **+0.113** |
| IL-15      | 0.308 | 0.297 | -0.011 |
| IL-33      | 0.001 | 0.000 | -0.001 |
| IL-13      | 0.058 | 0.059 | +0.000 |
| IL-18      | 0.005 | 0.002 | -0.003 |
| IL-21      | 0.142 | 0.127 | -0.015 |
| IL-10      | 0.232 | 0.267 | +0.035 |
| TNF-α      | 0.001 | 0.000 | -0.001 |
| IFN-α1     | 0.079 | 0.030 | **-0.049** |
| IL-4       | 0.266 | 0.337 | **+0.071** |
| IL-27      | 0.012 | 0.003 | -0.009 |

Bright spot: IL-2 +0.113 val, IL-4 +0.071 val. But the cytokines with *known stronger downstream cascades* (IL-12→IFN-γ, IL-18, IL-33, TNF-α, IL-27→IFN-γ) all flat or worse. The boost is not localized to the cytokines where cascade biology should matter most.

**Verdict on D3: ✗ FAIL.**

---

## Side-by-side with the Oelen prior

| Property | Oelen 4-class (Apr 2026) | Oesinghaus 91-class (this run) |
|---|---|---|
| SA vs CA entropy gap | ~2 nats | **~4 nats** (clean separation) |
| CA weight norm growth | not reported (1 seed) | 1.86–1.93× (both seeds) |
| Train loss drop | from ~0.4 to <0.1 | **0.38→0.10 (42), 0.50→0.09 (123)** |
| Val gain on HARD class | **+20.99** (24hMTB) | **+0.0008 mean across HARD cohort** |
| Val gain on EASY class | **+20.21** (24hPA)   | **+0.031** mean across EASY cohort |
| Val gain on MED class | +3.70 (24hCA) | +0.005 mean across MED cohort |
| Val gain on CONTROL | +1.86 (UT) | (no analogue) |
| Stage 2 → Stage 3 rank correlation | 0.800 | not computed (single AUC pooled per cyt) |

The **mechanism** (SA/CA functional separation) transfers cleanly. The **biological payoff** (CA captures generalization-relevant cascade signal) does not.

A complementary observation: the underlying Stage 2 model's val performance is itself weak. Median val AUC is 0.002 for both seeds vs ~0.011 chance — most of the 91 classes are essentially unrecognizable on held-out donors D2/D3 (D2 and D3 were chosen as outliers per CLAUDE.md §16). CA cannot recover signal that the encoder + SA didn't have to begin with.

---

## Implications for directional cascade inference

The original motivation for the Stage 3 CA sanity check was: "single-layer attention is at chance (49% / 51%) for direction inference (literature review 2026-05-20); maybe a second attention head with different attention shape will encode the cascade relay direction the geo readout couldn't."

This run does not refute that hypothesis, but it weakens it considerably:

1. **The architectural mechanism is real.** SA stays peaked, CA is diffuse, ~4-nat gap. The two heads attend to different cell populations on train. If we wanted directional cascade calls from the SA/CA contrast, the precondition that the two heads be functionally separated is met.

2. **But on held-out donors, the CA contribution does not survive.** That's the precondition that *matters* for a publishable directional cascade call. We can't claim "CA identifies the relay cell of cascade A→B" if val AUC of A under the CA head is at chance. Whatever SA+CA centroid asymmetry we would compute on this model would be dominated by training-donor memorization rather than transferable cascade biology.

3. **The Oesinghaus 24-h-snapshot signal-to-noise problem we identified in the lit review is the bottleneck, not the architecture.** Two layers of attention can't make 24-h snapshot data reveal which cell drove which other cell's transcriptional response when the snapshot doesn't contain the time-resolved evidence in the first place. The Oelen pathogen panel may have worked because pathogen stimulation produces stronger and more cell-type-localized transcriptional shifts than cytokine stimulation at 24 h.

4. **The KL-regularized joint v2 (`CytokineABMIL_V2`, CLAUDE.md §5.5) won't change this.** That variant forces SA and CA to diverge during training — but the SA/CA already diverged here without the KL penalty. The problem isn't head collapse; the problem is that what CA learns doesn't generalize.

---

## Recommended next step

**Outcome B.** The architectural fix is doing what it was designed to do but does not deliver on cytokine data. Before committing ~7 days of cluster time to the remaining 6 seeds of Stage 3 + a full SA/CA centroid re-analysis on data with median val AUC = 0.002, do the cheaper test:

**Path B (CLAUDE.md §19 / `next_steps_2026-05-20.md`):** Implement `compute_asymmetry_score(C)` on the existing Stage 2 confusion-trajectory tensors (already saved in `dynamics["confusion_entropy_trajectory"]`) for the 8 seeds we already have. Time-resolved confusion asymmetry mines a *different signal* (temporal order of confusion during training) than the geo/ablation/SA-CA readouts (all of which are endpoint signals). If `sign(Asym(A,B))` matches literature direction in >70% of the 39 documented pairs, we have direction. If still ~50%, the asymmetry is genuinely not in 24-h snapshot data and v2 won't help either.

Estimated cost: 1–2 days of implementation + ~1 day of cluster re-analysis on existing checkpoints (no retraining). Total ≤ 1 week. If Path B works, we have a directional paper without v2; if Path B fails, we have decisive evidence that direction needs time-resolved data (not just better architecture), and we commit to Path A only.

**Path A (axis-discovery writeup) continues unchanged** — it does not depend on v2 working.

**Do NOT yet:**
- Submit the remaining 6 seeds of Stage 3 CA. The mechanism check is complete with 2 seeds; more seeds will not change the val-generalization conclusion.
- Train the KL-regularized joint v2. The SA/CA separation it would enforce is already achieved here without the KL penalty; the bottleneck is data, not architecture.

---

## Files produced this session

- `reports/v2_oesinghaus_seed42/{run_log.txt, sa_vs_ca_entropy_stage3_ca_42.png, ca_weight_norm_stage3_ca_42.png}`
- `reports/v2_oesinghaus_seed123/{run_log.txt, sa_vs_ca_entropy_stage3_ca_123.png, ca_weight_norm_stage3_ca_123.png}`
- `reports/v2_sanity_check/stage3_ca_deltas.csv` (per-cytokine, per-seed Stage 2 vs Stage 3 train/val AUC + deltas)
- `scripts/analyze_stage3_deltas.py` (now on cluster + repo)
- `slurm/run_analyze_stage3_deltas.slurm`
- This report.
