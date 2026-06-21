# Pre-registration — Recurrent IG over training dynamics (Oesinghaus)

**Locked before the analysis job runs (§25.1 discipline).** Hypotheses and rationale:
`hypotheses/recurrent_training_dynamics_IG.md`. Method/spec: CLAUDE.md §31. This file
fixes the operationalizations and GREEN/AMBER/RED gates so the verdict is not chosen
after seeing the result.

## Experiment

- **Data/scope:** Oesinghaus 24h PBMC; all 45 cytokines in
  `reports/cascade_pairs/cytokine_axes.csv` present in the manifest, one binary AB-MIL
  (cytokine vs PBS) each, on a **single shared Stage-1 encoder per seed** + the **wide
  config** (embed=512, hidden=(512,512), attn=128, Stage1 20@0.005, Stage2 250@3e-5).
- **Seeds:** 42, 123, 7 (seed-stability is the robustness axis).
- **Recurrent IG:** every **10 epochs** of Stage-2 (full-model) training, NOT during the
  encoder pretrain. Full per-checkpoint ranking stored to a tracked band of **top-300**.
- **IG:** cascadir `integrated_gradients`, PBS-per-gene-mean baseline, 20 midpoint steps,
  `target_class=0` (positive), accumulated over up to 10 train tubes per cytokine.

## Locked operationalizations

- **Signature band:** top-`TOP_K=50` = "in the signature".
- **Recruitment epoch `τ_in`:** first checkpoint where the gene is in-band **and** stays
  in-band for ≥ `PERSIST=0.8` of the remaining checkpoints (persistence, not first touch).
- **Categories** (per gene×cytokine×seed, over the epoch range [10,250]):
  *Anchor* = `τ_in` in the first third (≤ ~90) and in-band at the final epoch;
  *Climber* = `τ_in` in the last third (≥ ~170) and in-band at final;
  *Flicker* = in-band early (≤ ~90) but NOT at final; *Mid* = otherwise.
- **Seed-stable signature member:** in top-50 at the final epoch in ≥ `SEED_STABLE_MIN=2`
  of the 3 seeds.
- **Effect size (P-A):** `mean(log-norm expr in cytokine) − mean(log-norm expr in PBS)`
  per (cytokine, gene), pooled over **train donors** (Donor2/Donor3 excluded).
- **Shared-activation genes (P-C/E):** in top-50 of ≥ `SHARED_FRAC=0.25 × n_cytokines`
  cytokines at the final epoch (seed-mean); **specific** = top-50 of exactly 1 cytokine.
- **Direction (P-D):** for labeled pairs (a<b alphabetically) `cross_asym(a,b) = s(a,S_b) −
  s(b,S_a)` (cascadir, degree-corrected coupling panel, final epoch); `expected_sign` from
  `cytokine_axes_audited.csv` (`counts_in_benchmark` ∧ sign ∈ {+1,−1}). Timing statistic
  `Δτ(g) = τ_in(g,b) − τ_in(g,a)` over shared genes `S_a∩S_b` (>0 ⇒ a upstream, matching
  `expected_sign=+1`). **Permutation null:** shuffle epoch labels within each
  (gene,cytokine,seed) rank trajectory, recompute `τ_in` and the signed-agreement rate,
  `N_PERM=1000`, `seed=123`.

## Pre-registered predictions and gates

| ID | prediction | GREEN | AMBER | RED |
|---|---|---|---|---|
| **P-A** (primacy) | `τ_in` ↑ with subtlety ⇒ negative Spearman(`τ_in`, \|effect\|) on seed-stable members | pooled ρ < −0.1 and p < 0.05 | ρ < 0 | ρ ≥ 0 |
| **P-B** (canonical anchors) | marker-panel genes are Anchors in their expected cytokine | anchor frac ≥ 0.6 | ≥ 0.4 | < 0.4 |
| **P-C** (sharpening) | specificity entropy `H_g(t)` falls; shared-gene promiscuity decays | entropy slope < 0 **and** shared-prom slope ≤ 0 | entropy slope < 0 | entropy slope ≥ 0 |
| **P-D** (direction triangulation) | sign(Δτ) recovers known direction, beating the timing-permutation null | p_perm < 0.05 **and** agree(expected) ≥ 0.6 | agree(expected) > 0.5 | else |
| **P-E** (collapse warning) | high early shared-program overlap flags the cross_asym sign-flip / wrong pairs | reported descriptively (early-shared vs correctness); not gated for the overall verdict |

**Overall:** GREEN iff **P-A and P-D** are both GREEN (recurrent IG adds biology worth
folding into the method); AMBER if any of P-A/P-B/P-C is GREEN but P-D is not (descriptive
interpretability layer only); RED if P-A fails (recurrent IG redundant with the static probe).

## Faithfulness guard (not a prediction)

Final-epoch (epoch 250) `cross_asym` signed accuracy on the labeled non-AMBIGUOUS pairs is
reported against the published §26 anchor (15/17 ≈ 0.88). An accuracy well below ~0.75 is a
**P4-style regression** flag (the run did not reproduce the published signatures) and is
called out in the results; trajectory claims are then interpreted with that caveat.
