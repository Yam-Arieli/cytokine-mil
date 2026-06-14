# Early-vs-Late Gene Distributional Statistics — PRE-REGISTRATION (locked before run)

**Standalone gene-cascade line (PI's "extend to genes" ask). Kept out of CLAUDE.md so the
cytokine experiment record is untouched.** Locked 2026-06-14, before any cluster run.

## Question (user)
Compare different time stamps in the Sheu BMDM time course and find differences between
**EARLY (source) vs LATE (downstream) genes** in aspects *beyond the mean* — variance, spread,
dispersion, bimodality, recruitment. **No model training**: pure statistics + figures + visual analysis.

## Framing / what this is NOT
This is **direction-AGNOSTIC**. The early→late ORDER is already established (validated onset-time
precedence: LPS source onset ≤1h → downstream ISG ~3h, AUC 0.94). This battery characterizes the
**distribution-SHAPE signature** of that order; it does **not** re-derive direction or causation
(the symmetric-correlation / cross_asym trap). Every spread/bimodality/coherence statistic measures
*existence of heterogeneity*, never *who is upstream*.

## Data
Sheu 2024 mouse BMDM, 500-gene targeted panel, time course **0(=PBS), 0.25, 0.5, 1, 3, 5, 8 hr**
(24h where present), stimuli **LPS** and **PIC** (run independently; headline requires BOTH).
4 pseudo-donors (M0_rep1, M0_rep2, M1_IFNg_rep1, M2_IL4_rep1). Raw counts (Fano/dispersion/Gini/
detection on raw counts; shape/skew/dip/BC on `log1p(normalize_total(1e4))`).

## Gene labels (locked)
- **EARLY / source** (IRF3_direct, drop Ifit3 overlap): `Ccl5, Cxcl10, Ifit2, Ifnb1`
- **LATE / downstream** (IFNAR_induced ISGs): `Mx1, Mx2, Ifit1, Ifit1bl1, Ifit3b, Rsad2, Irf7, Oasl1`
- **Ifnb1** analyzed separately as a producer-cell bimodality positive control; early-vs-late re-run with Ifnb1 held out.
- **Data-driven split** (sensitivity): onset ≤1h vs onset ~3h on all induced genes (validated onset metric).
- **NFkB_canonical targets** + **mean-matched random splits** as negative/specificity controls (H6).

## Statistic battery (per gene × stimulus × timepoint × donor)
Mean (nuisance): `mean_log1p`, `mean_counts`. Spread/dispersion: `var_counts`, `fano_raw` (Poisson=1),
`cv2`, **`fano_residual`** (PRIMARY decoupled = standardized residual of `log var ~ poly(log mean)` fit
**across all panel genes, per timepoint**), `robust IQR / QCD / range90-10`. Shape/bimodality:
`skewness`, `kurtosis`, `bimodality_coef_sarle` (+ expressing-cells-only), `hartigan_dip` (if available;
else BC + Poisson-dropout null). Recruitment/inequality: `frac_expressing` (+`frac_expressing_residual`),
`gini`. Per-gene trajectory features: `dispersion_peak_time`, `heterogeneity_lag` (peak − onset),
`dispersion_transience_index`, `recruitment_t50` + `max_slope`, `early_window_residual_auc`,
`late_module_coexpression_coherence`.

## Pre-registered hypotheses (direction of effect; n_eff = 4 donors, report AUC + sign-agreement, not hard p)
- **H1 (PRIMARY, mean-decoupled overdispersion):** at matched mean, LATE ISGs carry greater excess
  overdispersion (`fano_residual`) than EARLY genes at intermediate times (1–3h). **Passes if** donor-median
  AUC(late>early)>0.5 with label-permutation p≤0.10 at ≥1 intermediate timepoint, surviving **BOTH** the
  residual-fit AND matched-mean routes (signs agree), in **both LPS and PIC**, ≥3/4 donors same sign.
  **Fails (→ pure mean coupling)** if AUC→~0.5 after decoupling.
- **H2 (transient bimodality):** LATE genes transiently bimodal (off/on) at 1–3h, EARLY unimodal; biological
  (survives expressing-cells-only + beats per-gene Poisson/NB dropout null; bimodal-onset coincides ±1 bin
  with recruitment t50).
- **H3 (gradual recruitment):** LATE recruit cells gradually (shallow `frac_expressing` ramp, late t50);
  EARLY switch on fast/uniformly; at matched mean LATE `frac_expressing_residual`<0 at intermediate t.
- **H4 (timing of moments):** peak heterogeneity occurs later and lags onset more for LATE than EARLY
  (`dispersion_peak_time`, `heterogeneity_lag`, transience).
- **H5 (late-module coherence, CONSISTENCY only):** within-late-set across-cell correlation higher than
  within-early and rises 1h→3-5h. Never used as direction evidence.
- **H6 (NEGATIVE/specificity control):** raw variance shows late>early **trivially**; the gap must shrink
  under decoupling. Mean-matched random splits and NFkB-target sets must read AUC~0.5 with no 1–3h peak.

## Mandatory confound controls (implementation MUST include)
1. **Mean-variance coupling (FIRST):** every spread claim survives residual-from-panel-trend **AND**
   matched-mean stratification, and the two AGREE; report raw vs decoupled side by side.
2. **Pseudo-replication:** every stat computed per donor first → donor-median; tests at donor level
   (n_eff=4); per-cell distributions for plots only; no per-cell p-values.
3. **Dropout/zero-inflation:** bimodality recomputed on expressing cells only + beats Poisson null; gated on
   `frac_expressing∈[0.1,0.9]`.
4. **Unequal/small cell counts:** subsample to common n per (stim,t,donor) before moments; bootstrap CIs;
   grey out n<100; treat trends tracking the n-curve as suspect.
5. **Depth/normalization:** count-space stats on raw counts; shape on log1p; per-timepoint trend fit absorbs
   depth; panel-compositional caveat stated.
6. **Gene-set imbalance / Ifnb1:** always per-gene trajectories; rank/AUC set comparisons; leave-one-gene-out;
   Ifnb1 held-out re-run; data-driven split sensitivity.
7. **Cell-type composition (secondary):** report pooled; within-0h-cluster stratification as a caveat/check
   where feasible (loader has no trusted labels → documented as a limitation, not a gate).
8. **Coarse grid:** resolve WAVES (sets), not per-gene order; AUC/rank not median-diff; require excess to span
   ≥2 adjacent timepoints; anchor at 0.5–3h; replicate in both stimuli.

## Figures (must / high)
1. **(must)** Mean-variance cloud per timepoint (all 500 genes) + fitted trend, source/late overplotted — *the trap figure*.
2. **(must)** Set-level decoupled trajectory panel (fano_residual, dip/BC, frac_expressing(+residual), Gini, IQR) source vs downstream, faint per-donor lines, mean reference row; one per stimulus.
3. **(must)** Per-gene per-timepoint distribution ridgeline/violin grid (each source then late gene × timepoints), PBS shaded; Ifnb1 separate.
4. **(must)** Matched-mean honesty check: within-mean-decile late−early spread difference at 1h & 3h; paired raw-vs-decoupled boxplots with 4 donor points.
5. **(high)** Effect-size scorecard: donor-level AUC per feature (raw var, CV, fano_residual, matched-mean, dip, t50, peak-time, lag…) with label-permutation null band + NFkB/random controls.
6. **(high)** Bimodality heatmap (dip/BC genes×t) + expressing-only side panel + dropout null.
7. **(high)** Temporal-aggregate scatter: dispersion_peak_time vs transience; vs onset (y=x).

## Apparatus self-test (run BEFORE cluster)
`--synthetic`: plant EARLY genes (uniform shift, low dispersion, Fano~1, fast detection) vs LATE genes
(fraction-on ramps 0→1 through 1–3h → transient bimodality + overdispersion + gradual recruitment) + flat
background. **Pass = the decoupled battery (fano_residual, BC/dip, frac_residual, recruitment t50) recovers
late>early at intermediate t under BOTH decoupling routes, while raw mean alone barely separates.**

## Verdict targets
- **GREEN:** H1 passes (both routes agree, both stimuli, ≥3/4 donors) AND H6 controls clean → late genes are
  genuinely, mean-independently more heterogeneous at intermediate times (the cascade-propagation signature).
- **AMBER:** H1 direction holds but only pooled / one stimulus / fails one decoupling route.
- **RED / mean-coupling:** decoupled AUC ~0.5 → "late more variable only because higher mean" — reported honestly.

Results → `reports/sheu2024_temporal/EARLY_LATE_STATS_RESULTS.md`.
