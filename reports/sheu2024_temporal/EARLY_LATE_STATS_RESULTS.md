# Early-vs-Late Gene Distributional Statistics — RESULTS (Sheu time course)

**Job 30833452 (COMPLETED, 2:18), LPS + PIC, 8 pseudo-donors (4 with the full 0.25→8h grid).
Pre-reg: `EARLY_LATE_STATS_PREREGISTRATION.md`. Apparatus validated on synthetic first
(planted early-uniform vs late-transient-bimodal → decoupled Fano-residual/BC AUC = 1.0,
raw mean not the discriminator).** Direction-agnostic: characterizes distribution shape of the
already-established early→late order; does NOT re-derive direction.

## Headline
**The pre-registered hypothesis (late/cascade-product genes are MORE heterogeneous than
early/source genes at intermediate times) is REFUTED.** After mean-decoupling there is no
consistent excess-overdispersion difference by cascade role, and the raw differences run the
*opposite* way (source genes more variable — but that is mean-coupling). The honest, robust,
**replicated (LPS+PIC)** findings are:

1. **Heterogeneity is ONSET-LOCKED, not cascade-depth-locked.** Every induced gene is maximally
   spread/bimodal *at its own onset* (only a subset of cells have switched it on) and converges
   to a uniform-high distribution as it saturates (see `distributions_{LPS,PIC}.png`: Ccl5/Cxcl10/
   Ifit2 are broad/bimodal at 0.25–1h then sharpen by 3–8h; ISGs sit at 0 until 3h then broaden).
   Because source genes turn on early and ISGs late, their spread *pulses* are correspondingly
   early and late — so "which set is more variable at a fixed time" just reports which set is
   mid-onset then. **Spread adds no NEW directional information beyond the onset metric.**
2. **The one genuine cascade-role-specific, mean-decoupled signal: producer-cell burstiness at the
   SOURCE.** `detection_residual` (frac-expressing minus what the mean predicts) is **negative for
   source genes and positive for ISGs in BOTH stimuli** (trajectory panels), and the donor-level
   `det_res` AUC(late>early) = **1.00, p=0.00** at 3h. I.e. **Ifnb1 and the chemokines Ccl5/Cxcl10
   are expressed in FEWER cells than their bulk level implies (bursty / producer-restricted),
   while late ISGs respond broadly** — textbook "few producers → many paracrine responders". This
   INVERTS the naive prediction (heterogeneity is at the producer step, not the responder step).
   `Ifnb1` is the extreme: a spike at zero with a tiny expressing tail at all times.
3. **Apparent bimodality is mostly technical dropout.** `bimodality_{LPS}.png`: late ISGs look
   bimodal (Sarle BC) at 0.25h — but they are unexpressed then; on **expressing-cells-only** that
   bimodality COLLAPSES (zero-inflation, exactly the pre-registered dropout confound). No gene shows
   robust persistent biological off/on bimodality in this 500-gene panel except Ifnb1 (too sparse
   to quantify). **H2 fails its own dropout gate.**

## Quantitative readout (LPS @ 3h, donor-level AUC(late > early); 0.5 = no difference)
| stat | AUC | reading |
|---|---:|---|
| `det_res` (decoupled detection) | **1.00** (p=0.00) | late broad, **source bursty** — the robust signal |
| `iqr` | 0.70 | late wider at 3h (late mid-onset then) |
| `cv2` | 0.62 | mild |
| `gini` | 0.59 | mild |
| **`fano_res` (PRIMARY decoupled overdispersion)** | **0.36 (p=0.46, ns)** | **no directional difference after decoupling** |
| `bc_sarle` | 0.34 | source slightly more bimodal |
| `var_counts` (raw, coupled) | 0.05 (p=0.00) | source ≫ late — **but pure mean-coupling** |
| `fano_raw` (raw, coupled) | 0.00 | source ≫ late — mean-coupling |
| `early_window_fano_auc` | 0.09 | source accumulates spread in ≤3h window (its onset) |
| ctrl `random_split` | 0.44 | ≈ null ✓ |

PIC: same pattern (`fano_res` 1h=0.34, 3h=0.20; raw var 0.13/0.16; det_res negative for source;
random-split 0.59 ≈ null). H1 `fano_res` AUC(late>early): LPS 1h=0.06 / 3h=0.36; PIC 1h=0.34 / 3h=0.20
— all ≤0.5 → late NOT more overdispersed at any intermediate time, in either stimulus.

## What the figures show (visual analysis)
- **`meanvar_cloud_*`**: every gene hugs the per-timepoint variance~mean trend; source genes reach
  HIGHER mean and sit on/slightly-above the trend at 1–3h; late genes are NOT above the trend. Mean-
  variance coupling is the dominant structure; group differences are modest deviations from it.
- **`trajectories_*`** (8 panels): source (red) above late (blue) on Fano-residual / Sarle BC / Gini /
  IQR / CV² in the early window, and **negative detection-residual for source** — consistent across
  every spread metric and both stimuli. Mean (raw) also higher for source (they are strongly-induced
  chemokines), so the decoupled stats are what matter — and even they show source ≥ late early.
- **`distributions_*`**: the rawest evidence — the onset-locked spread story, and Ifnb1 as the
  producer-cell extreme.
- **`scorecard_*`**: the AUC table above, with the raw-vs-decoupled contrast and controls at null.
- **`bimodality_*`**: all-cells vs expressing-only — the dropout collapse.
- **`temporal_aggregate_*`**: timing-of-moments; note `disp_peak_time`/`het_lag` are CORRUPTED for late
  genes (their Fano-residual "peaks" at 0.25h where they are near-zero/noisy, not at a real bump) —
  the low-mean floor artifact; do not over-read these two features.

## Verdict
- **Pre-registered H1 gate (late > early decoupled overdispersion, both stimuli, donor-stable): RED /
  refuted.** Raw spread differences are mean-coupling; decoupled `fano_res` ≈ 0.5 (ns).
- **Scientific content (the real answer): a clear, replicated, mean-decoupled difference DOES exist —
  it is just (a) onset-locked rather than cascade-depth-locked, and (b) producer-cell burstiness at
  the SOURCE (Ifnb1/chemokines), the opposite of the naive prediction.** The cell-by-cell recruitment
  heterogeneity that the hypothesis attributed to downstream cascade products is in fact present at
  every gene's onset, and the residual cascade-role-specific heterogeneity lives at the producer step.

## Honest caveats / limits
- Fixed-timepoint early-vs-late comparisons conflate cascade role with onset stage; the cleaner
  comparison is each gene at its own onset (an onset-aligned re-analysis is the natural next step).
- `det_res` should ideally be confirmed at matched onset-stage (late genes at 3h are low-mean, where
  detection residual is most sensitive); the trajectory sign is consistent across both stimuli, which
  argues it is real, but matched-onset confirmation would harden it.
- 500-gene targeted panel; pooled across BMDM subsets (no within-cell-type stratification; the 8th
  donor set BMDM1/2 strain variants only contribute the 8h endpoint). n_eff = donors, not cells.
- Bimodality is dropout-limited in this panel — a deeper (whole-transcriptome) dataset would test the
  off/on responder structure properly.

## Throughline (gene-cascade line, Phases 1–4)
Snapshots can't give direction (relay RED; learning-order SNR-AMBER). Real time gives direction via
onset precedence (LPS IFN cascade, AUC 0.94). **This phase (distribution beyond the mean): the spread/
variance/bimodality signature is onset-locked — it re-encodes the same timing, not new direction — plus
a producer-cell burstiness signature at the source.** Gene cascades, read from the data, are a sequence
of onset waves where each gene first appears in a minority of cells; "early vs late" differs in WHEN
that partial-penetrance pulse happens, and source ligands (Ifnb1/chemokines) stay producer-restricted
while downstream ISGs spread broadly.
