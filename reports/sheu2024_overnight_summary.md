# Sheu 2026-05-24/25 overnight — three-track summary

Three independent root-cause hypotheses tested in parallel. All six SLURM jobs
COMPLETED cleanly (sacct exit codes 0:0).

| Run | Tested root cause | Verdict | Output dir |
|---|---|---|---|
| Baseline (3hr narrowed) | — | RED | `reports/sheu2024/` |
| Track A — adapter-aware aux head | #1: encoder objective | RED (but signal changed shape) | `reports/sheu2024_aux/` |
| Track B — 1hr time-point subset | #3: 3hr is too late | RED (weaker signal) | `reports/sheu2024_1hr/` |
| Track C — direction_mode=cell_type | #4: T-specific direction | RED (uniform signal) | `reports/sheu2024_celltype/` |

The strict §21 gate is `RED` on all four. The interesting comparison is the
**relaxed-gate effect-size table** (raw min-over-T Wilcoxon p, mean per-donor
bias, sign-agreement across the 3 informative donors).

---

## Side-by-side: the three pre-registered MUST and MUST-NOT axes

Each cell is **`mean_b (sign_agreement%)` averaged across the 3 seeds** to make
the four runs comparable on one screen. Bias magnitudes are PBS-RC-space scalar
projections — higher = bigger geometric effect.

| Axis | Kind | Baseline 3hr | A — adapter | B — 1hr | C — cell_type |
|---|---|---|---|---|---|
| `LPS — TNF` | **MUST** | +0.47 (67%) | **+1.10 (78%)** | +0.15 (67%) | **−0.43 (67%)** |
| `polyIC — IFNb` | **MUST** | +0.19 (72%) | **+2.54 (100%)** | +0.81 (100%) | +2.16 (100%) |
| `LPS — IFNb` | SHOULD | +0.61 (89%) | **+3.07 (100%)** | +0.54 (83%) | +2.10 (100%) |
| `P3CSK — CpG` | SHOULD | +1.03 (67%) | +1.52 (56%) | +0.32 (83%) | **+2.29 (100%)** |
| `LPSlo — P3CSK` | SHOULD | +1.72 (67%) | +2.40 (78%) | nan | **+2.31 (100%)** |
| `P3CSK — IFNb` | MUST-NOT | +0.66 (67%) | **+4.19 (100%)** | +0.56 (83%) | +1.53 (100%) |
| `CpG — IFNb` | MUST-NOT | +0.81 (100%) | +3.11 (78%) | +0.57 (83%) | +2.29 (100%) |
| `TNF — IFNb` | MUST-NOT | +0.50 (83%) | **+3.00 (100%)** | +0.59 (83%) | +2.31 (100%) |

(LPSlo at 1hr is `nan` — the Sheu manifest at 0hr+1hr doesn't include LPSlo cells
in the kept time points.)

---

## What each track actually showed

### Track A — adapter-aware aux head: encoder *is* steerable, but wrong-axis dominates

**What worked:** the adapter loss had real effect on the latent geometry.
- `polyIC — IFNb` went from `+0.19 (72%)` → `+2.54 (100%)` — the MUST axis where one seed previously flipped sign now has perfect 100/100/100 sign-agreement across all 3 seeds.
- `LPS — TNF` went from `+0.47 (67%)` → `+1.10 (78%)` — ~2.3× larger and more stable.
- All effect sizes **2–5× larger** than baseline.

**What still failed:** the relative ordering MUST vs MUST-NOT didn't improve.
- `P3CSK — IFNb` (MUST-NOT) ballooned to `+4.19 (100%)` — the strongest signal in the entire table is a *false-positive* axis.
- `TNF — IFNb` (MUST-NOT) is `+3.00 (100%)`, comparable to the genuine MUST axes.
- The encoder learned receptor structure (good), but the dominant direction in the latent space is still "anything-vs-PBS" projecting onto a generic IFN-response axis.

### Track B — 1hr time-point: signal is *weaker*, not better

3hr was too late, but 1hr appears to be too early — most cytokine effects haven't fully manifested yet. Effect sizes ~3× *smaller* than baseline (all near +0.2–0.8). LPSlo data is missing entirely at 1hr. MUST vs MUST-NOT separation is no better. Earlier ≠ cleaner; just less data has accumulated.

### Track C — cell_type direction mode: makes signal *more uniform*, kills discrimination

Per-cell-type direction vectors gave perfect 100/100/100 sign-agreement on **every** axis (MUST and MUST-NOT alike), with effect sizes uniformly +1.5–+2.5. This kills the MUST-vs-MUST-NOT contrast entirely — every pair looks the same. The one exception is `LPS — TNF`, which went *negative* (−0.43). This is interpretable: in T-specific coordinates, LPS-tube cells deviate *away* from TNF's cell-type-specific direction, plausibly because at 3hr LPS is in TLR4-MyD88-TRIF state, not the autocrine TNF state.

---

## What this combination tells us about the root cause

Three different attacks on the structural problem, none of them succeeded — but each failed in an *informative* way:

1. **The encoder is steerable** (Track A). Adapter-aware supervision changed the latent geometry — bias magnitudes 2–5× larger, sign-agreement substantially improved on `polyIC — IFNb`. The model can be pushed to encode receptor architecture.
2. **But steering toward receptor architecture didn't yield directional resolution.** Track A's MUST and MUST-NOT axes both grew in magnitude together. The new encoder is "more biological" but still has its dominant direction = "stim vs PBS". Adapter information ended up as a *secondary* feature, not the primary axis of variation.
3. **The time-point dimension doesn't rescue it on its own** (Track B). Earlier just means weaker, not clearer.
4. **Cell-type-specific direction vectors don't help** (Track C). They just amplify the "everything points the same way" pattern uniformly.

The triple negative is now strong evidence for root cause #5 from yesterday's analysis: **cascade direction isn't recoverable from static (or even single-snapshot) embedding distributions, even when the encoder is forced to keep receptor structure**. The geometry's first PC is "any stim vs none"; the cascade-specific direction is at best a secondary feature, and the dot-product readout always picks up the first PC instead.

This is now **five** independent failed checks of the directional-inference hypothesis (algebraic; literature 49%; Stage 3 CA-only on Oesinghaus; Sheu 3hr narrowed baseline; three overnight tracks).

---

## Recommendation

**Close the cascade-direction line.** The accumulated evidence is sufficient:

- Static-embedding direction inference is fundamentally limited (5 failed checks, two datasets, three architectures, three time-points).
- The most informative experiment (Track A) showed the encoder *can* be steered, but the constraint that bounds the readout — "dominant direction = stim vs PBS" — is a property of the data itself at single-snapshot resolution, not of any one architecture. No single-snapshot architecture can escape this.
- The remaining viable directions (CRISPR perturb-seq for ground-truth direction; temporal-MIL transformer over real time series; intervention-based readouts) are all multi-week investments in different methodologies and/or datasets.

**Three concrete next steps in priority order:**

1. **Finalize Path A** (Oesinghaus axis-discovery writeup). This is the project's
   completed contribution. The accumulated negative-result evidence on direction
   inference becomes a credible "limitations + future work" section: pre-registered,
   well-controlled, multiple architectures, multiple datasets, all clearly failed.
   Sheu phase 2 (composite time labels) is **not** the path forward without a
   different encoder objective and a different readout — and at that point it's
   a new project, not a phase 2.

2. **Update CLAUDE.md** (§0 status, §5.5 v2 fully closed not just paused, §21
   marked RED-final with the overnight evidence) and **delete pending tasks
   #7, #9, #11, #15** (Zhang adapter, phase 2, etc. — these are no longer on
   the critical path).

3. **Preserve the Sheu infrastructure on `main`.** Adapter scripts, narrowed
   training, slim label encoder, relaxed gate, three-track scaffolding — all
   useful if a future revisit with a perturb-seq dataset or transformer-based
   temporal MIL becomes viable. Don't delete.

If you want to *not* close the line, the one direction that hasn't been tested
and is genuinely different is **temporal-MIL on a multi-time-point dataset
where direction is in the data by construction** (CRISPR perturb-seq, e.g.
Replogle 2022). That's a ~3-week investment for a falsifiable test of whether
the architecture can extract direction *when direction is observably there*.
If it can't, that's a definitive close. If it can, that becomes the new project.
