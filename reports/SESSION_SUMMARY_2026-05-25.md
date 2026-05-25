# Session summary — 2026-05-23 to 2026-05-25

End of a multi-day push to test cascade-direction inference on the Sheu 2024
BMDM time-course dataset. This document is self-contained so the next session
can pick up without scrolling back.

---

## TL;DR

- Built complete Sheu phase 1 pipeline (adapter → train → §21 axis-discovery gate).
- Ran the gate **four times** under increasingly stringent diagnostic conditions
  (narrow encoder, relaxed readout, then three parallel root-cause variants).
- **All four runs RED.** This is the 4th, 5th, 6th, 7th, and 8th independent
  failed check of the cascade-direction hypothesis (cumulative with the 3 prior
  Oesinghaus checks).
- **But:** Yam pushed back on the close-the-line recommendation. Argument:
  "5 failed checks of one specific method ≠ no signal exists." Correct.
  Decision is to **double down**, rethink basic assumptions, and try
  fundamentally different attacks within the 3-month thesis window.
- **Three new experiments proposed** for next session — each tests a different
  assumption that all prior runs shared. None require retraining.

---

## What was built (artifacts on `main`)

### Cluster data
- `/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw/` — GSE224518 raw (62 MB tar + extracted)
- `/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/` — phase 1 pseudotubes (0hr+3hr), 300 manifest entries, 4 informative pseudo-donors
- `/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_1hr_pseudotubes/` — Track B variant (0hr+1hr)

### Pseudo-donors actually available at 3hr (CLAUDE.md §2.5)
| Pseudo-donor | Split | Stims at 3hr |
|---|---|---|
| `M0_rep1` | train | LPS, LPSlo, P3CSK, polyIC, TNF, CpG (no IFNb) |
| `M0_rep2` | train | LPS, P3CSK, polyIC, TNF, CpG, IFNb (no LPSlo) |
| `M1_IFNg_rep1` | train | LPS, P3CSK, polyIC, TNF, CpG, IFNb |
| `M2_IL4_rep1` | **val** | LPS, P3CSK, polyIC, TNF, CpG, IFNb |

Effective n = 3 informative pseudo-donors per cytokine.
GEO batches 14-16 (with `PM_B6` samples) are referenced in metadata but not
deposited — verified directly against the deposit. The other two pseudo-donors
visible in the manifest (`M1_IFNg_rep2`, `M2_IL4_rep2`) only contribute 0hr/PBS.

### Code (new files on `main`)
- `scripts/build_pseudotubes_sheu2024.py` — adapter (content-based batch_id inference; commit `b766918`)
- `scripts/train_sheu2024_stage12.py` — Stage 1+2 trainer with slim 8-class label encoder, narrowed dims (commit `9f9e6f5`)
- `scripts/train_sheu2024_stage12_aux.py` — Track A variant with receptor-adapter aux head in Stage 1 (commit `8007f80`)
- `scripts/run_sheu_axis_gate.py` — §20.1 PBS-RC + per-donor Wilcoxon + §21 verdict + `--relax_gate` option (commits `c040247`, `9f9e6f5`, `ee74f97`)
- `slurm/run_sheu2024.slurm`, `slurm/run_sheu_gate.slurm` (baseline)
- `slurm/run_sheu2024_aux.slurm`, `slurm/run_sheu_aux_gate.slurm` (Track A)
- `slurm/build_pseudotubes_sheu_1hr.slurm`, `slurm/run_sheu2024_1hr.slurm`, `slurm/run_sheu_1hr_gate.slurm` (Track B)
- `slurm/run_sheu_gate_celltype.slurm` (Track C)

### Configuration (narrowed, post-overfit fix)
- `embed_dim = 32` (was 128)
- `attention_hidden_dim = 16` (was 64)
- Slim 8-class label encoder (PBS at idx 0; replaces the 91-d head with 83 unused logits)
- Stage 1: 30 epochs @ lr=0.003
- Stage 2: 40 epochs @ lr=0.0005

### Reports (all on `main`)
- `reports/sheu2024/AXIS_GATE_VERDICT.md` — 3hr narrowed baseline (RED)
- `reports/sheu2024_aux/AXIS_GATE_VERDICT.md` — Track A: adapter aux head (RED)
- `reports/sheu2024_1hr/AXIS_GATE_VERDICT.md` — Track B: 1hr subset (RED)
- `reports/sheu2024_celltype/AXIS_GATE_VERDICT.md` — Track C: direction_mode=cell_type (RED)
- `reports/sheu2024_overnight_summary.md` — full side-by-side comparison

---

## The four runs, side by side

Each cell is **mean per-donor bias across 3 seeds (sign-agreement %)**.
Bias = scalar projection of cytokine-tube centroids in PBS-RC space; larger = bigger geometric effect.

| Axis | Kind | Baseline 3hr | A — adapter aux | B — 1hr | C — cell_type |
|---|---|---|---|---|---|
| `LPS — TNF` | **MUST** | +0.47 (67%) | **+1.10 (78%)** | +0.15 (67%) | **−0.43 (67%)** |
| `polyIC — IFNb` | **MUST** | +0.19 (72%) | **+2.54 (100%)** | +0.81 (100%) | +2.16 (100%) |
| `LPS — IFNb` | SHOULD | +0.61 (89%) | **+3.07 (100%)** | +0.54 (83%) | +2.10 (100%) |
| `P3CSK — CpG` | SHOULD | +1.03 (67%) | +1.52 (56%) | +0.32 (83%) | **+2.29 (100%)** |
| `LPSlo — P3CSK` | SHOULD | +1.72 (67%) | +2.40 (78%) | n/a | **+2.31 (100%)** |
| `P3CSK — IFNb` | MUST-NOT | +0.66 (67%) | **+4.19 (100%)** | +0.56 (83%) | +1.53 (100%) |
| `CpG — IFNb` | MUST-NOT | +0.81 (100%) | +3.11 (78%) | +0.57 (83%) | +2.29 (100%) |
| `TNF — IFNb` | MUST-NOT | +0.50 (83%) | **+3.00 (100%)** | +0.59 (83%) | +2.31 (100%) |

**Strict §21 verdict on all four runs:** RED (0/2 MUST, 0/3 SHOULD, 0/3 MUST-NOT violations). Strict §21 alpha = 0.05 with n=3 donors is mathematically unreachable (min Wilcoxon p = 1/8 = 0.125).

**Relaxed-gate verdict on all four runs:** RED. Pass criterion required raw p ≤ 0.05 in every seed (also unreachable at n=3). The interesting reading is the *pattern of effect sizes*, not the pass/fail column.

---

## What we learned from each run

### Baseline 3hr (narrowed) — `reports/sheu2024/`
First clean read after fixing the overfit (Stage 2 loss had collapsed to 0.005 on 230 tubes). Effect sizes now interpretable: positive across all axes, **CpG—IFNb (MUST-NOT) is the strongest signal**, polyIC—IFNb (MUST) is the weakest with one seed flipping sign. **Dominant latent axis is "stim vs PBS"; everything projects positively onto it.**

### Track A — adapter aux head
The encoder *is* steerable. Adding an auxiliary classification head that predicts the signaling adapter (MyD88_only / TRIF_only / MyD88_TRIF / TNFR / IFNAR / unstim) during Stage 1 made effect sizes **2-5× larger** and sign-agreement on polyIC—IFNb improved from 50% to 100%. **But MUST vs MUST-NOT discrimination didn't follow** — P3CSK—IFNb (MUST-NOT) became the *strongest* signal in the entire table at +4.19. The encoder learned receptor structure but kept it as a secondary axis; the dominant axis stayed "stim vs PBS".

### Track B — 1hr subset
Earlier ≠ cleaner. Effect sizes ~3× smaller than baseline. LPSlo data isn't present at 1hr. 1hr appears to be *too early* — cytokine responses haven't fully manifested. No MUST vs MUST-NOT separation.

### Track C — direction_mode=cell_type
T-specific direction vectors give perfect 100/100/100 sign-agreement on **every** axis, both MUST and MUST-NOT. Kills discrimination by amplifying the common direction uniformly. The one exception (LPS—TNF going negative) is interpretable but doesn't help direction inference.

---

## Cumulative evidence on cascade direction (8 failed checks now)

1. Algebraic (`geo` asymmetry score `bias − bias` is antisymmetric by construction; §20.1)
2. Literature (49% correct direction on 39 documented Oesinghaus pairs = chance; `reports/cascade_pairs/literature_review.md` §8)
3. Architectural (Stage 3 CA-only on Oesinghaus seeds 42 + 123 — SA/CA mechanism works ~4-nat entropy gap, no val AUC gain; `reports/v2_sanity_check/stage3_ca_oesinghaus_results.md`)
4. Sheu 3hr **(first gate run, the all-q=1 underpowered version)**
5. Sheu 3hr narrowed (first non-overfit; CpG—IFNb dominant)
6. Track A adapter aux (encoder steerable but wrong dominant axis)
7. Track B 1hr (signal weaker, no separation)
8. Track C cell_type (signal uniform across MUST and MUST-NOT)

---

## The decision moment (where the session ended)

Claude recommended **closing the cascade-direction line** and finalising Path A (Oesinghaus axis-discovery writeup) with the accumulated negative-result evidence as a credible limitations section. Yam pushed back:

> "5 failed checks of one specific method doesn't mean signal doesn't exist. If A causes B and B doesn't cause A, there must be some statistical signature in the data. Maybe the data is wrong, maybe the method is wrong, maybe the preprocessing. We have 3 months for thesis — let's double down on cascade direction. **We might need to rethink basic assumptions.**"

This is correct. Every experiment so far shares the same method bundle:
**encoder + PBS-RC + dot-product readout on per-donor centroids**.
We've varied pieces (training objective, time point, direction vector) but never tested whether the *signal is in the data* in a different form.

---

## Five assumptions we've been bundling that are separable

1. **Direction lives in the encoder's embedding.** Encoder is trained for cytokine classification; its dominant axis is "stim vs PBS" by construction. We've never tested direction in **raw gene space** or in a representation trained for something else.
2. **Direction = asymmetric centroid projection.** We compute `mean` then project. But cascade direction is fundamentally about **heterogeneity** — if A→B, A-tubes should contain *both* A-primary and B-cascade subpopulations; B-tubes contain only B-direct. **Variance / bimodality is asymmetric in a way means aren't.**
3. **Cytokine identity is the right grouping.** Cascade biology is at **pathway** level — LPS and Pam3CSK4 both signal through MyD88. Maybe direction sits at pathway level, not cytokine level.
4. **Static embeddings are sufficient.** Single-snapshot data still has **training dynamics** — the order in which the model learns to discriminate A from B. We have the §19 confusion-pair tensor infrastructure but never applied it to Sheu.
5. **Mean-centering on PBS is the right baseline.** PBS-RC subtracts the resting centroid per cell type. But A-tube cells "still in transit" carry cascade information *in the deviation*, which PBS-RC removes by design.

---

## Three concrete experiments for the next session

Each tests a **different** assumption. None requires retraining — all run on existing 3hr narrowed checkpoints. Total cost ~5 days; runs in parallel.

### Exp 1 — Within-tube variance / bimodality readout (tests assumption #2)
Replace `µ_{A,T} · û_B` with `Var_{cells ∈ A-tube, type T}(h_i · û_B)`. Prediction: if A→B, A-tube cells in T have higher variance along û_B than B-tube cells have along û_A — because A-tubes contain both pre-cascade and post-cascade subpopulations. **Cost: ~1 day.**

### Exp 2 — No-encoder gene-set readout (tests assumption #1)
Curate adapter-specific gene sets from Sheu's 500-gene panel:
- TLR4-MyD88 primary: `Nfkbia, Tnf, Il1b, Cxcl1, Cxcl2`
- TLR3-TRIF primary: `Ifnb1, Isg15, Ifit1, Mx1, Rsad2`
- TNFR cascade: `Tnfaip3, Nfkbid, Tnf` (autocrine)
- IFNAR cascade: `Stat1, Irf7, Oas1, Mx2`

For each (tube, cell), compute mean expression of each gene set. Then for each pair (A, B): is the fraction of A-tube cells with high B-set expression *greater* than the fraction of B-tube cells with high A-set expression? **Bypasses the encoder entirely.** If it works, encoder was the bottleneck. If it fails, the cascade signature genuinely isn't in 3h gene expression. Either result is informative. **Cost: ~2 days.**

### Exp 3 — Confusion-dynamics asymmetry (tests assumption #4)
We have `dynamics.pkl` from the narrowed runs. Build the pairwise confusion tensor `C(A, B, t)` from cell-level softmax across training time. Test: at which epoch does the model first separate A from B? Does the separation epoch for A→B differ from B→A? Cascade direction may sit in the *order* the model learns the discrimination, not the final embedding. **Cost: ~2 days, no retraining.**

---

## Cluster job IDs (for archaeology if needed)

| Job ID | Description | State |
|---|---|---|
| 30645700 | Sheu pseudotube build (3hr) | COMPLETED |
| 30645916, 30646827 | Initial training (cancelled / race condition) | CANCELLED |
| 30651963 | Clean narrowed retrain, 3 seeds | COMPLETED |
| 30652029 | Baseline gate on narrowed checkpoints | COMPLETED |
| 30652107 / 30652110 | Track A train + gate | COMPLETED |
| 30652111 / 30652112 / 30652113 | Track B build + train + gate | COMPLETED |
| 30652114 | Track C gate | COMPLETED |

All result directories are under `/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/`.

---

## Status of the rest of the project

- **Path A (Oesinghaus axis-discovery writeup): UNCHANGED.** 121 axes, ~50% lit-supported, publication-grade. Independent of everything above.
- **v2 two-layer attention: PAUSED, not retired.** Architecture preserved in code (`cytokine_mil/models/cytokine_abmil_v2.py`).
- **Zhang 2022 (§2.6): DEFERRED.** Was a secondary dataset; not on critical path now.
- **CLAUDE.md is current** through 2026-05-22 framing; §0 status block, §2.5 Sheu specs, §21 axis-discovery gate spec are all up to date. Will need an update once direction is decided.

---

## When you start the next session

1. Read this file and `reports/sheu2024_overnight_summary.md` to refresh context.
2. Decide which of Exp 1 / 2 / 3 to start with (or propose a 4th angle).
3. The cluster pipeline is fully working: edit local → `git push` → `cluster_cmd "cd cytokine-mil && git pull"` → `sbatch slurm/...`. Pseudotube builds ~3 min, training ~3-5 min/seed, gate ~3-5 min.
4. Note that all three proposed experiments run on the **existing** narrowed 3hr checkpoints — no retraining required for Exp 1 or Exp 3; Exp 2 doesn't use checkpoints at all.
