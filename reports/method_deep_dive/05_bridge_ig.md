# M5 — Bridge: binary AB-MIL + Integrated Gradients → `S_X`

Real code: `scripts/run_binary_ig_probe.py`. **Input:** the shared frozen binary models (M4)
+ pseudo-tubes. **Output:** `binary_ig.parquet` — the per-stimulus gene signatures `S_X`.

## 1. The job
Turn each trained binary model (stimulus X vs PBS) into a **gene signature `S_X`** = the
top-50 genes the model relies on to call "X". This is the **bridge** between the model (M4)
and the direction metric (M7): cross_asym needs each stimulus's signature.

## 2. Integrated Gradients (IG) in three sentences
IG attributes a model's output (the logit for "X", the positive class = index 0) back to the
input genes. It integrates the gradient of that logit w.r.t. the input along a straight path
from a **baseline** cell to the real cell. Genes with large integrated gradient are the ones
that moved the model from "looks like PBS" to "looks like X".

Per gene `g`:
`IG_g = (x_g − base_g) · (1/m) Σ_{k} ∂logit_X/∂x_g |_(base + α_k (x − base))`,
with **m = 20** path points at the **midpoints** `α_k = (k − 0.5)/m` (the midpoint rule).

## 3. The exact procedure (`run_binary_ig_probe.py`)
- Load the binary model for X (HPs inferred from state-dict shapes — robust to the variant).
- **Baseline = per-gene mean over ~10 PBS tubes** (the resting reference).
- For each of ≤ `max_tubes_per_cytokine=10` X-tubes: compute IG per cell (20 midpoint steps).
- **Average IG per gene** across all cells × tubes; rank descending → `rank_ig` (0 = highest).
- `S_X` = **top_n = 50** genes by `rank_ig`.
- Write `binary_ig.parquet`, **long format**, columns:
  `cytokine, gene, ig, mean_expression, rank_ig` (one row per (cytokine, gene)).

## 4. Why IG (not raw differential expression)?
Raw DE (X-mean − PBS-mean) tells you which genes *changed*. IG tells you which genes the
**trained model uses** to discriminate X from PBS — attribution of the classifier's decision,
through the learned representation. (They overlap, but IG ties `S_X` to the very model the rest
of the pipeline depends on, and the shared frozen encoder makes the `S_X` comparable across
stimuli.) **PBS baseline:** `S_X` should mean "what makes X different from *resting*", so
integration starts at the PBS state; an all-zeros baseline would attribute against "no
expression at all", which is not the right reference here.

## 5. The key diagnostic — `S_polyIC` collapses onto `S_IFNb`
This is where the M1 caveat becomes concrete (and is the project's one consistent failure).
- **polyIC biology:** TLR3→IRF3 makes a few IRF3-direct genes (`Ifnb1, Cxcl10, Ifit2/3`) **and**,
  via autocrine IFN-β, a large **ISG** response. **IFN-β biology:** IFNAR → the *same* ISGs.
- The autocrine **ISGs are the strongest differential signal** in polyIC tubes, so polyIC's
  top-50 IG genes are **ISG-dominated** — nearly identical to `S_IFNb`. The upstream-specific
  IRF3-direct genes (lower-expressed) get crowded out of the top-50.
- ⇒ `S_polyIC ≈ S_IFNb`. When `S_a ≈ S_b`, cross_asym loses its directional grip (M7): the
  sign is then driven by raw strength, and the pure/strong IFNAR agonist (IFN-β) out-engages
  polyIC's autocrine loop → **cross_asym sign flips** → polyIC→IFNb mispredicted at all time
  points (M8). LPS→IFNb works (LPS carries a broader non-ISG program, so `S_LPS ≠ S_IFNb`).
- **Proposed fix (M9):** re-derive `S_polyIC` excluding ISGs / up-weighting IRF3-direct genes;
  predicted to restore the correct sign — a clean "we understand our one failure" experiment.

## 6. I/O summary
**IN:** `model_X.pt` (frozen binary models), manifest, `hvg_list.json`.
**PROC:** PBS-mean baseline → 20-step midpoint IG per cell → average + rank per gene.
**OUT:** `binary_ig.parquet (cytokine, gene, ig, mean_expression, rank_ig)`; `S_X` = top-50.

## 7. Worked numbers (real IG run, planted `CytA→CytB` synthetic)
Ran `cascadir` end-to-end locally (`preprocess → 4 tubes → 5-epoch encoder → 40-epoch
binary CytA-vs-PBS & CytB-vs-PBS → derive_signatures(top_n=20)`; `signatures.py`).
Planted: UP=`gene0–9` (CytA only), DOWN=`gene10–19` (both; CytA relays), noise=`gene20–59`.

| signature | UP in top-20 | DOWN | noise | UP rank positions |
|---|---:|---:|---:|---|
| `S_CytA` (upstream)   | 5 | 4 | 11 | #2,#3,#4,#15,#16 (**high**) |
| `S_CytB` (downstream) | 3 | 5 | 12 | #10,#14,#19 (**ig≈0.001**) |

Top of `S_CytA`: gene18(DOWN,.0074), gene12(DOWN,.0060), **gene0(UP,.0057)**,
gene8(UP,.0048), gene6(UP,.0036). Top of `S_CytB`: gene18(DOWN,.0111), gene12(DOWN,.0073),
gene15(DOWN,.0039) — UP-block only at the bottom.

**Read:** the **upstream** signature carries **both** programs; the **downstream** one carries
**only its own**. CytB never elevates the UP-block (stays at baseline) ⇒ UP genes are
non-discriminative for CytB-vs-PBS ⇒ ~0 IG ⇒ excluded. This asymmetry (UP high in
`S_upstream`, absent in `S_downstream`) is exactly what makes `cross_asym > 0` for the
upstream (M7). Noise leaks at ig≈0.001 vs program 0.005–0.011 ⇒ **magnitude** separates them;
`top_n` is a knob (top-6 here ≈ pure program).

**Collapse knob (→ polyIC, M9):** UP and DOWN share one `program_rate` here, so UP survives in
`S_CytA`. Make the relayed DOWN program ≫ the UP-specific one and DOWN crowds UP out of the top
⇒ `S_CytA ≈ S_CytB` ⇒ cross_asym sign loses its grip. That is the polyIC→IFNb failure (M5 §5).
