# Session summary — the coupling-gate fix (2026-06-19)

A self-contained writeup so a fresh session can learn what we did, **why**, what worked,
what didn't, and what changed. Scope: the **coupling-existence gate** of the
signature-space method. The cascade-**direction** method (`cross_asym`) is unchanged.

---

## TL;DR

- **Problem:** the coupling gate (decides "is pair (A,B) coupled?") over-called badly —
  ~79% of Oesinghaus pairs "coupled", hub-dominated (IL-15 everywhere). It didn't
  discriminate, so it couldn't gate which pairs deserve a direction call.
- **Two causes:** (1) the cell-level null is **over-powered** (cells aren't independent;
  unit = donor); (2) **hub/degree bias** — a broadly-engaged signature looks coupled to
  everything.
- **Fix:** **(1) donor-level null** (effective N = donors, sign-flip test) + **(2) degree
  (hub) correction** (double-center the coupling matrix → pair-specific residual). The
  degree correction is the decisive piece.
- **Result:** Oesinghaus over-call **77%→31%** while known-cascade recall *rose*
  **8/17→11/17** (~2.1× enrichment). Generalized to Sheu's targeted panel at the cell
  level (kept the 2/2 IFN cascades, suppressed all negatives, over-call ~80%→~40%).
- **Boundary:** the donor-level *null* needs ~8+ donors → Oesinghaus only; few-donor data
  (Sheu 4, ID 3) uses the cell-level degree correction.
- **Side findings:** keep the encoder (DE ≠ IG); adopt panel-residualized signatures
  (`IG_vsPanel`) as the default specific signature.
- **Shipped:** `cascadir` `signature_coupling(..., degree_correct=True)` (default on).

---

## 0. Where we started & what triggered this

The method has two halves, computed from each cytokine's **discovered signature** `S_X`
(top-50 genes by Integrated Gradients on a binary stimulus-vs-PBS AB-MIL):
- **Coupling (existence):** are A and B related? — the **gate** (yes/no).
- **Direction:** if coupled, who's upstream? — `cross_asym` (validated: 88/86/83% on
  Oes/Sheu/ID; this was *not* touched this session).

Trigger: four doubts about the method — (1) signatures selected vs control, not vs other
cytokines, so are they specific? (2) top-50 is arbitrary; (3) coupling has no universal
solution; (4) the method (encoder + IG) may be over-complicated. We reframed these into
one testable thread: **is the gate's over-call caused by non-specific signatures, and do
we even need the encoder?**

---

## 1. Experiment 1 — signature-definition ablation (2×2)

**Idea/why:** hold the whole pipeline fixed and swap *only* how `S_X` is defined, as a
2×2: **{IG, DE} × {vs-PBS, vs-panel}**. IG = Integrated-Gradients (current); DE = plain
differential expression (no encoder). vs-PBS = current contrast; vs-panel = residualize
against the cross-cytokine mean (`Δ_X − mean_over_cytokines Δ`) to strip the shared
activation program while keeping partner-shared genes. One CPU job, no retraining, on
Oesinghaus (`binary_ig_all24`, 24 cytokines, 276 pairs, 17 labeled direction pairs).
Code: `scripts/run_signature_ablation.py` (+ `analyze_signature_ablation.py`). Job 30870957.

**Results (direction acc / over-call / hub):**

| variant | direction | coupled_frac | hub (max cyt in top-20) |
|---|---|---|---|
| IG_vsPBS (current) | 14/17 = 82% | 76.5% | IL-15: 10 |
| IG_vsPanel | 14/17 = 82% | 77.9% | IL-15: 9 |
| DE_vsPBS | 10/17 = 59% | 58.7% | IFN-β: 7 |
| DE_vsPanel | 9/17 = 53% | 46.7% | IFN-β: 6 |

Plus mean **Jaccard(DE_vsPBS, IG_vsPBS) = 0.11**.

**Findings:**
1. **Keep the encoder (DE ≠ IG).** DE and IG select almost-disjoint genes (Jaccard 0.11)
   and DE direction collapses to ~chance (59% vs 82%). The "drop the model, use DE"
   simplification is dead — IG attribution does real work a fold-change ranking doesn't.
2. **IG_vsPanel is a free, more-specific upgrade** — identical direction, slightly less
   hub-dominated, removes shared-activation genes. Adopt as the default signature.
3. **Specificity ALONE does not fix the gate.** Panel-residualizing left coupled_frac
   essentially unchanged (76.5%→77.9%). Conclusion: the over-call is not (mainly) a
   non-specific-signature problem — it's a **null/statistic** problem (Cause 1+2 below).
   (Note: IG_vsPBS reproduced 14/17 not the published 15/17 — the one differing pair is a
   near-zero coin-flip, consistent.)

---

## 2. Why the gate over-called — the two causes

The statistic: `s(a,S_b)` = mean expression of B's signature genes in A's cells;
`M[a,b] = s(a,S_b) − s(PBS,S_b)`; `coupling(a,b) = M[a,b] + M[b,a]` (symmetric).
Computed **per cell type** (A's monocytes vs PBS's monocytes, etc.), then **median across
cell types** — this controls for cell-type composition. The gate compares coupling to a
**random-gene-set null**.

- **Cause 1 — over-power.** We pool *thousands* of cells per (cytokine, cell-type), so
  every mean is razor-precise → the random-gene null is a thin spike → almost any nonzero
  coupling beats it → p≈0 for ~everything. The math assumes N = thousands of independent
  cells, but cells from one donor aren't independent — **the unit is the donor (~10).**
- **Cause 2 — hub/degree bias.** If B's signature contains genes many cytokines induce
  (a "promiscuous" signature), `s(a,S_b)` is high for *every* A → B looks coupled to
  everyone. IL-15 = high row + high column → couples to half the panel. This is in the
  coupling **value**, not the null — it survives even a perfect null.

Two diseases → two fixes.

---

## 3. Experiment 2 — donor-level null (attacks Cause 1)

**Idea:** make the donor the unit. Compute coupling **per donor** (each centered by that
donor's own random-gene baseline → per-donor "excess"), then test across the ~10 donors
with a **one-sided sign-flip permutation test**: is the pair consistently positive across
donors? Power is capped at ~10, not thousands. Code:
`cytokine_mil/analysis/signature_coupling.py` (`donor_excess_matrix`, `donor_coupling_test`,
`_signflip_p`, `_bh_fdr`); driver `scripts/run_donor_coupling_null.py`.

**Result (Oesinghaus, 10 train donors, q<0.10):**
- Over-call **77% (cell) → ~53% (donor, raw)** — real but partial.
- **IL-15 still the top hub** (≈9 of top-20); recall ~8/17.

**Why only halfway:** the donor null asks *"is this consistent?"* — and **a hub is
consistently non-specific.** IL-15 couples to everything in every donor, so it sails
through a consistency test. Cause 1 fixed; Cause 2 untouched.

---

## 4. Experiment 3 — degree (hub) correction (attacks Cause 2) — THE FIX

**Idea:** remove each condition's overall "strength" from the coupling matrix. Build the
symmetric coupling matrix C (diagonal excluded) and **double-center** it:
```
R[i,j] = C[i,j] − d_i − d_j + g     # d_i = mean off-diagonal coupling of i (its "degree")
                                     # g   = grand off-diagonal mean
```
A hub (large `d_i`) gets its pairs deflated; a specific pair (high beyond what the two
nodes' averages predict) survives. **Crucially symmetric → it changes only coupling
(existence), never `cross_asym` (direction).** It's degenerate for <3 conditions (residual
collapses to 0), so it's a no-op there. Code: `_degree_center`,
`donor_residual_coupling_matrix`, `cell_coupling_degree`.

**Result (Oesinghaus donor + degree, q<0.10):**

| mode | variant | over-call | recall | enrichment |
|---|---|---|---|---|
| raw | IG_vsPanel | 52.9% | 8/17 | 0.9× (none) |
| **hub** | **IG_vsPanel** | **30.8%** | **11/17** | **2.1×** |

Over-call nearly halved **and** recall rose — i.e. the gate went from not-discriminating
(raw recall ≈ base rate) to 2.1× enriched for known cascades. The top is now real biology
(IL-15/IL-2 #1, IFN cluster). IG_vsPanel is the best operating point. Job 30874970.
*Residual issue:* IL-15 only partly de-hubbed (top-20 9→7) but its survivors are the
genuine NK-axis pairs (IL-2, IL-12, IFN-γ/β/ω).

---

## 5. Experiment 4 — generalization on Sheu + the donor-count boundary

**Donor-level on Sheu FAILED structurally.** The Sheu pseudotube manifest is **3hr-only**
(don't pair it with the 5hr binary_ig — that filters out all stimulated cells). At 3hr only
~4 effective pseudo-donors exist, so only **6/21 pairs reach ≥3 donors**, and the IFN
cascades (LPS–IFNb, PIC–IFNb) + all MUST-NOTs (involve IFNb) are **untestable**. Lesson:
**the donor-level null needs ~8+ well-covered donors (Oesinghaus-scale).** Few-donor data
(Sheu 4, ID 3) can't support it. Jobs 30880974/30881068.

**Cell-level degree correction on Sheu WON** (all 21 pairs testable; the salvage). Job
30881261. `scripts/run_cell_degree_coupling.py`.

| variant | mode | over-call | MUST recall | MUST-NOT coupled |
|---|---|---|---|---|
| IG_vsPanel | raw | 16/21 (76%) | 5/5 | 0/3 |
| IG_vsPanel | **hub** | **9/21 (43%)** | **4/5** | **0/3** |

Under degree correction the textbook IFN cascades **survive and top the list**
(**PIC–IFNb is #1**, LPS–IFNb top-5), all 3 negatives are suppressed (and sit at the
bottom), over-call halved. The one MUST it drops is **LPS–TNF** — both partners are NF-κB
hubs, so degree correction removes their pairwise residual (the known-hard overlapping
pair; mechanistically expected, not a failure of the IFN result).

---

## 6. What changed in the method

- **Direction (`cross_asym`): unchanged.** Degree correction is symmetric → does not touch
  the antisymmetric direction half. The 88/86/83% direction result stands.
- **Coupling-existence gate: changed.** Now (a) uses **panel-residualized IG signatures**
  (`IG_vsPanel`) as the default specific dimensions, and (b) applies a **degree/hub
  correction** before gating, with a **donor-level null** when ≥~8 donors exist and a
  **cell-level degree-corrected gate** as the few-donor fallback.
- **Ported to `cascadir`:** `signature_coupling(..., degree_correct=True)` (default on;
  no-op for <3 conditions; donor path + cell null both degree-corrected; keeps
  `coupling_raw` for transparency). Threaded through `CascadeDirection.signature_coupling`.
  +2 tests, all 50 cascadir tests pass. MANUAL §4/§5/§8 + CLAUDE.md §28.2 document it.

## What did NOT change / what we ruled out
- The encoder + IG bridge stays (DE can't replace it).
- `cross_asym` direction method and its results.
- Path A (latent geometry, 121 axes) — independent standing result.

---

## 7. Open challenges (next milestones)

1. **Gate not yet "clean":** ~31% coupled, and the sign-test gauges *consistency* not
   *effect size* → weak-but-consistent pairs inflate `coupled_frac`. Owe it a **magnitude
   floor** (e.g. require `coupling` above a percentile of labeled positives).
2. **No donor-level *direction* null** — direction reliability is still informal (the §27
   cell-level attempt was over-powered). Needed for a novel-pair claim.
3. **Rigor ≠ breadth:** the donor-level null only runs on Oesinghaus (≥8 donors);
   Sheu/ID use cell-level degree (ranking, no honest significance).
4. **Signature specificity is the linchpin** — known collapses (polyIC→IFNb ISG flip; weak
   VEGF; IL-1β→IL-6 on ID) remain unfixed.
5. **Scale debt:** full-panel (all-45) signatures regressed (chunked encoder); blind
   discovery across the whole panel needs retraining signatures with one shared encoder.
6. **Ceiling:** direction ≠ causation (single-snapshot inference; wants wet-lab to close).

**Single best next milestone:** a donor-level direction null + a magnitude-floored gate →
would move Oesinghaus from "recovers known cascades" to "proposes a short list of novel
coupled, directed pairs worth a wet-lab test."

---

## 8. Reproducibility pointers

**Code (research):** `cytokine_mil/analysis/signature_coupling.py` —
`_degree_center`, `donor_excess_matrix`, `donor_residual_coupling_matrix`,
`donor_coupling_test`, `cell_coupling_degree`. Drivers: `scripts/run_signature_ablation.py`,
`run_donor_coupling_null.py`, `run_cell_degree_coupling.py`, `analyze_signature_ablation.py`.

**Code (package):** `cascadir/src/cascadir/signature_coupling.py` (`degree_correct=True`),
estimator `CascadeDirection.signature_coupling`.

**Inputs:** Oes signatures `results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet`
(24 cytokines, reproduces ~88% — NOT the regressed group_u all45); Sheu
`results/sheu_cascade/3hr/binary_ig/binary_ig.parquet` + the 3hr-only pseudotube manifest.
Audited labels `reports/cascade_pairs/cytokine_axes_audited.csv`; Sheu MUST/MUST-NOT from
`cytokine_mil/analysis/eda_pair_benchmark.labeled_pair_status`.

**Outputs:** `results/sig_ablation/oes/{ablation_report.md, analysis/, donor_coupling/}`,
`results/sig_ablation/sheu3hr/cell_degree/cell_degree_report.md`.

**SLURM jobs:** 30870957 (ablation), 30870977 (analysis), 30874970 (Oes donor raw+hub),
30881068 (Sheu donor — boundary), 30881261 (Sheu cell+degree — win).

**Spec:** CLAUDE.md §28.1 (the over-call diagnosis) and §28.2 (this fix, validated).
