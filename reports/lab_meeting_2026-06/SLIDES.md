# Slide texts — VERBATIM (do not rephrase)

Lab meeting. Audience: comp-bio MSc/PhD + PI; they know scRNA-seq + ML, not this project.
Each slide below = the exact title + body text to typeset. Figures/tables to place are named
in `[[ ... ]]`. ~14 slides + 2 backup.

---

## Slide 1 — Title
**Reading cytokine cascades from a single-cell snapshot**
Which cytokines are coupled, and which one is upstream — without a time course.

Cascades detection

---

## Slide 2 — The biology
**Cytokines talk to each other in cascades**

- Cytokines are the immune system's signaling molecules.
- They act in **cascades**: stimulus A makes a cell secrete cytokine B, which acts on other cells.
- Two questions about any cytokine pair (A, B):
  - **Coupling** — are they part of the same signaling axis at all?
  - **Direction** — who is upstream (A → B or B → A)?

[[ Claude Design: draw a simple cascade cartoon — stimulus A → cell secretes B → B acts on neighbor cells. ]]

---

## Slide 3 — The hard part
**Direction from one frame in time**

- Cascades are usually read from **time courses** (B rises after A) — expensive, often unavailable.
- We have **single-cell snapshots**: one time point, thousands of cells, ~hundreds–thousands of genes.
- **Coupling** from a snapshot is plausible; **direction** from a snapshot is the novel, hard claim.
- Our bet: an upstream stimulus's cells carry **both** programs (its own + the autocrine downstream one); the downstream ligand's cells carry **mainly their own**. That asymmetry is readable in one frame.

---

## Slide 4 — Data
**Three deliberately different datasets**

- A win on all three = a property of the *method*, not one system.
- Human/mouse, blood/macrophage/lymph-node, broad/targeted gene panels, ex vivo/in vivo.

[[ TABLE: table_datasets.xlsx (or table_datasets.md) ]]

---

## Slide 5 — Representation
**Pseudo-tubes: a bag of cells per condition**

- We don't model single cells in isolation — we build **pseudo-tubes**: bags of cells, stratified by cell type.
- One pseudo-tube = one example; its **label** = the stimulus applied.
- This is **multiple-instance learning (MIL)**: the label is on the bag, the model learns which cells carry the signal.
- Split is **by donor**, never by tube (tubes from one donor are correlated).

[[ Claude Design: draw a "bag of cells" → label icon (pseudo-tube → 'IL-2'). ]]

---

## Slide 6 — Method, step 1
**Find each cytokine's specific gene signature**

- Train one **binary classifier per cytokine**: "this tube is cytokine X" vs "PBS (resting)". (Attention-MIL on a shared, frozen cell encoder.)
- Open the classifier with **Integrated Gradients** (attribution from a PBS baseline): *which genes did it use to tell X from resting?*
- Top-50 genes = **`S_X`**, X's specific signature — the genes unique to X, **not** the activation genes every cytokine shares.

[[ Claude Design: draw the flow — binary model → Integrated Gradients → top-50 gene list S_X. ]]

---

## Slide 7 — Method, step 2 (the core idea)
**One matrix → coupling and direction**

- Build the **cross-engagement matrix** in gene space:
  **`M[a,b]` = how much cytokine a's cells express cytokine b's signature `S_b`** (PBS-normalized).
- It is *directed*: `M[a,b] ≠ M[b,a]`. Read its two halves:
  - **Coupling = `M[a,b] + M[b,a]`** (symmetric) — are a and b mutually engaged in each other's specific programs?
  - **Direction = `M[a,b] − M[b,a]`** (antisymmetric) — who is upstream? (upstream carries both programs.)
- No encoder embedding here — just gene expression in the specific dimensions.

[[ Claude Design: draw M as a small heatmap, then split into M+Mᵀ (coupling) and M−Mᵀ (direction). ]]

---

## Slide 8 — Result 1 (headline)
**Direction is recovered on all three datasets**

- On known cascades: **88% (15/17) · 86% (6/7) · 83% (5/6)** correct direction — from a single frame.
- The **antisymmetric** statistic does the work: a symmetric control on the *same data* is at chance (47% / 33%).

[[ FIGURE: fig_direction_accuracy.png ]]

---

## Slide 9 — Result 1, the biology
**It recovers textbook cascades**

- Immune Dictionary (mouse, in vivo): the NK-cell **IFN-γ axis** and **TNF → IL-6**, all correct direction.
- These are cascades the original paper documents — recovered here from one snapshot, blind to the labels.

[[ FIGURE: fig_cascade_examples.png ]]

---

## Slide 10 — Result 2 (the recent turn)
**Where to measure coupling? Two paths**

- Coupling can be measured two ways, with mirror-image strengths:
  - **Latent space** (the encoder embedding) — rich, but dominated by the *shared activation* program every cytokine induces. Works on broad human data; **had no power on the targeted mouse panel**.
  - **Gene signatures** (the `M` above) — the specific dimensions; **recovered the textbook IFN cascades the latent-space method completely missed**.

[[ FIGURE: fig_sheu_coupling_win.png ]]

---

## Slide 11 — Honest limitations
**What is NOT solved yet**

- **Direction is validated on *known* pairs, not blind discovery.** Small n (17 / 7 / 6 pairs).
- **Coupling gate over-calls on broad data** (it admits ~80% of pairs) — and our significance test is **over-powered**: with thousands of cells, almost anything is "significant."
- The fix is the same in both places: **test at the donor level** (≈10 independent donors), not the cell level. That's the current frontier.
- One understood failure: polyIC's signature collapses onto the interferon program, so its *direction* can flip.

---

## Slide 12 — Summary
**What we have**

- A method that reads **cascade direction from a single snapshot** — validated at ~85% on known cascades across human/mouse, ex vivo/in vivo.
- The trick: **discover each cytokine's specific genes, then read one cross-engagement matrix — its symmetric half is coupling, its antisymmetric half is direction.**
- Measuring coupling in **specific genes** (not the latent space) rescued the dataset where the original approach failed.

[[ TABLE: table_results_grid.xlsx (or table_results_grid.md) — method × dataset grid ]]

---

## Slide 13 — Next
**Where this goes**

- **Donor-level statistics** for coupling and direction (removes the over-power) — the immediate next step.
- A **degree-corrected coupling gate** (so a few "hub" cytokines don't look coupled to everything).
- Run the signature-coupling on the third dataset; then **blind discovery** of novel axes + directions.
- The endgame: rank candidate cascade *chains*, and hand the top predictions to wet-lab.

---

## Slide 14 — Backup: how the project evolved
**(only if asked)**

- The project began as an analysis of MIL **training dynamics** (which cytokines are learned fast vs slow).
- It evolved into the current **gradient-attribution + gene-geometry** method, which is what these results use.
- Mentioned so the change of framing is clear; the dynamics analysis is not part of the current pipeline.

---

## Slide 15 — Backup: Integrated Gradients in one line
**(only if asked)**

- For each cell, walk a straight line from the **PBS baseline** to the real cell in 20 steps; average the model's gradients along the way; multiply by (cell − baseline).
- Gives a per-gene attribution that **doesn't saturate** and is **relative to resting** — exactly "which genes' departure from rest made the model call cytokine X."

[[ FIGURE (optional): fig_coupling_vs_pathA.png — the two coupling notions disagree (ρ=0.29), for the "two paths" point. ]]
