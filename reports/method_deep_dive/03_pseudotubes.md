# M3 — Preprocessing & pseudo-tubes (Step 0)

**Job:** turn a raw `cells × genes` count matrix into the labeled **bags** the model eats.
Real code: `scripts/build_pseudotubes_sheu2024.py` (Oes analogue: `build_pseudotubes.py`).

## A. Preprocessing — make cells comparable (lines 473–475)
1. `sc.pp.normalize_total(adata, target_sum=1e4)` — rescale each cell so its counts sum to
   10,000. Kills the sequencing-depth artifact (a deeper-sequenced cell isn't "more
   active", it just has bigger raw counts). Each cell becomes *relative composition*.
2. `sc.pp.log1p(adata)` — `x → log(1+x)`. Compresses the huge dynamic range, stabilizes
   variance. Standard.
3. **HVG** — Oesinghaus selects 4000 highly-variable genes; **Sheu skips it** (its 500-gene
   panel is already curated).

Order: preprocessing runs **before** tube-building, so every tube is already normalized.

## B. Pseudo-tube construction — build the bags (lines 356–428)
Constants (lines 54–56): `N_PER_CELL_TYPE=30`, `MIN_CELLS_THRESHOLD=10`, `N_PSEUDO_TUBES=10`.

For each **(donor, stimulus)**:
- **eligible cell types** = those with ≥ 10 cells (line 362–366); skip the condition if none.
- make **10** pseudo-tubes (line 375). Each tube (`_sample_one_tube`, 405–428):
  - for each eligible cell type, sample `min(30, available)` cells **without replacement**
    (412–414) → *stratified by cell type*;
  - `sc.concat` them, then **shuffle row order** (418–420) so cell type is **not** implicit
    in row position — the model can't cheat off the index;
  - cast to `float32`.
- write `.h5ad` + a manifest row `{path, donor, cytokine(label), n_cells,
  cell_types_included, tube_idx}` (387–396).

A tube ≈ `30 × (#eligible cell types)` cells. Tube composition varies by condition (some
cell types fall below threshold) — **on purpose**.

Cell types come from `assign_cell_types_global_leiden` (line 240): Leiden clustering on
resting (0 h) cells → `mac_c0..c3`; post-stim cells assigned to the nearest cluster
centroid. Same label space everywhere, so stratification means the same thing across donors.

## C. The deliberate NON-choices (scientific decisions)
1. **No z-score per gene.** Z-scoring removes each gene's *absolute* level, but absolute
   level is informative — let the encoder learn how to weight genes, don't pre-flatten.
2. **No perturbation / log2FC-vs-PBS in the input.** Representing a cell as its deviation
   from resting (PBS) *injects a prior*: it pre-decides that resting is the right reference
   and that deviation is what matters. Philosophy: feed raw (normalized) expression and let
   the model **discover** what's informative. PBS *does* return later — but as **explicit,
   auditable steps** (IG baseline, M5; cross_asym normalization, M7), never baked into the
   model's input.
3. **Variable tube sizes preserved — do NOT equalize.** Stimuli that kill cells (apoptosis)
   or drive proliferation make tubes different sizes; that size difference is *signal*.
4. **Cell-type labels dropped before the model.** Used only for (a) stratified sampling
   here and (b) post-hoc analysis (M6/M7). Enforced by the row-shuffle. The model classifies
   the stimulus from **expression alone**.

### Where PBS actually appears (resolves the apparent contradiction)
"No PBS in the *input*" does **not** mean PBS is unused — it means we never bake a
PBS-relative transform into what the model *sees*. PBS returns later as **explicit,
auditable steps**:

| stage | uses PBS? | how | acts on |
|---|:--:|---|---|
| model input (M3) / training (M4) | **no** | — | raw normalized expression |
| Bridge — IG (M5) | yes | PBS-mean = the IG **baseline** (integration start) | raw expression |
| Path A — latent geometry (M6) | yes | **PBS-RC**: `h̃ = h − µ_PBS, cell_type` | **embeddings** (post-encoder) |
| Path B — cross_asym (M7) | yes | **score-level**: subtract `s_PBS` (`pathway_audit.py:464–467`) | signature **scores** (no encoder) |

**PBS-RC** (embedding-centroid subtraction) is a **Path A** construct (§20.1). The current
direction spine (**Path B / cross_asym**) does **not** use embedding PBS-RC — it uses the
simpler score-level subtraction. Same intent (remove resting baseline), different stage & space.

## D. Why this shape → MIL
A pseudo-tube is a **bag** of cells (instances) with **one** bag-level label (the stimulus).
That is exactly the multiple-instance-learning setup the model in M4 consumes.
