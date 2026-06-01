# Cytokine cascade-direction — method bible

Personal, from-first-principles reference (for Yam, not an audience). Built module by
module. Companion to the audience-facing talk report in `reports/group_talk_2026-06/`.

## The map (where am I?)

**Question:** can we read the *direction* of a cytokine signaling cascade (who is
upstream, X→Y) from a **single time-point** single-cell snapshot?

Two independent "paths" in the project:

- **Path A — coupling / axis discovery (standing result).** Latent geometry of an AB-MIL
  encoder → tells you *which* cytokine pairs are coupled (unordered axes).
  Direction-blind by construction. Output: `cytokine_axes.csv`.
- **Path B — direction (this work).** Given a coupled pair, assign *direction* with the
  antisymmetric **`cross_asym`** statistic. Output: `per_celltype.csv` → accuracy.

**Which model feeds what (binary vs multiclass):**
- **Path A** uses a **multiclass** AB-MIL (one model classifying *all* stimuli at once) — its
  embedding geometry gives the coupling.
- **Path B / cross_asym** uses **per-stimulus binary** AB-MILs (each = one stimulus vs PBS) →
  Integrated Gradients on each → the signatures `S_X`. The 88% / 86% direction result is a
  **binary-model** story; the multiclass model is *not* used in cross_asym.

**Headline:** `cross_asym` scores **88%** on Oesinghaus 24h (vs **47%** for the old
symmetric `directional_score`) and **86%** on Sheu BMDM 5h — single-frame, no time leakage.

## Modules

legend: `[x]` taught + written · `[d]` reference drafted (interactive walkthrough pending) · `[~]` in progress

- `[x]` M0 — Orientation / project map (this file)
- `[x]` M1 — Biology & the challenge → `01_biology_and_challenge.md`
- `[x]` M2 — Datasets → `02_datasets.md`
- `[x]` M3 — Preprocessing & pseudo-tubes → `03_pseudotubes.md`
- `[x]` M4 — AB-MIL model + hyperparameters → `04_model_and_hyperparams.md`
- `[d]` M5 — Bridge: binary AB-MIL + Integrated Gradients → `05_bridge_ig.md`
- `[d]` M6 — Path A: latent geometry → `06_path_a_latent_geometry.md`
- `[d]` M7 — Path B: the cross_asym crux → `07_cross_asym.md`
- `[d]` M8 — Experiments & results → `08_experiments_results.md`
- `[d]` M9 — Consolidation & open questions → `09_open_questions.md`

## Glossary (grows as we go)

- **pseudo-tube** — a *bag* of single cells (stratified across cell types) given one
  stimulus label; the unit the model classifies.
- **AB-MIL** — attention-based multiple-instance learning: encode each cell, attention-pool
  to a bag vector, classify the bag.
- **instance / bag** — instance = one cell; bag = one pseudo-tube.
- **S_X** — the discovered gene *signature* of stimulus X (top-50 genes by IG).
- **IG** — Integrated Gradients: attribution of the model's logit to input genes along a
  straight path from a PBS baseline.
- **PBS** — the resting/unstimulated control (training class index 90).
- **PBS-normalized** — subtract the PBS baseline so a score measures deviation from rest.
- **cross_asym** — `s(a, S_b) − s(b, S_a)`; antisymmetric under a↔b → encodes direction.
- **directional_score** — `asym_PA − asym_PB`; symmetric → direction-blind (the old metric).
- **relay cell type** — the cell type that carries the coupling/cascade signal.
- **expected_sign** — benchmark ground truth: `+1` = a_to_b, `−1` = b_to_a, `0` = no
  cascade, null = unknown (pairs stored alphabetically, a < b).

## Canonical code

- model: `cytokine_mil/models/{instance_encoder,attention,bag_classifier,cytokine_abmil}.py`
- §24 metric: `cytokine_mil/analysis/pathway_audit.py::directional_asymmetry_test`
- bridge/IG: `scripts/run_binary_ig_probe.py`
- Path B driver: `scripts/run_pipeline_a_bridge_b{,_sheu}.py`
- eval: `scripts/retally_pipeline_against_audit.py`
