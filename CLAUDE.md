# CLAUDE.md — Implementation Guide
# Systemic Mapping of Cytokine Signaling Cascades via MIL Dynamics

---

## 0. Project Overview

AB-MIL model classifies which stimulus was applied to a pseudo-tube of immune cells.
Training dynamics (per-stimulus learning curves, attention entropy, instance-level
confidence) are analyzed to infer cytokine signaling axes — pairs of stimuli that share
signaling biology — and the cellular relays through which they couple.

**Central hypothesis (axis discovery, validated on Oesinghaus):** stimuli that share a
signaling axis produce overlapping single-cell transcriptional signatures, detectable
via cross-stimulus prediction (alignment), latent-space centroid geometry (geo), and
cell-type ablation. The cellular relay (the cell type that mediates A's effect on B's
signature) can be identified by per-cell-type ablation.

### Project status (current headline; full narrative in `reports/method_deep_dive/`)

Two standing, independent contributions. **All prior work, code, and results remain in the
repo and on `main`** — nothing below retires anything; sections that superseded an earlier
approach say so explicitly at that section.

- **Path A — axis discovery on Oesinghaus 24h PBMC (91 cytokines, §20).** 121 cytokine
  coupling axes recovered (17 textbook directional + 2 pre-registered + 29 coregulated + 13
  partial + 54 novel; ~50% lit-supported vs ~1% chance baseline). Publication-grade, committed
  to `main` (`reports/cascade_pairs/cytokine_axes_report.md`). Direction-agnostic by
  construction (§20) — unaffected by everything below.
- **Path B — `cross_asym` cascade direction (§26), the primary direction metric.** Scores
  **88%** on Oesinghaus, **7/7** on Sheu BMDM (cascadir-native fit; 86% on the legacy fit —
  see §26.3), **83%** on Immune Dictionary (vs ~47% chance
  for the earlier symmetric `directional_score`, §24). This superseded **eight** independently
  failed direction-inference attempts on the old encoder + PBS-RC + dot-product-on-centroid
  bundle, and a since-audited curated-pathway-penetration method (§23's audit callout) —
  do not re-attempt those approaches; post-mortems in `reports/SESSION_SUMMARY_2026-05-25.md`
  and `reports/sheu2024_overnight_summary.md`.

**Two-layer attention v2 (§5.5):** paused, not retired — architecture preserved in code,
candidate for reactivation if a future dataset's structure motivates it.

**Extensions built on `cross_asym`** (§27–§34): Group-U direction FDR, signature-space
coupling, disease-progression (COVID-Haniffa), T-cell maturation (vaccination atlas),
attention- and self-attention training-dynamics. Each section states its own headline
verdict and pointer to its full writeup; `reports/method_deep_dive/` (the "method bible")
and `reports/progress_report/progress_report.pdf` are the canonical up-to-date synthesis
across all of them.

**Datasets in active use (complementary roles, all retained):** Oesinghaus 24h PBMC (§2,
Path A's home), Sheu 2024 mouse BMDM time-course (§2.5), Immune Dictionary in-vivo lymph
node (§2.7), Zhang 2022 human monocytes (§2.6, secondary), COVID-Haniffa and the
SARS-CoV-2 vaccination CITE-seq atlas (§30/§32, progression/differentiation extensions).

---

## 1. Agent Workflow Process

For any change or new feature, follow this order:

1. **`claude-md-updater`** — update CLAUDE.md first.
2. **`disciplined-implementer`** — implement per updated spec.
3. **Periodically:** `sci-decision-auditor` (scientific validity) + `spec-drift-auditor` (implementation matches spec).

---

## 2. Server & Data Locations

**Working directory:** `/cs/labs/mornitzan/yam.arieli/`

**Cluster repo (canonical):** `/cs/labs/mornitzan/yam.arieli/cytokine-mil/`
This is a git clone of `https://github.com/Yam-Arieli/cytokine-mil.git`.
**Dev workflow:** edit locally → `git push` (from local Mac) → `cluster_cmd "cd cytokine-mil && git pull"` → run jobs from the clone.
Scripts are invoked as `python scripts/<name>.py` from the repo root (paths are script-file-relative, not CWD-relative).
Results are written to `cytokine-mil/results/` on the cluster (gitignored).

**Python environment:** `/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python`
Package installed editable: `pip install -e /cs/labs/mornitzan/yam.arieli/cytokine-mil`

**Data paths:**
- Raw dataset: `/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus/Parse_10M_PBMC_cytokines.h5ad`
- Pre-built pseudo-tubes: `/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/`
- Manifest: `…/Oesinghaus_pseudotubes/manifest.json`
- HVG list: `…/Oesinghaus_pseudotubes/hvg_list.json`
- Stage 1 manifest: `…/Oesinghaus_pseudotubes/manifest_stage1.json` *(generated at runtime — one tube per cytokine, ~91 entries)*

**Manifest entry structure:**
```json
{
  "path": "/cs/.../pseudotubes/Donor1/IL-2/pseudotube_0.h5ad",
  "donor": "Donor1",
  "cytokine": "IL-2",
  "n_cells": 480,
  "cell_types_included": ["CD4_T", "NK", "CD14_Mono", ...],
  "tube_idx": 0
}
```

**Pseudo-tube folder structure:**
```
Oesinghaus_pseudotubes/
  manifest.json
  hvg_list.json              <- 4000 HVGs, saved by preprocess_tubes.ipynb
  Donor1/
    IL-2/
      pseudotube_0.h5ad      <- preprocessed (normalized + log1p + HVG filtered)
      pseudotube_0_raw.h5ad  <- original raw counts
      ...
    IFN-gamma/
    PBS/                     <- treated as a regular class during training
  Donor2/
    ...
```

### §2.5 Sheu 2024 dataset (primary for direction inference, phase 1+)

- **Source:** Sheu et al., Molecular Cell 2024; GEO accession: GSE224518.
- **Platform:** BD Rhapsody targeted scRNA-seq (500 immune-response mouse genes) with
  MULTI-seq / sample-tag hashing. 13 GSM accessions are multiplexed sequencing libraries
  demultiplexed by `GSE224518_samptag.all_cellannotations_metadata.txt.gz` into the full
  experimental design.
- **Raw path (cluster):** `/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw/`
- **Pseudo-tube path (cluster):** `/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/`
- **Manifest:** `…/Sheu2024_pseudotubes/manifest.json`
- **HVG list:** `…/Sheu2024_pseudotubes/hvg_list.json` (n_hvgs = 500 — full targeted panel; HVG selection is a no-op for this dataset)

**Active classes (phase 1, 3h time point):** `PBS`, `LPS`, `LPSlo`, `Pam3CSK4`, `polyIC`,
`TNF`, `CpG`, `IFNb` — **8 active classes** (7 stimuli + Unstim relabeled as PBS).
PBS = index 90 (all unstim/0h cells across pseudo-donors pooled and relabeled to literal
`"PBS"` so PBS-RC code in `analysis/pbs_rc.py:59` works unchanged). Unused indices
remain; `n_classes` stays at 91 in code, 8 in practice.

**Pseudo-donor scheme (replaces the biological-donor convention for this dataset):**
Sheu has only 2 biological replicates per condition, so the cytokine-MIL pipeline's
≥3-donor design is satisfied by pooling **(biological context × replicate)** pairs as
pseudo-donors.

**Important data-availability note (verified 2026-05-22 against the actual GEO
deposit, not just the metadata):** the samptag metadata references 16 biological
batches, but only batches **1–13** are deposited as GSM files. Batches 14–16
(which contain the **PM_B6** peritoneal-macrophage samples) are referenced in
metadata but were not deposited. The PM_B6 samples are therefore not downloadable
and the planned pseudo-donor `PM_B6.old_rep1` is unavailable.

After filtering to time_point ∈ {0hr, 3hr} and pooling Unstim/0hr as PBS, the
**4 pseudo-donors actually available at 3hr** are:

| Pseudo-donor      | Cells at 3hr | Split | Available stimuli at 3hr                              |
|-------------------|-------------:|:------|:------------------------------------------------------|
| `M0_rep1`         | 46,451       | train | LPS, LPSlo, P3CSK, polyIC, TNF, CpG  (**no IFNb**)    |
| `M0_rep2`         |  8,897       | train | LPS, P3CSK, polyIC, TNF, CpG, IFNb   (no LPSlo)        |
| `M1_IFNg_rep1`    |  7,177       | train | LPS, P3CSK, polyIC, TNF, CpG, IFNb   (no LPSlo)        |
| `M2_IL4_rep1`     |  6,243       | **val** | LPS, P3CSK, polyIC, TNF, CpG, IFNb (no LPSlo)        |

Plus all 0hr Unstim cells across these 4 pseudo-donors are pooled as the PBS class
(see "Cell types" below for which 0hr cells exist).

**Uneven per-donor class coverage is by Sheu's experimental design** — different
multiplexing rounds covered different stimulus subsets. The pipeline supports
this: axis-discovery tests use whatever pseudo-donors have both endpoints of a
given axis. `polyIC—IFNb` is tested against 3 train donors that have both stimuli;
`LPSlo—P3CSK` is tested against the 1 donor that has LPSlo. Statistical power
varies by axis but the MUST axes (`LPS—TNF`, `polyIC—IFNb`) are well-supported.

The val pseudo-donor `M2_IL4_rep1` tests cross-polarization generalization
(IL-4-primed alternative-activation macrophages, not represented in train).
Both M0 reps remain in train so the §21 M0-only secondary check is well-defined.

**Cell types:** Global Leiden clustering on 0h Unstim cells pooled across all pseudo-donors,
labeled `mac_c0`, `mac_c1`, … (expect 2–4 clusters). Post-stim cells assigned to nearest
0h cluster centroid in PCA space. Same label space across all pseudo-donors so within-tube
stratified sampling means the same thing everywhere.

**Phase 1 time-point subset:** keep only `time_point ∈ {"0hr", "3hr"}`. 3hr is the
earliest time point where both M0 reps are present and where secondary responses
(TNF/IFN feedback) are transcriptionally visible. 0hr cells provide the PBS baseline.

### §2.7 Immune Dictionary — Cui et al., Nature 2024 (3rd dataset, Path B priority)

- **Source:** Cui A. et al., "Dictionary of immune responses to cytokines at single-cell resolution," *Nature* 625, 377–384 (2024). DOI: 10.1038/s41586-023-06816-9.
- **Data deposit:** Broad Institute Single Cell Portal **SCP2554**. Raw counts + log-normalized counts + metadata + tSNE coordinates freely accessible.
- **Platform:** 10x Genomics Chromium 3′ v3, whole transcriptome (~31,053 genes).
- **Raw path (cluster):** `/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary/raw/`
- **Pseudo-tube path (cluster):** `/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/`
- **Manifest:** `…/ImmuneDictionary_pseudotubes/manifest.json`
- **HVG list:** `…/ImmuneDictionary_pseudotubes/hvg_list.json` (n_hvgs = 4000, Seurat v3, matching Oesinghaus)

**Active classes:** 86 cytokines + PBS = **87 active classes**. PBS = index 90 (relabeled at adapter boundary so PBS-RC code works unchanged); `n_classes` stays at 91 in code, 87 in practice. Representative cytokines: IL-1β, IL-6, TNF, IFN-α/β/γ, IL-12, IL-2, IL-15, IL-4, IL-10, IL-17, IL-22, IL-23, IL-33, GM-CSF, TGF-β, BAFF, APRIL, FLT3L (plus ~67 others).

**Pseudo-donor scheme:** `mouse_id` directly. 3 C57BL/6 mice per cytokine (in vivo subcutaneous/intradermal injection). Unlike Sheu, no need to pool across replicates — cell counts per mouse per cytokine are adequate.

**Train/val split:** 2 train + 1 val mouse per cytokine. Val mouse selection deferred to runtime: choose the mouse with the most outlier PBS PCA position (by analogy to Oesinghaus D2/D3 selection). Document the chosen val mouse in the manifest comment after build.

**Cell types (IMPLEMENTED — expert labels, NOT Leiden):** the paper's expert-curated
annotations, pulled per-cell from the **public** SCP2554 REST API (no login — see
`scripts/fetch_scp_id_metadata.py`; committed as `reports/immune_dictionary/scp_metadata.parquet`)
and used directly: `T_cell_CD4`, `T_cell_CD8`, `NK_cell`, `B_cell`, `B_cell_GC`,
`Treg`, `T_cell_gd`, `Macrophage`, `Monocyte`, `cDC1`, `cDC2`, `pDC`, `MigDC`,
`Langerhans`, `ILC`, `Neutrophil`, `Basophil`, `Mast_cell`, `Plasma_cell`, `eTAC`,
`Keratinocyte`, `BEC`, `LEC`, `FRC`, … (26 types; `doublet` dropped). These are the
Stage-1 encoder pre-training targets AND the per-cell-type stratification key for
pseudo-tubes. **No Leiden clustering is run for ID** (unlike Sheu §2.5, which has no
trusted annotations). The earlier GEO-only plan (Leiden `id_c*` on PBS cells) was
dropped once the SCP-API join (§2.7 data-source note in the adapter) made the
expert labels freely available — they are trusted and higher-resolution than ad-hoc
clusters.

**Time point:** single 4 h post sc/id injection (in vivo).

**Replicates:** 3 mice per cytokine. ~386,703 total cells.

**In vivo design note:** the lymph-node microenvironment lets paracrine relays develop, so cascade products are biologically real. However, the relay cell type sometimes differs from the responder cell type. Report relay cell types as informative, not causal.

**Known cascades documented in the paper:** IL-2/IL-12/IL-15/IL-18 → IFN-γ (NK source) → secondary B-cell/DC/macrophage IFN-γ signatures. This is the canonical positive control for §25 on ID.

---

### §2.6 Zhang 2022 dataset (secondary)

- **Source:** Zhang et al., JCI 2022; GEO accession: GSE181475 (probable — verify before first run)
- **Raw path:** `/cs/labs/mornitzan/yam.arieli/datasets/Zhang2022/raw/`
- **Pseudo-tube path:** `/cs/labs/mornitzan/yam.arieli/datasets/Zhang2022_pseudotubes/`
- **Manifest:** `…/Zhang2022_pseudotubes/manifest.json`
- **HVG list:** `…/Zhang2022_pseudotubes/hvg_list.json`
- **Active classes:** `PBS`, `betaglucan`, `uricacid`, `oxLDL`, `MDP`, `LPS` — 6 active classes.
- **Cell types:** Leiden clusters on CD14+ monocytes (expect 2–4 states).
- **Known limitation:** ≤3 donors may break the seed-stability gate. If <3 donors, fall back to plate-id-as-donor and **explicitly flag in the run summary**. Zhang's phase-1 verdict is "consistent / inconsistent with Sheu" only — not a primary gate.

---

## 3. Preprocessing Decisions

**Applied (in order):**
1. **Doublet removal** — Scrublet before building pseudo-tubes. *(Skipped in actual run — `pseudo_tubes_creation.ipynb` built raw tubes without it. Future runs use `scripts/build_pseudotubes.py`.)*
2. **Total count normalization** — normalize each cell to fixed total count.
3. **Log1p transformation** — variance-stabilizing.
4. **HVG selection** — 4000 HVGs (hyperparameter; standard 2000–2500 is tuned for clustering, not neural nets).

Steps 2–4 applied post-hoc via `notebooks/preprocess_tubes.ipynb`. HVGs estimated from one tube per cytokine (~40k cells); all tubes normalized + log1p + filtered in-place. Raw tubes preserved as `pseudotube_N_raw.h5ad`.

**Multi-dataset normalization:** the same pipeline (`normalize_total → log1p → HVG (4000, Seurat v3)`) applies to the Immune Dictionary (§2.7) on its raw 10x counts. No new preprocessing steps beyond what Sheu and Oesinghaus use.

**Never applied:**
- **Z-score per gene:** removes absolute expression the encoder should learn to weight itself.
- **Perturbation scores (log2FC vs PBS):** computing deviation from PBS is a prior injection — assumes resting state is the correct baseline. The model should discover what is informative on its own.

---

## 4. Pseudo-tube Construction Decisions

**Parameters** (`build_pseudotubes.py`):
- `N_PER_CELL_TYPE = 30`, `MIN_CELLS_THRESHOLD = 10`, `N_PSEUDO_TUBES = 10`
- Tube size ≈ 30 × n_eligible_cell_types (≈480 with 16 cell types)

**Design decisions:**
- **Stratified by cell type:** cellular abundance does not drive learnability; any delay reflects transcriptional complexity.
- **Variable tube sizes preserved:** apoptosis/proliferation differences are a meaningful biological signal. **Do NOT equalize tube sizes.**
- **Rare cell types excluded** (< MIN_CELLS_THRESHOLD for a given donor/cytokine) — intentional; tubes are not identical in composition across conditions.
- **Cell type labels dropped** before passing to the network; used only for stratified sampling and post-hoc analysis.
- **PBS = class index 90** during training. Excluded from biological interpretation; tracked as a sanity check.

**Statistical caveat:** pseudo-tubes from the same donor are highly correlated — effective N = 12 (donors), not 120. All statistical comparisons must aggregate to donor level first.

**Multi-dataset adapter convention:** Sheu, Zhang, and ID adapter scripts (`scripts/build_pseudotubes_sheu2024.py`, `scripts/build_pseudotubes_zhang2022.py`, `scripts/build_pseudotubes_immune_dictionary.py`) relabel each dataset's resting/unstim condition to the literal string `"PBS"` at the adapter boundary. This keeps the PBS-index-90 contract (`cytokine_mil/data/label_encoder.py:11`, `label_encoder.py:33`) and PBS-RC computation (`cytokine_mil/analysis/pbs_rc.py:59`, which hard-checks `cytokine == "PBS"`) working unchanged. **No edits to the `cytokine_mil/` package itself in phase 1.**

---

## 5. Model Architecture

Two training stages; each component is a separate class. Stages 2 and 3 should both be run and compared — stable cascade ordering across both is evidence of robustness.

### 5.1. `InstanceEncoder` (`models/instance_encoder.py`)
```
Input:  x_i ∈ R^G  →  Output: h_i ∈ R^128
```
MLP with residual connections. Helpers: `_build_layers()`, `_init_weights()`.
Pre-training objective: supervised cell-type classification. After pre-training, classification head is detached; only backbone carries into Stage 2.
```python
class InstanceEncoder(nn.Module):
    """
    MLP encoder: maps single-cell expression -> dense embedding.
    Pre-trained with cell-type supervision before MIL training.
    """
```

### 5.2. `AttentionModule` (`models/attention.py`)
```
Input:  H ∈ R^(N×128)  →  Output: a ∈ R^N  (sum to 1)
Formula: a_i = softmax( w^T * tanh(V * h_i) )
```
Standard softmax over ALL N cells — no sparsity. Zero dropout (required for stable dynamics tracking).
```python
class AttentionModule(nn.Module):
    """
    Learnable attention aggregation over cell embeddings.
    No dropout — stability of attention weights is required for dynamics tracking.
    """
```

### 5.3. `BagClassifier` (`models/bag_classifier.py`)
```
Input:  z_tube ∈ R^128  →  Output: y_hat ∈ R^K  (K = 91: 90 cytokines + PBS)
```
```python
class BagClassifier(nn.Module):
    """Linear classifier on the aggregated pseudo-tube representation."""
```

### 5.4. `CytokineABMIL` (`models/cytokine_abmil.py`)
```
Input:  X ∈ R^(N×G)
Output: y_hat ∈ R^K, a ∈ R^N, H ∈ R^(N×128)
```
Forward pass returns all three outputs (no second forward pass needed for dynamics).
```python
class CytokineABMIL(nn.Module):
    """
    Full AB-MIL pipeline: InstanceEncoder -> AttentionModule -> BagClassifier
    Accepts a pre-trained InstanceEncoder.
    encoder_frozen: bool controls whether encoder weights are updated.
    """
```

### 5.5. Two-Layer Attention (v2 architecture)

**Status (2026-05-22): PAUSED, not deprecated.** Stage 3 CA-only sanity check on full
Oesinghaus 91-class data (seeds 42, 123; see
`reports/v2_sanity_check/stage3_ca_oesinghaus_results.md`) confirmed the SA/CA
architectural mechanism works (~4-nat entropy gap, exceeds the Oelen prior) but does not
deliver held-out validation AUC gain (median val delta ~0) on Oesinghaus. Diagnosis: the
Oesinghaus 24h-snapshot data is the bottleneck, not the architecture. Remaining 6 seeds
of Stage 3 CA on Oesinghaus are cancelled. Cascade direction is being tested via Sheu
2024 time-resolved data (§2.5) instead.

The architecture, code, and `use_two_layer_attention` config switch are **preserved**.
If a future dataset's structure motivates reactivation — e.g., one where SA and CA can
leverage shared statistical strength across heads — v2 is a candidate to revisit. The
section below documents the architecture for that case.

Two-layer SA+CA attention for cascade specialization. Controlled by
`model.use_two_layer_attention` in `configs/default.yaml`. See `/v2-two-layer-attention`
skill for full architecture spec, KL regularization formula, loss logging, and v2
dynamics extension.

---

## 6. Data Pipeline (`data/dataset.py`)

### `PseudoTubeDataset`
- Reads manifest.json at init; loads one `.h5ad` per `__getitem__`.
- Returns `(X: FloatTensor, label: int, donor: str, cytokine_name: str)`.

### `CellDataset`
**`preload=True` (recommended for Stage 1):**
- Loads all tubes at init into contiguous numpy arrays; `__getitem__` is pure array index.
- Use with Stage 1 manifest (~91 tubes ≈ 40k cells ≈ 640 MB). **Do not use with full 10k-tube manifest (~79 GB).**

**`preload=False` (lazy, default):**
- LRU tube cache (`tube_cache_size=64`). Only efficient with `shuffle=False`, `num_workers=0`; random access defeats cache (~38 h/epoch).

```python
# Stage 1 setup
cell_dataset = CellDataset(STAGE1_MANIFEST_PATH, gene_names=gene_names, preload=True)
cell_loader  = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)
```

### `CytokineLabel` (`data/label_encoder.py`)
Consistent cytokine → integer mapping. PBS → index 90. Saves/loads to JSON.

### Collation
Variable N per tube (rare cell type exclusion). Write custom `collate_fn` (pad/truncate to fixed N) or use batch size = 1 per cytokine in mega-batch.

---

## 7. Training Strategy (`training/trainer.py`)

- **SGD with momentum** — not Adam (adaptive LR causes non-monotonic jumps that obscure dynamics).
- **LR Scheduler:** optional; warmup recommended if early-epoch loss is erratic.
- **Zero dropout in attention layer** (enforced in AttentionModule).

**Gradient accumulation (mega-batch):** one mega-batch = one tube from every cytokine (K=91). Accumulate gradients, step once. Prevents erratic steps from class imbalance.
```python
def train_one_megabatch(model, optimizer, tubes_per_cytokine):
    """
    tubes_per_cytokine: dict mapping cytokine_index -> (X, label)
    Accumulates gradients over all cytokines, steps once.
    """
```

**Stage separation:**
```python
train_encoder(encoder, cell_type_dataloader, n_epochs=...)
mil_model = CytokineABMIL(encoder, encoder_frozen=True)
train_mil(mil_model, pseudotube_dataloader, n_epochs=...)
mil_model.unfreeze_encoder()  # Stage 3 optional
train_mil(mil_model, pseudotube_dataloader, n_epochs=...)
```

---

## 8. Dynamics Tracking (`analysis/dynamics.py`)

All metrics logged per epoch, per cytokine.

### 8.1. Macro View — Systemic Complexity
```
P(t)(Y_correct) = bag-level correct class probability at epoch t
```
Aggregate to donor level (median across tubes per donor, then across 12 donors).

### 8.2. Distribution View — Attention Entropy
```
H(t) = -sum_i a_i(t) * log(a_i(t))
```
Per tube, per logged epoch. Shape: `(n_logged_epochs,)`. Summary: mean across all epochs, aggregated to donor level.
- Low entropy → targeted pathway; High entropy → pleiotropic response *(correlation, not proof)*

### 8.3. Micro View — Instance-Level Confidence
```
C_i(t) = a_i(t) * P(t)(Y_correct)
```
Per cell, per logged epoch. Shape: `(n_cells, n_logged_epochs)`. **Do not collapse** — full trajectory stored; aggregation in analysis layer.
Post-training: reintroduce cell_type annotations, group by cell type, compute AUC of mean C_i(t).
- Early spike → Primary Anchor; Steady climb → Secondary Relay; High inter-donor variance → Conditional Responder

**v2 architecture (two-layer attention):** `compute_instance_confidence` accepts an optional second attention vector. When using `CytokineABMIL_V2`, two trajectories are stored per cell per logged epoch:
```
C_SA_i(t) = a_SA_i(t) * P(t)(Y_correct)   <- primary responder signal
C_CA_i(t) = a_CA_i(t) * P(t)(Y_correct)   <- cascade responder signal
```
Both are stored with the same shape `(n_cells, n_logged_epochs)` and analyzed separately. Divergence between SA and CA cell-type profiles is the key readout — cells prominent in SA but not CA are interpreted as direct targets; cells prominent in CA but not SA are interpreted as secondary cascade responders.

### 8.4. Confusion View — Confusion Entropy (scalar per cytokine)
```
H_confusion(C, t)  — shape: (n_logged_epochs,) per cytokine
```
Computed across all tubes of cytokine C simultaneously:
1. Per tube b: take full softmax ŷ_b ∈ R^K.
2. Average: ȳ_C(t) = mean_b ŷ_b(t).
3. Remove true class, renormalize: q_k(t) = ȳ_{C,k}(t) / sum_{j≠C} ȳ_{C,j}(t).
4. H_confusion(C,t) = -sum_{k≠C} q_k(t) log(q_k(t)).

Low entropy → confusion on similar cytokines; High entropy → genuine difficulty.
Computed in `_compute_confusion_entropy_snapshot` inside `_log_dynamics`. Returned as `dynamics["confusion_entropy_trajectory"]`: `{cytokine_name: np.array(n_logged_epochs)}`.

**Distinction from pairwise confusion trajectory (Section 19):** H_confusion(C,t) is a *scalar* per cytokine per epoch — it summarizes how concentrated or diffuse the off-diagonal softmax mass is. It does not reveal *which* other cytokines absorb that mass, nor the direction of confusion. Section 19 introduces a (K, K, T) pairwise tensor C(A, B, t) that tracks softmax mass assigned to every other class for every true class at every epoch. The scalar H_confusion is computed inside `analysis/dynamics.py`; the pairwise tensor is computed in `analysis/confusion_dynamics.py`. Both are kept and serve different purposes.

### Helper functions
```python
def compute_entropy(attention_weights: torch.Tensor) -> float
def compute_instance_confidence(attention: torch.Tensor, p_correct: float) -> torch.Tensor
def aggregate_to_donor_level(records, trajectory_key="p_correct_trajectory") -> dict
def group_confidence_by_cell_type(confidences, cell_type_labels) -> dict
def compute_confusion_entropy_summary(confusion_entropy_trajectory, exclude=None) -> dict
def build_cell_type_confidence_matrix(records, cell_type_obs) -> dict
```

### Precise output labels
Every ranking/summary function returns a `metric_description` key. Examples:
```
Learnability ranking
Metric: AUC of mean p_correct_trajectory across pseudo-tubes, aggregated to donor
        level (median across pseudo-tubes per donor, then mean across donors)

Attention entropy summary
Metric: mean across epochs and pseudo-tubes of H(attention_weights) = -sum a_i log(a_i),
        aggregated to donor level

Cell-type cascade profile for IL-2
Metric: AUC of mean C_i(t) = a_i(t) * P(t)(Y_correct), averaged across cells of
        the same type within each pseudo-tube, then across pseudo-tubes per donor

Confusion entropy
Metric: AUC of H_confusion(C,t) = -sum_{k≠C} q_k(t) log q_k(t)
```

---

## 9. Validation Plan (`analysis/validation.py`)

- **9.1 Seed stability:** learnability ordering must be consistent across multiple seeds; if not, signal is too noisy.
- **9.2 Known cascade recovery:** state directional predictions before unblinding. Pre-registered: type I interferons learned earliest. Known control: IFN-γ → NK (primary), monocytes (secondary).
- **9.3 Known functional groupings:** correlated cytokines (e.g., IL-2 / IL-15, r=0.92 in CD14 Mono) should show similar learnability and entropy profiles.
- **9.4 Multiple testing correction:** Benjamini-Hochberg FDR for all pairwise comparisons across 91 classes.
- **9.5 24-hour snapshot confound** (analysis only, no experiment change): build cytokine → expected response maturity table (~50–60% feasible). After unblinding: if "hard" cytokines correlate with slow kinetics → report as confound; if not → evidence of cascade complexity.
- **9.6 Confusion Dynamics Validation:** See Section 19.5 for Experiments 0–5 (synthetic positive control go/no-go gate, IL-12→IFN-γ biological recovery, IL-6/IL-10 shared-pathway negative control, seed stability of asymmetry scores, cytokine family clustering sanity check, 24h kinetics confound check post-unblinding).

---

## 10. Experiment Variants & Setup Module (`cytokine_mil/experiment_setup.py`)

Shared setup logic extracted from `experiment.ipynb` so variants don't copy-paste:
- **Full experiment** — all 91 cytokines, multi-class
- **Subset experiment** — selected cytokines, multi-class
- **Binary experiment** — one cytokine vs PBS, one model per cytokine

### Functions

```python
def build_stage1_manifest(manifest, save_path=None) -> list
```
One tube per cytokine, rotating donors (donor i mod n_donors). Optionally saves to JSON.

```python
def filter_manifest(manifest, cytokines, include_pbs=True) -> list
```
Filter to cytokine subset. Always includes PBS unless `include_pbs=False`.

```python
def make_binary_manifest(manifest, target_cytokine, control="PBS") -> (list, BinaryLabel)
```
2-class manifest + `BinaryLabel` encoder (positive→0, negative→1).

```python
def split_manifest_by_donor(manifest, val_donors) -> (train_manifest, val_manifest)
```
Donor-level train/val split. See Section 16.

```python
def build_encoder(n_input_genes, n_cell_types, embed_dim=128) -> InstanceEncoder
def build_mil_model(encoder, embed_dim=128, attention_hidden_dim=64,
                    n_classes=91, encoder_frozen=True) -> CytokineABMIL
```

### `BinaryLabel` (`data/label_encoder.py`)
Two-class encoder: `positive→0`, `negative→1`, `n_classes()→2`. Same interface as `CytokineLabel` (`.encode`, `.decode`, `.n_classes`, `.cytokines`).

### Typical usage
```python
from cytokine_mil.experiment_setup import (
    build_stage1_manifest, filter_manifest, make_binary_manifest,
    build_encoder, build_mil_model,
)

# Subset experiment
subset_manifest = filter_manifest(manifest, cytokines=["IL-2", "IL-15", "IFN-gamma"])

# Binary experiment (IL-2 vs PBS)
bin_manifest, bin_label = make_binary_manifest(manifest, "IL-2")
# bin_label.n_classes() == 2 → pass to build_mil_model(n_classes=2)
```

---

## 11. Project File Structure & Packaging

```bash
pip install git+https://github.com/Yam-Arieli/cytokine-mil.git
pip install -e ".[dev]"  # editable dev install
```
Cluster venv has the package installed editable at `/cs/labs/mornitzan/yam.arieli/cytokine-mil`.
Edit locally → `git push` → `cluster_cmd "cd cytokine-mil && git pull"` (no reinstall needed for editable installs).

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "cytokine-mil"
version = "0.1.0"
dependencies = ["torch>=2.0", "scanpy>=1.9", "anndata>=0.9",
                "numpy>=1.24", "pandas>=2.0", "scikit-learn>=1.3",
                "pyyaml>=6.0", "tqdm>=4.65"]

[project.optional-dependencies]
dev = ["pytest>=7.0", "jupyter>=1.0", "ipykernel>=6.0"]
```

Top-level layout (for the current exact file listing, use the project's `/index` skill
rather than this hand-maintained tree — it goes stale as sections are added):

```
cytokine_mil/               <- repo root
├── cytokine_mil/           <- importable package: data/, models/, training/, analysis/
├── cascadir/                <- reusable dataset-agnostic package (Section 29)
├── scripts/                <- one driver script per experiment/section
├── configs/                 <- per-dataset YAML (default, sheu2024, zhang2022, immune_dictionary)
├── slurm/                    <- sbatch wrappers + DAGs, one subdir per experiment
├── reports/                  <- pre-registrations + results writeups, one subdir per experiment
├── notebooks/                <- experiment.ipynb and variants (Sections 10, 17-18)
├── tests/                    <- demo-data fixtures + unit tests (Section 12)
├── hypotheses/                <- falsifiable-prediction docs, written before an experiment runs
└── thesis/                   <- METHOD_GROUND_TRUTH.md, WONDERINGS.md (see /thesis-sync skill)
```

---

## 12. Demo Data & Local Testing

Real data is on the cluster; use simulated demo data locally.

**Demo spec (`tests/make_demo_data.py`):**
- 10 cytokines + PBS = 11 classes
- 3 donors, 1 tube per (donor, cytokine) — Donor3 held out for val split testing
- 5 cell types, 20 cells each → 100 cells/tube; 200 simulated genes (log-normalized)
- Writes `.h5ad` files + `manifest.json` mirroring cluster structure

```bash
pip install -e ".[dev]"
pytest tests/test_demo.py tests/test_demo_sheu.py tests/test_demo_id.py -v
```

**Sheu demo spec (`tests/make_demo_data_sheu.py`):**
- 6 stimuli + PBS = 7 classes, mirroring Sheu's active classes at 3h
- 3 donors, 1 tube per (donor, stimulus) — PBS cells pooled from all donors per Sheu adapter convention
- 3 cell types (`BMDM_c0`, `BMDM_c1`, `BMDM_c2`), 20 cells each → 60 cells/tube; 200 simulated genes (log-normalized)
- Writes `.h5ad` files + `manifest.json` mirroring Sheu pseudo-tube structure
- `cytokine` column uses Sheu stimulus names; `"PBS"` string for control

**Tests cover:**
- Label encoder roundtrip and PBS index
- Dataset loading and item shapes
- All model forward pass shapes (InstanceEncoder, AttentionModule, BagClassifier, CytokineABMIL)
- Encoder freeze/unfreeze; attention weights sum to 1
- Stage 1 encoder pre-training runs without error
- Stage 2 MIL training runs and returns dynamics dict
- Learnability ranking and instance confidence grouping by cell type
- Donor-level manifest split correctness
- `train_mil` with `val_dataset` returns `val_records` with correct structure
- `CytokineABMIL_V2` forward pass returns correct shapes for y_hat, a_SA, a_CA, H
- a_SA and a_CA each sum to 1 independently
- Both C_SA_i(t) and C_CA_i(t) confidence trajectories logged correctly when using v2
- Confusion trajectory tensor shape is (K, K, T) and diagonal is excluded from asymmetry scores
- `compute_asymmetry_score` output is antisymmetric: Asym[A,B] = -Asym[B,A] (see Section 19)
- Sheu demo manifest has `"PBS"` string in `cytokine` field for all control entries
- `PseudoTubeDataset` loads Sheu demo manifest and returns correct shapes (60 cells, 200 genes)
- `split_manifest_by_donor` on Sheu demo produces donor-disjoint train/val sets
- ID demo manifest has `"PBS"` string in `cytokine` field for all control entries
- `PseudoTubeDataset` loads ID demo manifest and returns correct shapes (60 cells, 200 genes)
- `split_manifest_by_donor` on ID demo produces mouse-disjoint train/val sets

**ID demo spec (`tests/make_demo_data_id.py`):**
- 5 cytokines + PBS = 6 classes, mirroring ID's structure
- 3 mice, 1 tube per (mouse, cytokine) — 1 mouse held out for val split testing
- 2 cell types (`id_c0`, `id_c1`), 30 cells each → 60 cells/tube; 200 simulated genes (log-normalized)
- Writes `.h5ad` files + `manifest.json` mirroring ID pseudo-tube structure
- `cytokine` column uses ID cytokine names; `"PBS"` string for PBS-injected control mice

---

## 13. Hyperparameters (`configs/default.yaml`)

```yaml
data:
  manifest_path: /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
  n_hvgs: 4000
  n_per_cell_type: 30
  min_cells_threshold: 10
  n_pseudo_tubes: 10
  val_donors: ["Donor2", "Donor3"]

model:
  embedding_dim: 128
  attention_hidden_dim: 64
  n_classes: 91
  use_two_layer_attention: false

training:
  stage1_epochs: 50
  stage2_epochs: 100
  stage3_epochs: 50
  optimizer: sgd
  momentum: 0.9
  lr: 0.01
  lr_scheduler: null
  lr_warmup_epochs: 5
  kl_lambda: 0.1          # v2 only: weight for KL(a_CA || a_SA) divergence penalty
  aux_loss_weight: 0.5    # v2 only: weight for SA and CA auxiliary classification losses

dynamics:
  log_every_n_epochs: 1
  random_seeds: [42, 123, 7]
  confusion_late_epoch_fraction: 0.3   # fraction of final epochs for asymmetry score
  confusion_fdr_alpha: 0.05            # FDR threshold for cascade graph edges
  cascade_graph_min_seed_rho: 0.7      # min Spearman rho across seeds for reportable pairs

aux_decoder:
  embed_dim: 64
  tau_values: [0.3, 0.5, 1.0]         # temperature sweep for bag-level softmax sharpening
  tau_default: 0.5
  lr: 1e-3
  epochs: 50
```

---

## 14. Code Style Preferences

- **Break long functions into helpers.** Every function does one thing.
- **Each model component is a separate class** in its own file; combined only in `cytokine_abmil.py`.
- **PyTorch throughout.** No framework mixing.
- **Strict separation:** data loading, training, analysis are fully decoupled. Training does not import from analysis; models do not import from data.
- **Private helpers** use leading underscore: `_build_layers()`, `_init_weights()`.
- **Precise output labels (mandatory):** every analysis output (rankings, plots, axis labels, report strings) states exactly what is computed. Every ranking/summary function returns a `metric_description` string printed alongside results.
  - Use `AUC(mean_donor_p_correct_trajectory)` not `AUC`
  - Use `mean H(attention_weights) = -sum a_i log(a_i)` not `entropy`
  - Use `AUC(mean_donor_C_i(t))` not `confidence score`
  - Use `AUC(H_confusion)` not `confusion entropy`

---

## 15. Key Reminders

- **Never pass cell_type labels to the MIL model.** Only for pseudo-tube construction and post-hoc analysis.
- **PBS is class 90 during training.** Exclude from biological interpretation; track as sanity check.
- **Aggregate to donor level before any statistical comparison.** Effective N = 12, not 120.
- **Inspect loss curves.** Enable LR warmup if early-epoch loss is erratic.
- **Run multiple seeds** before drawing conclusions about learnability ordering.
- **State directional predictions before unblinding** dynamics results.
- **Hold out Donor2 and Donor3.** Never use val donors during training or optimizer steps. (Oesinghaus-specific; see below for Sheu/Zhang.)
- **Sheu val pseudo-donor:** `M2_IL4_rep1` (single val; only 4 pseudo-donors
  are downloadable at 3hr — see §2.5 for the GEO-deposit gap). Train set: 3
  pseudo-donors — `M0_rep1`, `M0_rep2`, `M1_IFNg_rep1`. Pseudo-donor =
  `(type × replicate)` because Sheu has only 2 biological reps per condition.
  Per-donor class coverage is uneven by design (M0_rep1 has no IFNb;
  M0_rep2 / M1 / M2 have no LPSlo).
- **Zhang val donors:** TBD pending donor-count verification. If <3 donors, fall back to plate-id-as-donor and skip the seed-stability gate.
- **ID val mouse:** 1 of 3 (chosen at build time based on PBS PCA outlier position across all cytokines). Train set: 2 mice. Document chosen val mouse in manifest comment field after build. See §2.7.
- **ID time point:** single 4 h (in vivo). No kinetic validation as in Sheu. The §24/§25 directional-asymmetry method is single-time-point by construction; evaluate the §25 cascade sweep with this caveat explicitly stated in the verdict.

---

## 16. Donor-Level Validation Split

**Rationale:** pseudo-tubes from the same donor are highly correlated (effective N = 12). Holding out individual tubes is not independent validation. **The only valid generalization test is a donor-level hold-out.** Without it, we cannot distinguish genuine cytokine-specific programs from memorized donor-specific patterns.

**Hold-out donors: D2 and D3**
- **D3** — Strongest outlier in the interferon group: highest baseline ISG expression, weakest correlation to consensus IFN response. Tests generalization of canonical IFN programs.
- **D2** — Aberrant CD14 Mono baseline resembling an IL-32-β-stimulated state. Tests whether monocyte-stimulating cytokines are confused with D2's resting state.

Training donors: D1, D4–D12 (10 donors).

**What is tracked:** at every logged epoch, P(Y_correct) evaluated on held-out val tubes with no gradient updates. Val split is observer-only.

Returned in dynamics dict:
- `val_records`: list of per-tube dicts (same structure as `records`)
- `val_confusion_entropy_trajectory`: `{cytokine_name: np.array(n_logged_epochs)}`

**Interpretation:**
- Train ≈ Val: generalizable programs; dynamics valid.
- Train >> Val (widening gap): partial donor memorization; learnability ranking is confounded.
- Val converges slower than train: expected (10 train vs 2 val donors); focus on trajectory shape.

If Train >> Val gap exceeds empirical threshold, report as limitation and compare train vs. val rankings separately.

### Implementation

**`split_manifest_by_donor` (`cytokine_mil/experiment_setup.py`):**
```python
def split_manifest_by_donor(
    manifest: List[dict],
    val_donors: List[str],
) -> Tuple[List[dict], List[dict]]:
    """
    Split a manifest into train and val sets at the donor level.

    Args:
        manifest: Full manifest list.
        val_donors: Donor names to hold out (e.g., ["Donor2", "Donor3"]).
    Returns:
        (train_manifest, val_manifest) where val_manifest contains all
        entries whose donor is in val_donors, and train_manifest contains
        the rest. Both retain the full set of cytokines.
    """
```

**`train_mil` signature (`cytokine_mil/training/train_mil.py`):**
```python
def train_mil(
    model: CytokineABMIL,
    dataset: PseudoTubeDataset,
    n_epochs: int,
    ...,
    val_dataset: Optional[PseudoTubeDataset] = None,
) -> Dict:
```

When `val_dataset` is provided:
- `val_entries = val_dataset.get_entries()`
- `val_tube_trajectories = _init_tube_trajectories(val_entries)`
- `val_cytokine_confusion_epochs: Dict[str, List[float]] = defaultdict(list)`
- At each logging step: `_log_dynamics(model, val_dataset, val_entries, val_tube_trajectories, val_cytokine_confusion_epochs, val_dataset.label_encoder, device)`
- Returns `"val_records"` (empty list if no val_dataset) and `"val_confusion_entropy_trajectory"` (empty dict if no val_dataset).

**Typical usage:**
```python
from cytokine_mil.experiment_setup import split_manifest_by_donor

train_manifest, val_manifest = split_manifest_by_donor(manifest, val_donors=["Donor2", "Donor3"])
train_dataset = PseudoTubeDataset(train_manifest_path, label_encoder)
val_dataset   = PseudoTubeDataset(val_manifest_path, label_encoder)
dynamics = train_mil(model, train_dataset, n_epochs=100, val_dataset=val_dataset)
train_donor_traj = aggregate_to_donor_level(dynamics["records"])
val_donor_traj   = aggregate_to_donor_level(dynamics["val_records"])
```

**Demo data:** `tests/make_demo_data.py` uses 3 donors (`DONORS = ["Donor1", "Donor2", "Donor3"]`). Tests use `split_manifest_by_donor(manifest, val_donors=["Donor3"])`.

---

## 17. Binary Experiment Notebook (`notebooks/experiment_binary.ipynb`)

One binary AB-MIL per cytokine (cytokine vs PBS, n_classes=2) with a shared frozen
encoder. Uses same 10-cytokine subset as `experiment_subset.ipynb`. See `/binary-experiment`
skill for training protocol, metrics (Normalized Trajectory AUC, Final Probability),
group thresholds (EASY/HARD/MED), and precise output labels.

---

## 18. Bootstrap Experiment (`notebooks/experiment_bootstrap.ipynb`)

Tests SIMPLE vs COMPLEX cytokine learnability via a bootstrapped 5+5 subset
(controlled by `BOOTSTRAP_SEED = 42`). Pre-registered: one-sided Mann-Whitney U,
never repeated. See `/bootstrap-experiment` skill for pool definitions, sampling
logic, hypothesis test, and validation checks. Cytokine pool definitions also in
`/cytokine-pools` skill.

---

## 19. Cascade Inference via Confusion Dynamics (`analysis/confusion_dynamics.py`)

Asymmetric confusion between cytokine classes over training time reveals cascade
direction. Builds a (K, K, T) confusion tensor; computes asymmetry scores and
temporal profiles; outputs a directed cascade graph. Config params under `dynamics:`
in `default.yaml`. See `/confusion-dynamics` skill for hypothesis, tensor math,
function signatures, validation experiments (Exp 0–5), and precise output labels.

---

## 20. Latent Space Cytokine Geometry (`analysis/latent_geometry.py`)

Detects cytokine-pair coupling as per-cell-type directional bias of cell embeddings
toward other cytokine centroids. **Direction-agnostic on single-layer attention** —
the readout is symmetric by construction (§20.1) and the empirical 2026-05-20 lit
review showed directional inference is at chance (49% / 51%). Outputs should be
treated as **cytokine axis** calls (unordered pairs with a relay cell type), not
directed cascades, until two-layer attention v2 (§5.5) is trained and the SA/CA
asymmetry is wired in.

Run on 20-cytokine subset first. GO/NO-GO gate: Exp 0 (cytokine alignment score vs
null). Contingency path: AuxDecoder (Exp 3). See `/latent-geometry` skill for full
experiment specs (Exp 0–3), math, function signatures, attention proxy check
results (2/5 FAIL → uniform KL), and precise output labels.

### 20.1 Refined readout (current default)

The legacy `bias(A,B,T) = (µ_{A,T} − µ_A) · û_{A→B}` followed by
`ASYM(A→B) = max_T [bias(A,B,T) − bias(B,A,T)]` had two problems:

1. The subtraction injects a `µ_{B,T} · û_{A→B}` contamination term — a strong
   direct B-responder cell type inflates the asymmetry score even when no
   cascade exists.
2. The score is antisymmetric by construction: it cannot distinguish a genuine
   directional cascade from an algebraic sign flip.

The refined pipeline (in `cytokine_mil.analysis.pbs_rc` + `latent_geometry.py`):

1. **PBS-RC space first.** Compute `µ_{PBS, T}` per cell type from training donors
   only via `pbs_rc.compute_pbs_centroids_per_cell_type`. Subtract per cell type:
   `h̃_i = h_i − µ_{PBS, τ(i)}`. In PBS-RC space `µ_{A,T}` is T's deviation from
   its own resting state (step 1 of the "oranges vs oranges" comparison).
2. **Per-donor projection with centroid subtraction.** For each cytokine pair (A, B),
   each cell type T, each training donor d:
   `b_fwd^{(d)}(A→B, T) = (µ_{A,T}^{(d)} − µ_A) · û_{A→B}`
   where `µ_A` is the pooled training-donor PBS-RC centroid of cytokine A (average
   deviation from PBS across all cell types). The `µ_A` subtraction removes A's
   generic cross-cell-type signal so that the score reflects T's *specific* cascade
   component beyond A's direct effect on all cells.
3. **Two independent one-sided Wilcoxon signed-rank tests** across donors —
   no `b_fwd − b_rev` subtraction anywhere. The "reverse" of (A, B) is the forward
   test for (B, A): `b_fwd^{(d)}(B→A, T) = (µ_{B,T}^{(d)} − µ_B) · û_{B→A}`.
4. **Bonferroni** across cell types per ordered pair (relay search), then
   **BH-FDR** across the K(K−1) ordered pairs.
5. **Cascade decision per pair (A → B):**
   `fwd_sig = ∃T : p_fwd_bonf(A→B, T) ≤ α`,
   `rev_sig = ∃T : p_fwd_bonf(B→A, T) ≤ α`.
   Calls: `'A->B'`, `'B->A'`, `'shared'`, `'none'`.
   Relay: `T* = argmin_T p_fwd_bonf(A→B, T)`.

Direction modes (config `latent_geometry.direction_mode`): `'global'` uses
`û_{A→B} = (µ_B − µ_A)/||µ_B − µ_A||`; `'cell_type'` uses `µ̂_{B,T}` as the
direction (T-specific; safe because forward/reverse are no longer subtracted).

Public API in `cytokine_mil.analysis.latent_geometry`:
- `compute_directional_bias_per_donor(cache, label_encoder, pbs_ct_means, train_donors, direction_mode)`
- `test_directional_significance(bias_per_donor, label_encoder, alpha)`
- `build_latent_cascade_graph_from_calls(significance, label_encoder)`

Deprecated (kept for backwards compatibility behind `--legacy-asymmetry`):
- `compute_asymmetry_matrix`
- `build_latent_cascade_graph`

### 20.2 Axis reframing (current reporting default)

Because directional inference is at chance under single-layer attention, the
default reporter for downstream consumption is `scripts/report_cytokine_axes.py`,
which collapses `cascade_call`'s directional output into direction-agnostic axis
calls:

- **axis_a, axis_b**: unordered canonical pair (`a ≤ b` lexicographically).
- **axis_strength**: `max(pooled_relay_a→b, pooled_relay_b→a)`.
- **relay_T_candidates**: top-3 most-frequent argmax cell types across seeds.
- **literature_status**: PRE_REGISTERED / KNOWN_DIRECTIONAL / KNOWN_COREGULATED /
  PARTIAL / NOVEL / NAME_AMBIGUOUS — joined from
  `reports/cascade_pairs/literature_review_aggregate.json`.
- **literature_direction**: tag indicating whether literature says A→B (`a_to_b`),
  B→A (`b_to_a`), both (`bidir`), antagonist/family coregulation
  (`coregulated_other`), or no literature support (`no_lit`).

Headline reporting language: "discovered cytokine coupling axes" — *not*
"discovered cascades". Cascade language returns when v2 is trained.

---

## 21. Phase 1 Axis-Discovery Gate (Sheu 2024)

**Purpose:** method validation. Does the axis-discovery pipeline, already validated on
Oesinghaus (121 axes, ~50% lit-supported, §0), recover textbook TLR cascade pairs from
a Sheu Stage 2 model trained on the 3h time point? This is a **sanity check that the
pipeline transfers to the new dataset**, not the directional-inference experiment itself
(that is phase 2 if this gate is GREEN). The Oesinghaus axis-discovery result is
**unaffected** by this gate — it stands independently.

After training Sheu Stage 2 (3 seeds: 42, 123, 7), run `scripts/run_latent_geometry.py`
then `scripts/report_cytokine_axes.py`.

**Pre-registered expected axes** (chosen before analysis, based on shared TLR adaptor /
autocrine cascade biology — see commit message 2026-05-22 for the receptor-by-receptor
rationale):

**MUST recover** (failure ⇒ pipeline broken on this dataset OR signal absent in 3h BMDM):
1. `LPS — TNF` (TLR4 → NF-κB → autocrine TNF loop)
2. `polyIC — IFNb` (TLR3/TRIF → IRF3 → type-I IFN — cleanest cascade in the panel)

**SHOULD recover** (textbook but secondary; partial failure is informative, not fatal):
3. `LPS — IFNb` (LPS engages TRIF arm in addition to MyD88)
4. `Pam3CSK4 — CpG` (both MyD88-only, no TRIF arm)
5. `LPSlo — Pam3CSK4` (both MyD88-biased; tests whether low-dose LPS phenotype is
   correctly distinguished from full LPS)

**MUST NOT call** (false positives ⇒ pipeline over-calls cascades that have no biology):
- `Pam3CSK4 — IFNb` (TLR2 has no TRIF arm; no IRF3 / type-I IFN induction)
- `CpG — IFNb` (TLR9 → IFN-α is plasmacytoid-DC-restricted; macrophages produce
  minimal type-I IFN through this route)
- `TNF — IFNb` (no cross-induction in macrophages)

**Quantitative pass criterion (go/no-go for phase 2 time-axis work):**

For each pre-registered axis (positive and negative):
- BH-FDR ≤ 0.05 on the pseudo-donor-level Wilcoxon (`latent_geometry.test_directional_significance`)
  computed across the 5 train pseudo-donors `M0_rep1, M0_rep2, M1_IFNg_rep1,
  PM_B6.HFD_rep1, PM_B6.LFD_rep1`
- Axis-ranking Spearman ρ ≥ 0.7 across all 3 seeds (matches `cascade_graph_min_seed_rho: 0.7`)

**Primary analysis: all 5 train pseudo-donors pooled.** Cross-context variation (M0 vs
PM strain backgrounds) may dominate latent geometry. To control for this, a secondary
analysis is pre-registered.

**Secondary analysis: M0-only sub-check.** Re-run `latent_geometry` restricted to the
two M0 pseudo-donors (`M0_rep1`, `M0_rep2`) for the same pre-registered axis list.
The 2-donor Wilcoxon is severely underpowered; this is not a separate gate but a
direction-of-effect check — at minimum, the sign of the per-pseudo-donor bias
projection for the MUST axes should agree across `M0_rep1` and `M0_rep2`.

**Composite verdict:**
- **GREEN**: 2 of 2 MUST pass + ≥2 of 3 SHOULD pass + 0 of 3 MUST-NOT called (primary)
  AND M0-only sub-check agrees in sign for both MUST axes (secondary) → start phase 2
  (time-axis extension via composite-label encoding `cytokine@time_point` + new
  `analysis/temporal_confusion.py`; the 0.25h/0.5h vs 3h/5h/8h asymmetry is the actual
  direction test).
- **AMBER**: 1 of 2 MUST pass OR 1 of 3 MUST-NOT called OR M0-only sub-check
  disagrees → re-run with `direction_mode: cell_type` and `n_per_cell_type: 50`; if
  still amber, write up partial result and defer the direction question (axis discovery
  on Oesinghaus is the standing result).
- **RED**: 0 of 2 MUST OR ≥2 MUST-NOT called → cascade signal not recoverable from 3h
  BMDM with this architecture; try 1h or 5h time-point subsets before reconsidering
  phase 2.

Verdict written to `reports/sheu2024/AXIS_GATE_VERDICT.md`. **In any outcome, the
Oesinghaus axis-discovery result is independent of this gate** — Path A continues.

---

## 22. Pair-level EDA Benchmark (`analysis/eda_pair_benchmark.py`, `analysis/eda_pair_plots.py`)

**Motivation:** all eight prior cascade-direction checks (§0 status block) shared
the same method bundle: encoder embedding + PBS-RC + dot-product readout on
per-donor centroids. The decision after 2026-05-25 was to *invert* the workflow:
stop designing methods from cascade-signal assumptions, build a labeled-pair
benchmark from Sheu §21, compute a wide statistic battery directly on
normalized expression, and let the data show where (if anywhere) the cascade
signature lives.

**Labeled pairs** (constants in `eda_pair_benchmark.py`):

| Status | Pair | Reason |
|---|---|---|
| positive (MUST) | `LPS — TNF` | TLR4 → NF-κB → autocrine TNF |
| positive (MUST) | `PIC — IFNb` | TLR3/TRIF → IRF3 → type-I IFN |
| positive (SHOULD) | `LPS — IFNb` | LPS engages TRIF arm |
| positive (SHOULD) | `P3CSK — CpG` | both MyD88-only |
| positive (SHOULD) | `LPSlo — P3CSK` | both MyD88-biased |
| negative (MUST-NOT) | `P3CSK — IFNb` | TLR2 has no TRIF arm |
| negative (MUST-NOT) | `CpG — IFNb` | TLR9 IFN restricted to pDC |
| negative (MUST-NOT) | `TNF — IFNb` | no cross-induction in macrophages |

**Statistic battery** (computed per ordered (A, B) pair per cell type T, vs PBS):

*Symmetric / similarity:* `centroid_distance`, `log2fc_spearman`, `de_jaccard`,
`var_ratio_AB`.

*Asymmetric (cascade-relevant by construction):*
`frac_A_closer_to_B`, `frac_B_closer_to_A`, `reciprocal_asymmetry`,
`mean_sigB_in_A`, `mean_sigA_in_B`, `sigB_in_A_norm`, `sigA_in_B_norm`,
`signature_asymmetry`, `frac_A_with_high_sigB`, `frac_B_with_high_sigA`,
`tail_asymmetry`, `kl_A_to_B_along_AB`, `kl_B_to_A_along_AB`, `kl_asymmetry`.

*Heterogeneity / mixture (within-tube shape, not means):*
`var_A_along_AB`, `var_B_along_AB`, `bimodality_A_along_AB`,
`bimodality_B_along_AB`.

**Discrimination test:** for each statistic, AUC of ranking the 5 labeled
positives above the 3 labeled negatives, on the per-unordered-pair `max`
aggregator (across ordered directions and cell types). Permutation null:
shuffle the positive/negative labels n_permutations times; report the 0.95
quantile per statistic as the discrimination floor.

**Plots** (under `<out_dir>/plots/`):
- `statistic_heatmap.pdf` — labeled pairs × statistics, z-scored per column
- `auc_bars.pdf` — AUC per statistic with permutation null overlay
- `signature_scatter/<A>__<B>.pdf` — per-cell (s_A, s_B) scatter faceted by cell type
- `projection_density/<A>__<B>.pdf` — overlaid histograms on û_{A→B}, faceted by cell type

**Driver:** `scripts/run_sheu_eda_benchmark.py` + `slurm/run_sheu_eda.slurm`.
Default output dir on cluster: `results/sheu_eda/`. Entire pipeline runs in
minutes on a single CPU node; no model training, no checkpoints needed.

**Interpretation rule:** statistics with AUC > permutation null upper quantile
are candidates. Inspect (1) what they actually measure (variance? signature
tail? KL?) and (2) which labeled pairs they get right vs wrong. Then sharpen
the next round of methods from the data, not from priors.

**Outcome of first run on Sheu 3hr (2026-05-25):** No statistic clears the
permutation null (best empirical p ≈ 0.064). Visual scatters reveal the
confounder: in the 500-gene targeted Sheu panel, top-DE-up signatures of
every stimulus correlate strongly with every other stimulus' top-DE-up,
because the panel is curated to immune-response genes. Empirical signatures
aren't pathway-specific in this panel. The strongest *direction* of effects
in the heatmap is "cascade pairs are more similar" (lower centroid distance)
— a similarity signal, not a direction signal. Conclusion driving §23: try
curated, adaptor-specific gene sets.

---

## 23. Pathway-Signature Cascade Analysis (`analysis/pathway_signatures.py`, `analysis/pathway_plots.py`, `analysis/pathway_audit.py`)

> **⚠ Audit revision (2026-05-26):** the single-pathway penetration test
> described in this section was audited via `scripts/run_pathway_audit.py`
> and found to be **statistically unreliable**: random gene sets of the
> same size give similar AUC, the per-cell Mann-Whitney p-values were
> inflated by pseudo-replication, and the binary-AUC test does not clear
> a cytokine-label permutation null. **The defensible methodology is the
> two-paired-pathways directional asymmetry score** (Audit 4 in
> `pathway_audit.py`), which compares each cascade's upstream and
> downstream pathway signatures jointly. See
> `reports/sheu2024_pathway/cascade_direction_results.md` for the revised
> result and §24 below for the asymmetry methodology specification.

**Motivation:** §22 showed that empirical top-DE signatures collapse onto a
correlated diagonal in the 500-gene Sheu panel — the panel is curated to
immune-response genes that all move together on activation, so signature
overlap is by construction. §23 replaces these with **literature-curated,
signaling-adaptor-specific gene sets**, then asks how much of each pathway's
signature appears in tubes that are NOT directly stimulated through that
pathway. That's a direction-relevant readout if the curated genes are
adaptor-specific.

**Curated pathway library** (mouse symbols; constants in `pathway_signatures.py`):

| Pathway | Marker genes | Primary stimuli | Cascade-induced from |
|---|---|---|---|
| `IRF3_direct` | `Ifnb1, Ccl5, Cxcl10, Ifit2, Ifit3` | PIC, LPS (TRIF arm) | — |
| `IFNAR_induced` | `Isg15, Mx1, Mx2, Oas1a, Oas2, Oas3, Ifit1, Rsad2, Stat1, Irf7, Usp18` | IFNb | PIC, LPS (autocrine IFN-β) |
| `NFkB_canonical` | `Tnf, Il1b, Il6, Nfkbia, Nfkbid, Tnfaip3, Cxcl1, Cxcl2, Ccl3, Ccl4, Birc3` | LPS, LPSlo, P3CSK, CpG, TNF | — |
| `TNFR_autocrine` | `Tnfaip3, Nfkbid, Birc3` | TNF | LPS, LPSlo, P3CSK, CpG (autocrine TNF) |

**Cascade penetration** (in `compute_penetration`):
```
penetration(A → P, B) = (mean(s_P, A-tube) − mean(s_P, PBS))
                      / (mean(s_P, B-tube) − mean(s_P, PBS))
```
Where B is the primary stimulus for pathway P. `s_P(cell) = mean(cell, pathway_genes)`
(no per-cell control subtraction — random control genes from a 500-gene
immune panel carry their own pathway signal and bias the subtraction; the
PBS baseline at tube level already removes resting-state activity).
Penetration ≈ 1 means A fully recapitulates B's pathway; ≈ 0 means A doesn't
engage P; intermediate = partial cascade. Asymmetric by construction.

**Pre-registered binary test (`ifnar_binary_test`):**
- Pathway: `IFNAR_induced`, primary: `IFNb`
- Positives (predicted high penetration): `PIC, LPS, LPSlo, IFNb`
- Negatives (predicted ~0 penetration): `P3CSK, CpG, TNF`
- AUC=1.0 has empirical p < 1/35 ≈ 0.029 (passes 0.05).
- Computed per cell type; "clean separation" = all positives > all negatives.

**Magnitude cascade test (`magnitude_cascade_test`):**
For cascade pairs that share a pathway (e.g., LPS→TNF on `NFkB_canonical`),
predict `s(A) > s(B) > s(PBS)` because A engages B's pathway directly *plus*
gets autocrine boost from cascade B.

**Files:**
- `cytokine_mil/analysis/pathway_signatures.py`
- `cytokine_mil/analysis/pathway_plots.py`
- `scripts/run_sheu_pathway_signatures.py`
- `slurm/run_sheu_pathway.slurm`

**Outputs (default `results/sheu_pathway/`):**
- `resolved_pathways.json` — which curated genes are present in the panel
- `penetration_long.parquet` — (pathway × primary × A × cell_type) → penetration
- `ifnar_binary_summary.csv` + `ifnar_binary_summary.pdf` — the pre-registered test
- `magnitude_lps_tnf.csv` — does LPS > TNF > PBS on NF-κB hold?
- `plots/penetration_heatmap.pdf` — full penetration matrix faceted by cell type
- `plots/pathway_strip_<pathway>.pdf` — per-pathway violins across stimuli (visual sanity)

**Runtime safety:** at startup the script reports which curated genes are
present in the panel per pathway. Pathways with < 3 curated genes resolved
are skipped. If no pathway resolves (e.g., wrong gene-symbol case),
the script aborts cleanly.

**Mouse pathway library extension (for §25 ID sweep):** in addition to the
TLR-centric IRF3_direct / IFNAR_induced / NFkB_canonical / TNFR_autocrine
sets, the following JAK-STAT family pathways are curated for the ID phase 2
sweep (full gene lists TBD by literature review, committed to `pathway_signatures.py`
before any audit script runs):

- `IL12_STAT4_target` — IL-12 specific STAT4 targets (NOT shared with STAT1)
- `IFNg_STAT1_target` — IFN-γ specific STAT1 targets (NOT shared with type-I IFN ISGs)
- `IL6_STAT3_target` — IL-6 specific STAT3 targets
- `IL2_STAT5_target` — common γ-chain STAT5 targets
- `IL4_STAT6_target` — IL-4 specific STAT6 targets

Critical: each pair (P_A, P_B) used in a cascade test must be transcriptionally
distinct from one another for the §24 directional asymmetry test to work. The
overlap matrix of curated genes must be computed and verified before commit.
Do NOT specify actual gene lists in CLAUDE.md — that belongs in `pathway_signatures.py`.

---

## 24. Directional Asymmetry Cascade Test (`analysis/pathway_audit.py`)

Post-audit primary methodology for cascade-direction inference. Distinct from
§23's single-pathway penetration test — that one was retired after the
2026-05-26 audit (it reads "activation level" rather than "pathway
specificity"; see §23 audit note).

> **⚠ SUPERSEDED AS PRIMARY (2026-06) — see §26.** When `P_A`/`P_B` are the cytokines' own
> *discovered* signatures, `directional_score` (defined below) is algebraically **symmetric**
> under swapping A↔B, so its *sign cannot encode direction* (it scored 47% ≈ chance on
> Oesinghaus). The current **primary** direction metric is the **antisymmetric `cross_asym`**
> (§26). `directional_score` is retained below as a secondary *coupling-distinctness* reference.

### 24.1 Construction

For a candidate cascade A → B with two transcriptionally distinct paired
pathways (`P_A` = the pathway A engages directly; `P_B` = the pathway B
engages directly), compute per cell type T:

1. Per-cell pathway scores: `s_X_on_PY = mean expression of P_Y genes`
   for X ∈ {A, B, PBS} on Y ∈ {A, B}. Four core means: `s_A_on_PA`,
   `s_A_on_PB`, `s_B_on_PA`, `s_B_on_PB`. Plus PBS baselines on each.
2. PBS-normalised: subtract `s_PBS_on_PY` from each tube's score on P_Y.
3. Two asymmetries:
   - `asym_PA = s_A_on_PA_norm − s_B_on_PA_norm` — does A engage P_A more
     than B does? Cascade A→B predicts **positive** (A's own pathway,
     B has no upstream signal).
   - `asym_PB = s_A_on_PB_norm − s_B_on_PB_norm` — does A engage P_B more
     than B does? Cascade A→B predicts **negative** or **≈0** (both engage
     P_B, but B is direct → maximum; A is partial via autocrine).
4. `directional_score = asym_PA − asym_PB`. **Positive ⇒ A→B cascade
   consistent with biology. Negative ⇒ reverse direction would be implied.
   ~0 ⇒ no cascade asymmetry detected.**

### 24.2 Critical preconditions

The test only works when:
- **P_A and P_B are transcriptionally distinct.** If the two curated gene
  sets overlap (e.g., NFkB_canonical and TNFR_autocrine — both are NF-κB
  targets), the asymmetry test fails not because the cascade is absent but
  because the two pathways are not discriminable. The audit confirmed this:
  IFN cascades (IRF3-direct ↔ IFNAR-induced) discriminate cleanly;
  NF-κB → TNFR cascades cluster on the diagonal in audit-4 plots.
- **PBS baseline is stable across cell types.** If the cell type has
  unusual constitutive expression on one of the pathways (mac_c3 had
  elevated PBS s_NFkB > s_TNF), the asymmetry is corrupted. Sanity check
  PBS baselines per cell type before interpreting.
- **Stimulus identities map correctly to P_A and P_B.** Misassignment of
  primary pathway will flip the directional_score sign.

### 24.3 Pre-registered positives from Sheu 2026-05-26 audit

| Cascade | P_A | P_B | mac_c2 directional_score | mac_c3 directional_score |
|---|---|---|---:|---:|
| PIC → IFNb | IRF3_direct | IFNAR_induced | +1.87 | +1.67 |
| LPS → IFNb | IRF3_direct | IFNAR_induced | +2.35 | +2.02 |

NF-κB cascades (LPS/LPSlo/P3CSK/CpG → TNF, with P_A=NFkB_canonical and
P_B=TNFR_autocrine) gave directional_score values in [-0.10, +0.84],
mostly small magnitudes — **not** clean cascade evidence. The two paired
pathways overlap.

### 24.4 Public API

```python
from cytokine_mil.analysis.pathway_audit import directional_asymmetry_test

df = directional_asymmetry_test(
    cells_by_pair,              # {(cytokine, cell_type) -> (N, G) array}
    pathway_idx_dict,            # {"IRF3_direct": ndarray, "IFNAR_induced": ...}
    A="PIC", B="IFNb",
    P_A="IRF3_direct", P_B="IFNAR_induced",
    pbs_label="PBS",
    min_cells=10,
)
# Returns DataFrame with columns: cell_type, sA_on_PA_norm, sB_on_PA_norm,
# sA_on_PB_norm, sB_on_PB_norm, asym_PA, asym_PB, directional_score, interpretation
```

### 24.5 Honest power discussion

For Sheu 3hr the test gives 4 observations (2 cascade pairs × 2 informative
cell types). Each is a single number, not a sample of cells, so per-cell
inflation is not a concern — but inference across observations is
underpowered. The honest claim is "consistent positive directional_score
pattern across the four pre-registered observations, with magnitudes
+1.7 to +2.4". A donor-level extension (computing directional_score per
pseudo-donor then signed-rank across the 3-4 donors) is the next rigour
upgrade.

### 24.6 What this methodology does NOT do

- Does not prove causation. Asymmetric signature is consistent with cascade
  direction but interventional data (e.g., IFNAR knockout) is needed to
  close the causal loop.
- Does not generalise to overlapping pathway pairs (NF-κB family
  cascades). Failure mode is silent — directional_score just drifts to ~0.
- Does not work on stimuli without a defined `P_A` (cytokine cascades
  where the upstream stimulus engages JAK-STAT directly, e.g., IL-2 → ?
  in Oesinghaus, have no obvious upstream P_A to use).

---

## 25. Phase 2 §24 Cascade Sweep on Immune Dictionary (`scripts/run_immune_dictionary_pathway_audit.py`)

**Purpose:** extend the §24 directional-asymmetry test from Sheu's 2 cascades
(PIC→IFNb, LPS→IFNb) to a larger pre-registered set spanning JAK-STAT, NF-κB, and
SMAD pathway families. The 31K-gene 10x transcriptome resolves distinct curated
pathways that the 500-gene Sheu panel could not separate. The §24 methodology and
`directional_asymmetry_test` API are unchanged; only the cascade list and pathway
library are extended.

> **⚠ UPDATED (2026-06): run the ID sweep with the `cross_asym` pipeline (§26), not the
> curated-pathway `directional_score` sweep described below.** The pre-registered cascade
> list in §25.2 is retained as the **evaluation benchmark**, re-expressed as directional
> `cross_asym` labels (alphabetical pair + `expected_sign`; bidirectional pairs such as
> IL-12↔IFN-γ are excluded from signed accuracy). The §24-style curated sweep is kept as an
> optional secondary comparison. Full method: §26 + `reports/method_deep_dive/`.

### 25.1 Pre-registration discipline

The cascade list, P_A / P_B pairing, and predicted directional_score sign per
cascade **must be committed** to `reports/immune_dictionary/PRE_REGISTRATION.md` on
`main` **before** any audit script runs. The JAK-STAT pathway gene sets in
`pathway_signatures.py` (§23 extension) must also be locked at that same commit.
Running `run_immune_dictionary_pathway_audit.py` before this commit is a protocol
violation.

### 25.2 Pre-registered cascade list

**MUST-PASS — distinct pathways, predicted directional_score > 0:**

| # | Cascade | P_A | P_B | Biological rationale |
|---|---|---|---|---|
| 1 | IL-12 → IFN-γ | `IL12_STAT4_target` | `IFNg_STAT1_target` | IL-12 drives STAT4 in NK/T; IFN-γ engages STAT1-only ISGs |
| 2 | IL-1β → IL-6 | `NFkB_canonical` | `IL6_STAT3_target` | IL-1β signals NF-κB; IL-6 signals STAT3 — distinct downstream targets |
| 3 | IFN-γ → IL-12 | `IFNg_STAT1_target` | `IL12_STAT4_target` | IFN-γ → STAT1 → Il12b upregulation (positive feedback) |
| 4 | TNF → IL-6 | `NFkB_canonical` | `IL6_STAT3_target` | TNF → NF-κB → IL-6 induction in myeloid cells |
| 5 | IFN-β → IFN-γ | `IFNAR_induced` | `IFNg_STAT1_target` | Type-I IFN primes NK for IFN-γ production (in NK cell type) |

**MUST-FAIL — overlapping pathways, predicted |directional_score| < 0.5:**

| # | Cascade | P_A | P_B | Predicted failure mode |
|---|---|---|---|---|
| 6 | IL-2 → IL-15 | `IL2_STAT5_target` | `IL2_STAT5_target` | Both common γ-chain → STAT5; gene sets overlap by design |
| 7 | IL-1β → TNF | `NFkB_canonical` | `NFkB_canonical` | Both NF-κB; replicates the §24 known failure from Sheu |
| 8 | IL-4 → IL-13 | `IL4_STAT6_target` | `IL4_STAT6_target` | Both STAT6; gene overlap by shared receptor IL-4Rα |

**NEGATIVE CONTROLS — no cascade biology, predicted directional_score ≤ 0:**

| # | Cascade | P_A | P_B | Reason |
|---|---|---|---|---|
| 9 | IL-4 → IFN-γ | `IL4_STAT6_target` | `IFNg_STAT1_target` | Th2 actively inhibits Th1; antagonistic not cascade |
| 10 | IL-10 → IL-12 | `NFkB_canonical` | `IL12_STAT4_target` | IL-10 suppresses IL-12 production; reverse of any induction |

### 25.3 Scoring and verdict

Each cascade scored per cell type via `directional_asymmetry_test`, then aggregated
per mouse as mean(directional_score) across informative cell types. Donor-level
Wilcoxon signed-rank across 2 train mice + 1 val mouse (n=3) is underpowered —
report magnitude and direction, not p-values, as the primary readout.

**GREEN verdict:** 4 of 5 MUST-PASS show directional_score > +1.0 in ≥1 cell type
AND 2 of 3 MUST-FAIL show |directional_score| < 0.5 AND both NEGATIVE CONTROLS show
directional_score ≤ 0 → §24 generalises beyond TLR-IFN cascades; Path B claim
upgrades to "cascade-direction inference across JAK-STAT, NF-κB, and IFN pathway
families."

**AMBER:** 2–3 of 5 MUST-PASS pass, OR 1 of 3 MUST-FAIL is miscalled → identify
which pathway pairs caused failure; check gene-set overlap matrix; refine before
reporting.

**RED:** 0–1 of 5 MUST-PASS → §24 does not generalise; report as TLR-IFN-specific
result only.

Verdict written to `reports/immune_dictionary/CASCADE_SWEEP_RESULTS.md`.

### 25.4 Driver script and slurm wrapper

- **Script:** `scripts/run_immune_dictionary_pathway_audit.py` — mirrors
  `scripts/run_pathway_audit.py`; parameterised for ID manifest and gene symbols.
- **Slurm:** `slurm/run_id_pathway_audit.slurm` — single CPU node; no model
  training, no checkpoints; expected runtime < 30 min.
- **Output dir (cluster):** `results/immune_dictionary_pathway/`

---

## 26. Cross-Engagement Asymmetry (`cross_asym`) — PRIMARY cascade-direction metric (2026-06)

**Status:** supersedes §24's `directional_score` as the primary direction method. Full
pedagogical reference: **`reports/method_deep_dive/`** (the "method bible", modules M0–M9).

### 26.1 The fix (why `directional_score` failed)
With per-cytokine **discovered** signatures `S_a, S_b` (so `P_A = S_a`, `P_B = S_b`), the §24
`directional_score = asym_PA − asym_PB` is **algebraically symmetric** under swapping a↔b:
`directional_score(b,a) == directional_score(a,b)`. Its sign therefore **cannot encode
direction** — it measures *coupling distinctness*, not who is upstream (47% ≈ chance on
Oesinghaus). The antisymmetric **cross-engagement** statistic is the fix:

```
# per cell type T (only where a, b AND PBS all have >= min_cells):
cross_asym_T(a,b) = s_T(a, S_b) − s_T(b, S_a)   # = sA_PB_norm − sB_PA_norm  (PBS-normalised)
# the statistic = the MEDIAN of those, over the cell types the pair shares:
cross_asym(a,b)   = median over T of cross_asym_T(a,b)
cross_asym(b,a)   = − cross_asym(a,b)           # ANTISYMMETRIC ⇒ the sign encodes direction
```
Convention (pairs stored alphabetically, `a < b`): `+` ⇒ a upstream (`a_to_b`); `−` ⇒ b
upstream (`b_to_a`). Biology: an upstream stimulus's cells carry **both** programs (their own +
the autocrine downstream one), the downstream ligand's carry mainly their own ⇒
`s(upstream, S_down) > s(down, S_up)`.

### 26.2 The pipeline (discovered signatures — no curated gene sets)
1. **Path A** — multiclass AB-MIL → latent geometry (§20) → *which* pairs are coupled
   (unordered axes). Direction-blind; the standing coupling result.
2. **Bridge** — per-stimulus **binary** AB-MIL (stimulus vs PBS) + **Integrated Gradients**
   (PBS-mean baseline, 20-step midpoint) → discovered signature `S_X` = top-50 genes by IG
   (`scripts/run_binary_ig_probe.py` → `binary_ig.parquet`). The binary models share one
   frozen Stage-1 encoder so the `S_X` are comparable.
3. **Path B** — `cross_asym` on `S_X` per cell type, then **median across cell types +
   sign-consensus**; random-gene-set null. Drivers: `scripts/run_pipeline_a_bridge_b.py`
   (Oesinghaus) / `…_sheu.py` (Sheu). Eval: `scripts/retally_pipeline_against_audit.py
   --metric cross_asym` (default).

### 26.3 Results
- **Oesinghaus 24h PBMC: cross_asym 15/17 = 88%** (vs `directional_score` 8/17 = 47%, same
  data, same signatures); 34/53 axes beat the random-gene-set null (p<0.05); benchmark
  label-permutation p = 0.003.
- **Sheu BMDM single-frame, no cross-time leakage: 5h = 7/7** on the **cascadir-native** fit
  (`results/sheu_cascadir_native/5hr/`) — the thesis's source since 2026-07-28. Six are STRONG
  (permutation p<0.01); the seventh, P3CSK→TNF, is a near-zero non-call (+0.006, p=0.95) whose
  sign agrees. Symmetric control 2/6 = 33% (chance). NF-κB→TNF 4/4.
  - The **legacy** fit (`results/sheu_cascade/5hr/pathB/`, the old headline) scores **6/7 = 86%**
    (1h 4/5, 3h 5/7), missing polyIC→IFNb. Both fits read the same 5h data; they are separate
    training runs, agreeing on sign for 17/21 pairs. The 4 disagreements are IFNb–PIC,
    P3CSK–PIC, PIC–TNF, CpG–LPS — three involve PIC (S_polyIC is ISG-dominated ≈ S_IFNb, §26.4).
  - **Why native is the thesis source:** Sheu's *coupling* only ever existed for the native fit
    (`coupling_cell.csv`), so using it for direction too removes a mixed-provenance split and
    makes Sheu reproducible end-to-end through the packaged `cascadir` API. **Not** because it
    scores higher — one pair out of seven is inside run-to-run noise, and selecting the fit on
    its benchmark score would bias the reported number. Quote the fit whenever quoting a Sheu
    5h direction value; do not average the two.

### 26.4 Honest caveats
- **cross_asym gives direction, NOT existence.** Negative pairs also have large `|cross_asym|`;
  deciding *whether* a pair is coupled is Path A's job. Magnitude is not a coupling gate.
- Small n (17 Oes, 7 Sheu directional axes). The 88% is vs a conservative hand-audit.
- Requires `S_a ≠ S_b`; when a signature collapses onto the shared program (polyIC ISGs) the
  sign can flip — a known, mechanistically-understood failure.

### 26.5 Immune Dictionary run
Run the §26 pipeline on the ID (§2.7) using the §25.2 pre-registered cascades as the
benchmark, re-expressed as `cross_asym` directional labels (alphabetical pair + `expected_sign`;
bidirectional pairs such as IL-12↔IFN-γ excluded from signed accuracy). Verdict →
`reports/immune_dictionary/CASCADE_SWEEP_RESULTS.md`.

---

## 27. Full Path A → Path B + Group-U direction FDR (Oesinghaus, 2026-06)

**Motivation (the gap this closes).** The `cross_asym` benchmark (§26) was evaluated on a
hand-curated / audit-derived shortlist of labeled directional pairs (17 on Oesinghaus, 53
pairs total with a direction). That measures **recall on knowns** but leaves two holes:
(1) the two-stage "Path A couples → Path B directs" pipeline was never actually run
end-to-end — the benchmark pairs were carved out of Path A's own output (circular); and
(2) the **Group-U** pairs (Path-A-coupled pairs with *no* directional prior) get a
confident `cross_asym` sign but were never scored — so the method's behaviour on the
*unknown* coupled pairs (where novel-cascade discovery would live) is uncharacterised.
This section runs the full pipeline over **all 121 Path A coupled axes** and quantifies
the unknown calls with a **direction-specific permutation-null FDR**.

**Scope:** Oesinghaus only (Path A genuinely works there — 121 axes; Sheu's Path A gate
FAILED, ID's emitted nothing, so the two-stage claim is Oesinghaus-only and the
direction-only Sheu/ID results in §26 are unchanged). Direction-not-existence and
not-causation caveats (§26.4) carry over verbatim.

### 27.1 Full Path A → Bridge → Path B (de-circularise)
`run_pipeline_a_bridge_b.py` already reads `cytokine_axes.csv` (121 axes) as input but
only 53 pairs resolved (21 cytokines had binary IG signatures). Train binary AB-MIL +
IG for the **missing cytokines** (= cytokines in `cytokine_axes.csv` − cytokines already
in `binary_ig.parquet`; ~24 of the 45) using the wide config (embed=512, hidden=(512,512),
attn=128, Stage1 20@0.005, Stage2 250@3e-5 — matches §17/§26 bridge so the IG probe loads
all models uniformly), merge into a 45-cytokine `binary_ig.parquet`, then run the pipeline
over all 121 axes. The 17 labeled pairs reproduce the §26 headline (regression check); the
remaining **~104 = Group U**.

### 27.2 Direction-permutation null (`cytokine_mil/analysis/direction_null.py`)
The §26 random-gene-set null tests "are the `S_X` genes cytokine-specific" — NOT "is the
direction real" (overlap pairs pass it because shared-program genes *are* specific). The
correct null for direction holds `S_a, S_b` **fixed** and breaks only the a-vs-b label:

1. Per axis (a, b), per cell type T: precompute each pooled (a∪b)-cell's score on `S_a`
   and on `S_b` (mean expression over the gene set) — done **once**.
2. Observed `cross_asym(T) = (mean_{a-cells} score_Sb − pbs_Sb) − (mean_{b-cells} score_Sa
   − pbs_Sa)`; matches `directional_asymmetry_test`'s `sA_PB_norm − sB_PA_norm`. PBS
   baselines are fixed (not permuted).
3. Permute the a/b membership **within each cell type** (preserve counts n_a, n_b),
   recompute; aggregate **median across cell types** per permutation. `n_perm=1000`,
   `seed=123`.
4. The null has a **nonzero baseline centre** by construction (the `S_a`/`S_b` magnitude
   offset is a nuisance, not direction). Recentre: `p_emp_two_sided = mean_k(|null_k −
   null_centre| ≥ |observed − null_centre|)`, `null_centre = mean(null_k)`. The direction
   **call** (sign) stays the observed `cross_median` sign; the null only asks whether the
   a-vs-b asymmetry is beyond label noise.

This null tests "**is the directional asymmetry statistically reliable**", not "is this a
cascade" (existence = Path A) and not "is it causal" (wet-lab). Overlap pairs with a real
magnitude asymmetry *may* pass — that is expected and correct; coupling is Path A's gate.

New driver flags (default-off, backwards compatible): `--n_direction_perms` (0 = skip),
`--direction_null_seed`. New per-axis columns: `dir_n_perms, dir_null_center,
dir_null_q025, dir_null_q975, dir_p_emp`.

### 27.3 Group-U FDR (`scripts/run_group_u_fdr.py`)
Reads `per_axis_summary.csv` + the audited labels; partitions **labeled** (counts_in_benchmark)
vs **Group U** (coupled, no prior); over Group U computes **BH-FDR** on `dir_p_emp`
(manual, no scipy) and a **Storey π₀** estimate (λ=0.5) → "K of the ~104 unknown coupled
pairs carry a reliable directional asymmetry at FDR q". Emits the ranked Group-U
hypothesis list and `reports/cascade_pairs/GROUP_U_RESULTS.md`.

### 27.4 Pre-registration (locked BEFORE the run, per §25.1)
`reports/cascade_pairs/GROUP_U_PREREGISTRATION.md` (committed to `main` before any audit
script runs) locks: n_perm=1000, FDR q∈{0.05,0.10}, confident-hypothesis definition
(`dir BH-q ≤ 0.10` AND `cross_consensus ≥ 0.7` AND `|cross_median| ≥ 25th-pctile of
labeled-positive |cross_median|`), top-K=10, and the calibration predictions:
- **P1 (power):** ≥ 80% of labeled non-AMBIGUOUS positives pass the direction null (q ≤ 0.10).
- **P2 (specificity):** near-zero / known-miss pairs do NOT pass.
- **P3 (headline, discovery-capable):** Group-U π₀ < 0.9 (i.e., > 10% of unknown coupled
  pairs carry a reliable direction). π₀ ≈ 1 ⇒ method is confirmation-only.
- **P4 (regression):** the §26 labeled accuracy is unchanged by adding the ~24 cytokines.

### 27.5 Driver / SLURM
- New: `scripts/train_oesinghaus_binary_groupu.py` (computes the missing set from
  axes_csv − existing binary_ig; clones the missing16 trainer), `scripts/run_group_u_fdr.py`,
  `cytokine_mil/analysis/direction_null.py`.
- SLURM DAG `slurm/group_u/`: `train.slurm` (GPU) → `ig.slurm` (CPU) → `merge.slurm` (CPU)
  → `pipeline.slurm` (CPU, `--n_direction_perms 1000`) → `fdr.slurm` (CPU); submitter
  `submit_group_u_dag.sh` (dry-run via `SUBMIT=echo`). Output dir `results/group_u/`.
- **Bottom line:** `reports/cascade_pairs/GROUP_U_RESULTS.md` + `results/group_u/pipeline_full121/per_axis_summary.csv`.

### 27.6 Results (2026-06-03) — NOT a valid discovery claim

Ran end-to-end (jobs 30726479–30726489). P1 (power) passed, but two pre-registered checks
falsified the headline: the cell-level permutation null is over-powered (thousands of cells
per cell-type make ~everything "significant" — the null must be donor-level, §16), and the
all-45-cytokine re-run failed to reproduce §26's labeled accuracy (6/11 vs 15/17), likely
because `train_oesinghaus_binary_groupu.py` used a separate encoder per chunk instead of a
shared one. **Verdict: OPEN**, pending a donor-level null and signature reproduction — §26's
88% and Path A's 121 axes are unaffected. Full writeup:
`reports/cascade_pairs/GROUP_U_RESULTS.md`.

---

## 28. Signature-space coupling — the "specific-dimensions" reframe of Path A (2026-06)

**Motivation.** Latent-geometry Path A (§20) measures coupling in the **encoder
embedding** with PBS-RC. PBS-RC removes the *resting* baseline but NOT the
**shared post-activation program** — the immune-response genes almost every
cytokine co-induces. So apparent coupling can be dominated by that shared
direction (cytokines look similar because they all *activate*), not by
cytokine-**specific** biology. Evidence: on Oesinghaus, latent-geometry
`axis_strength` correlates only weakly (Spearman ρ ≈ 0.29) with signature-space
coupling; the textbook IL-2/IL-15 pair tops signature space but not Path A; and
IL-6/TNF-α (pre-registered) has **negative** signature-space coupling (IL-6→STAT3
and TNF→NF-κB are specifically *distinct*, coupled only via shared activation).
On Sheu the latent-geometry gate had **no power at all** (q=1 everywhere) — the
500-gene immune panel is *all* shared-activation genes.

**The reframe.** Run gene-set detection (binary-IG `S_X`) **first**, then measure
coupling DIRECTLY in those specific dimensions, bypassing the encoder embedding.
Build the **cross-engagement matrix** from the per-cell-type engagement
`s_T(a,S_b) = mean(S_b in a's T-cells) − mean(S_b in PBS's T-cells)`
(= `directional_asymmetry_test`'s `sA_PB_norm`, generalised to every ordered pair):

```
M[a,b] = median over cell types T of s_T(a, S_b)
```

Each entry medians over the cell types where **a and PBS** qualify (b is *not* required),
so `M[a,b]` and `M[b,a]` may run over different cell-type sets.

- **Coupling(a,b)** (SYMMETRIC) — do a and b *mutually* engage each other's specific
  programs? Raw `C[a,b] = M[a,b] + M[b,a]`; the **reported and gated** score is its
  degree-corrected (double-centred) residual `R[a,b] = C[a,b] − d_a − d_b + ḡ` (§28.2 —
  the hub fix, on by default). Gate: donor-level sign test where donors allow, else the
  **gene-set null** (is it > engagement of random gene sets of the same size, drawn
  disjoint from any `S_X`? — the "strong enough signal" gate; over-powered at cell scale).
- **Direction** is **not** read off `M`. It is `cross_asym` (§26): the median over the
  cell types the pair *shares* of `s_T(a,S_b) − s_T(b,S_a)`. Read only on coupled pairs
  (existence ≠ direction). ⚠ This is a **median of differences** and is *not* equal to
  `M[a,b] − M[b,a]` (a difference of medians) — the median is not linear, and the two `M`
  entries may use different cell types. The `cross_asym` column emitted by
  `signature_coupling` / `coupling_trajectory` **is** that `M[a,b] − M[b,a]`
  approximation, kept for compatibility (on Sheu 5 h it differs in **sign** on 6/21
  pairs); take direction from `cascadir.cross_asym.direction_table` /
  `directional_asymmetry_test`.

**Code:** `cytokine_mil/analysis/signature_coupling.py`
(`engagement_per_celltype`, `cross_engagement_matrix`, `coupling_direction`;
numpy-only, unit-tested to match `directional_asymmetry_test`'s per-cell-type terms).
Driver `scripts/run_signature_coupling.py --dataset {oesinghaus,sheu}`.

**Two pre-registered tests (run 2026-06, alongside the §27 DAG):**
- **Oesinghaus** (`slurm/group_u/coupling_oes.slurm`, all-45 signatures from the
  §27 `ig_merge`): compare the coupling axis set + literature support to
  latent-geometry Path A's 121 axes; report Spearman(coupling, axis_strength).
- **Sheu** (`slurm/group_u/coupling_sheu.slurm`, 3hr + 5hr, single-frame):
  **decisive "irrelevant features" test** — does signature-space coupling recover
  the textbook MUST pairs (LPS–TNF, polyIC–IFNb) that the latent-geometry gate
  FAILED (0/2)? If yes, Sheu's Path A failure was measuring shared activation, not
  geometry. Output `results/group_u/coupling_{oes,sheu_*}/coupling_report.md`.

**Honest caveat.** It all rides on `S_X` being *specific*. On Sheu's 500-gene
panel `S_X` may still be activation-dominated (the §22 collapse) — in which case
signature coupling also struggles there, and the bottleneck is the panel, not the
geometry. That is exactly what the Sheu test decides. Does NOT replace Path A's
published 121-axis result until this is validated; it is a candidate reframe.

### 28.1 Results (2026-06-03)

**Sheu — WIN (decisive).** Recovers 2/2 MUST IFN cascades (LPS–IFNb, polyIC–IFNb) at both
3hr and 5hr that latent-geometry Path A failed (0/2, q=1); clean negatives stay uncoupled.
Confirms the diagnosis — Sheu's Path A failure was measuring shared activation, not a
missing signal. **Oesinghaus — signal present, gate too loose.** Right biology at the top
(IL-15/IL-2 #1), but the gene-set null is over-permissive (894/1128 pairs "coupled") and
hub-dominated (IL-15 in 11/20 top pairs) — needs donor-level + degree correction (fixed in
§28.2). **Immune Dictionary — not run.** The two coupling paths are complementary (latent:
rich, shared-activation-confounded, works on Oes; signature: specific, over-permissive
gate, rescues Sheu). Full writeup: `reports/cascade_pairs/SIGNATURE_COUPLING_RESULTS.md`.

### 28.2 Gate fix — donor-level + degree(hub) correction (VALIDATED 2026-06-19)

The §28.1 over-call was diagnosed and fixed via `scripts/run_signature_ablation.py` (2×2
{IG,DE}×{vsPBS,vsPanel}), `run_donor_coupling_null.py` (donor sign-flip null + degree
correction), `run_cell_degree_coupling.py` (cell-level degree). Code in
`cytokine_mil/analysis/signature_coupling.py` (`donor_excess_matrix`,
`donor_residual_coupling_matrix`, `donor_coupling_test`, `_degree_center`,
`cell_coupling_degree`).

- **Default signature is IG_vsPBS** (raw top-50 by IG, no residualisation), matching
  `cascadir.derive_signature` and `reports/progress_report/progress_report.pdf` (Table 2).
  DE ≠ IG (Jaccard 0.11, DE direction ≈chance) — IG is not replaceable by DE. IG_vsPanel
  (panel-residualised) ties IG_vsPBS on direction and doesn't reduce the coupling over-call,
  so it was **not adopted** as default. *(Correction 2026-07-07: an earlier draft of this
  bullet said to adopt IG_vsPanel — wrong; the progress report and cascadir's raw-IG path
  both contradict it. A marginal vsPanel edge exists on Oesinghaus only and over-strips on
  ID/Cano-Gamez: `reports/cascade_pairs/COUPLING_DONOR_COUNT_OES.md`.)*
- **Degree (hub) correction is THE fix** (double-center the coupling matrix; symmetric, so
  `cross_asym`/direction unaffected). Oesinghaus donor+degree: over-call 77%→**31%**, recall
  8→**11/17**. Sheu cell+degree: keeps 2/2 IFN cascades, suppresses all 3 negatives.
- **Boundary:** donor-level null needs ~8+ well-covered donors (Oesinghaus); Sheu (4)/ID (3)
  use cell-level degree correction instead.
- **Ported to `cascadir`** (`signature_coupling(..., degree_correct=True)` default). MANUAL
  §4/§5/§8 updated; tests pass.

---

## 29. `cascadir` — the reusable, dataset-agnostic package (2026-06)

The method is packaged as a standalone, dataset-agnostic library at `cascadir/`
(`cascadir/src/cascadir/`, its own `pyproject.toml`, tests, examples). Use it to re-run the
whole method on a **new dataset** without touching the research scripts — it's pure in-memory
(no cluster/SLURM/filesystem).

**For a new dataset, read `cascadir/MANUAL.md` first** — the agent-facing guide: data
contract, the one-call `CascadeDirection`, the **two coupling paths + which to use per
dataset** (the Oes/Sheu lesson), the donor-level/over-power caveat, a step-by-step recipe,
and the validated calibration numbers (88/86/83%).

- One call: `cd.CascadeDirection(condition_col=…, donor_col=…, celltype_col=…,
  control_label="PBS").fit(adata)`.
- Coupling path 1 — latent geometry: `est.discover_axes()` (broad panel + many donors).
- Coupling path 2 — signature space: `est.signature_coupling(donor_level=True)` (the §28
  reframe — `coupling` = degree-corrected `M+Mᵀ` off the cross-engagement matrix; use on
  targeted panels / few donors). Its `cross_asym` column is the `M−Mᵀ` approximation, not
  the direction statistic — see §28.
- Direction: `est.direction_table()` (cross_asym; the validated 88/86/83% output). **This
  is the only direction API** — do not read direction off the coupling table.

**Perf (results-preserving, on by default): `TrainConfig.cache_frozen_embeddings=True`.**
With a frozen encoder (the default/validated regime) Stage-2 re-runs the identical
`encoder(X)` on every tube every epoch. The flag pre-encodes each tube **once** (shared
across all per-condition binary models via `train.build_frozen_embedding_cache`) and trains
the attention/classifier head on the cached embeddings (`AbMil.forward_from_H`). The encoder
has no stochastic/mode-dependent layers, so the trained models and IG signatures are
**bit-identical** to the un-cached path — a pure speedup (encoder MLP dominates the FLOPs;
IG is untouched, still running the full model from gene inputs). Auto-bypassed when
`encoder_frozen=False`; set `False` only to A/B verify. Covered by
`cascadir/tests/test_embedding_cache.py`.

`cascadir` mirrors the research code in `cytokine_mil/analysis/…` but is decoupled from the
cluster. When the method changes, update both (and `cascadir/MANUAL.md`).

---

## 30. Disease-progression extension — cascade direction on the COVID-Haniffa scRNA atlas (2026-06)

**Headline.** Extend the validated `cross_asym` cascade-**direction** method (§26/§28) from
cytokine perturbations to **disease progression**, on a public scRNA atlas. The novelty: recover
the **direction of a progression from a single cross-sectional snapshot**, validated against the
known clinical ordering — the disease analog of the cytokine cascade-direction result. This is the
§28 signature-coupling path on real cells; there is **no MIL collapse** (real single cells) and
**no forecast/Cox/EHR machinery** (cross-sectional data has no per-subject future — the earlier
UK-Biobank/MIMIC EHR plan was aborted as gated/un-downloadable).

**Dataset.** Stephenson et al. 2021 "Haniffa" COVID-19 PBMC atlas (public, no-auth; download
`scripts/download_covid_haniffa.sh` → `datasets/COVID_Haniffa/raw/haniffa21.processed.h5ad`,
7.2 GB, ~647K cells, log-normalized + author annotations). obs: a severity field
(`Status_on_day_collection_summary`: Healthy / Asymptomatic / Mild / Moderate / Severe / Critical
+ LPS + Non_covid arms — auto-detected/validated at prepare time), `patient_id`, and cell-type
annotations `initial_clustering` (coarse, primary) / `full_clustering` (fine, robustness).

**Knob mapping.** condition = COVID **severity grade** (the 5 ordered grades; drop LPS/Non_covid);
control_label = `Healthy`; donor = `patient_id`; cell type = `initial_clustering`. **Direction
oracle** = the clinical severity order — all C(5,2)=10 ordered grade pairs, less-severe = upstream
(expected `cross_asym` sign); the 4 adjacent pairs are the cleanest.

**Two design subtleties (both real, both mitigated).**
1. **Nested donors.** Each patient has exactly ONE grade → donors are nested in conditions (unlike
   the cytokine data). Pooled `cross_asym` `M[a,b]=mean(grade-a cells on S_b)−mean(Healthy cells on
   S_b)` and `CascadeDirection.fit().direction_table()` still work (population statistic; control =
   healthy donors). But per-donor-paired `signature_coupling(donor_level=True)` and Path A
   `discover_axes` do **NOT** apply (no within-donor pairing / no per-donor control centroid).
   Donor-level rigor therefore comes from a **donor-bootstrap** (resample donors within each grade
   and within Healthy; recompute pooled `cross_asym`), unit = donor — `cascadir.progression`.
2. **Magnitude confound.** Severity is monotone-intensity (severe ≈ "more of" mild) → grade
   signatures overlap on a shared inflammation program and `cross_asym` could reflect magnitude,
   not a genuine "seed." Mitigations: report `cross_accuracy` vs the **symmetric `directional_score`
   control** (`score_directions`; if cross ≫ dirscore≈chance, direction is real beyond magnitude);
   a synthetic apparatus that explicitly tests a **monotone-intensity ladder**; per-cell-type sign
   consensus (a real seed is consistent across cell types).

**Method (reuse).** Fit-from-h5ad: `CascadeDirection(condition_col="severity",
donor_col="patient_id", celltype_col="initial_clustering", control_label="Healthy").fit(adata,
assume="auto").direction_table()` + `.benchmark(labels)`. Single GPU ~4 h for this scale.

**New reusable helpers** → `cascadir/src/cascadir/progression.py` (+ `tests/test_progression.py`):
`bootstrap_cross_asym` (nested-donor donor-bootstrap CI), `recover_order` (Borda / topological sort
of pairwise `cross_asym` → a total order) + `kendall_tau` vs the true order.

**Apparatus GO/NO-GO** (`scripts/apparatus_cross_asym_ladder.py`, CPU, before trusting real
results): (A) **distinct-program** ladder — `cross_asym` MUST recover the order (hard gate);
(B) **monotone-intensity** ladder — report whether `cross_asym` recovers order or is
magnitude-fooled (calibrates the COVID interpretation). Uses `cytokine_mil/data/synthetic_cascade_sim.py`
+ `coupling_direction`.

**Analysis + figures** (`scripts/analyze_covid_progression.py`): `score_directions` vs the oracle;
donor-bootstrap; `recover_order` + Kendall τ; 7 figures (direction-accuracy bar cross_asym vs
symmetric control; per-pair cross_asym bar; grade×grade cross-engagement heatmap; per-cell-type
sign consensus; signature scatter for the cleanest adjacent pair; **severity-order recovery** ladder
vs truth — the headline; donor-bootstrap CIs). Reuses `scripts/make_report_figures.py` (fig9/fig10)
and `cytokine_mil/analysis/eda_pair_plots.py`.

**Cluster orchestration — SLURM DAG** `slurm/covid_progression/` + `submit_covid_dag.sh`
(`sbatch --parsable` + `--dependency=afterok:$JOBID`; `SUBMIT=echo` dry-run; mirrors
`slurm/group_u/submit_group_u_dag.sh`): `apparatus` (CPU, independent) ∥ `prepare` (CPU) →
`fit` (GPU) → `analysis` (CPU, depends on `afterok:fit:apparatus`). Outputs under
`results/covid_progression/` (figures in `plots/`).

**Pre-registration** (`reports/covid_progression/PRE_REGISTRATION.md`, committed BEFORE the analysis
job runs — §25.1): the 10 ordered grade pairs + expected signs; P1 apparatus distinct-gate ≥ 90%,
P2 monotone-ladder characterization, P3 cross_accuracy ≫ symmetric dirscore on COVID, P4 donor-
bootstrap CI excludes chance + Kendall τ ≥ 0.6. **GREEN** = cross_accuracy ≥ 0.8 AND dirscore
clearly below AND bootstrap CI excludes 0.5 AND τ ≥ 0.6 AND apparatus distinct-gate passed;
**AMBER**/**RED** scale down. Honest results → `reports/covid_progression/COVID_PROGRESSION_RESULTS.md`.

**Validity boundaries (honest caveats).** Cross-sectional (no per-subject future — direction, not
forecast); nested donors (donor-bootstrap, not within-donor pairing); severity = one-disease
monotone axis (magnitude confound; this is "state-of-health → state-of-health", NOT "disease A →
disease B"); PBMC blood only; direction ≠ causation; single dataset.

**File layout (new).** `cascadir/src/cascadir/progression.py` (+test); `scripts/{download_covid_haniffa.sh,
prepare_covid_haniffa.py, run_covid_cascadir.py, apparatus_cross_asym_ladder.py,
analyze_covid_progression.py}`; `slurm/covid_progression/{apparatus,prepare,fit,analysis}.slurm`
+ `submit_covid_dag.sh`; `reports/covid_progression/{PRE_REGISTRATION,APPARATUS_GATE_RESULTS,
COVID_PROGRESSION_RESULTS}.md`.

---

## 31. Recurrent IG over training dynamics — gene-recruitment trajectories (2026-06)

**Headline.** Run Integrated Gradients **every 10 epochs** of binary-MIL (full-model,
Stage-2) training instead of once on the final model, turning each static signature `S_X`
into a **gene-recruitment trajectory**: *when* each gene enters (and leaves) a cytokine's
top-50. Tests whether recruitment **order** carries biology — a within-model "primary
anchor vs secondary relay" analog of §8.3, a temporal view of the §22/§28 shared-activation
confound, and an *independent* (timing-based) corroboration of the `cross_asym` direction
call (§26). Full hypotheses + falsifiable predictions: `hypotheses/recurrent_training_dynamics_IG.md`.

**Scope:** Oesinghaus only, all 45 cytokines in `cytokine_axes.csv`, 3 seeds (42/123/7),
single shared Stage-1 encoder per seed + the wide config (the §27.6 reproduction lesson).
Direction-not-existence and not-causation caveats (§26.4) carry over. The recurrent IG is a
**read-out** trajectory over a FROZEN encoder (the gene→feature map is fixed), so recruitment
order is the attention/classifier learning to weight features, not the representation drifting.

### 31.1 cascadir option (opt-in; does not overwrite the default path)
`TrainConfig.checkpoint_ig_every_n_epochs` (+ `checkpoint_ig_top_n`); `train_binary_mil` /
`train_all_binary` gain `checkpoint_every` / `on_checkpoint(_factory)` hooks;
`CascadeDirection.fit(ig_checkpoint_every=N)` captures `self.signature_trajectories`, exposed
via `signature_trajectory_table()` and `coupling_trajectory()` (per-epoch degree-corrected
cross-engagement panel = the "panel matrix correction", reusing `cross_engagement_matrix` +
`_degree_center` unchanged). New module `cascadir/dynamics.py`
(`derive_signature_trajectory`, `signature_trajectory_collector`, `coupling_trajectory`) +
types `SignatureCheckpoint`/`SignatureTrajectory`; `cascadir/tests/test_dynamics.py`;
MANUAL.md §3.5. All default behavior unchanged when the option is off.

### 31.2 Experiment (Part B) + analysis (Part C)
`scripts/run_recurrent_ig_oesinghaus.py` reproduces §26 training faithfully and checkpoints
the model every 10 epochs (reusing `train_mil`'s `checkpoint_dir/checkpoint_epochs`, so all
250 epochs run as one momentum-preserving optimization), then runs IG (cascadir
`integrated_gradients`) over each checkpoint → `ig_traj.parquet` (cytokine, gene, epoch, ig,
rank_ig, seed) + `final_signatures.parquet`. `scripts/analyze_recurrent_ig.py` builds the
recruitment table (`τ_in/τ_out/stab/vol/category`), tests P-A..P-E, builds the per-epoch
coupling/cross_asym panel via cascadir, runs the **§26 final-epoch direction regression
check**, renders ~9 figures, and writes the verdict.

### 31.3 Pre-registration + verdict
`reports/recurrent_ig/PRE_REGISTRATION.md` (locked BEFORE the analysis job; §25.1) fixes the
operationalizations (band top-50, persistence 0.8, Anchor/Climber thirds, shared-gene
fraction 0.25, timing-permutation n=1000) and the P-A..P-E GREEN/AMBER/RED gates. Verdict +
objective "what visually emerges" read → `reports/recurrent_ig/RECURRENT_IG_RESULTS.md`.

### 31.4 Cluster orchestration
SLURM DAG `slurm/recurrent_ig/` + `submit_recurrent_ig_dag.sh` (`sbatch --parsable` +
`--dependency=afterok`; `SUBMIT=echo` dry-run): `train.slurm` (GPU array 0-2, one seed each)
→ `analysis.slurm` (CPU 128G, afterok). Output dir `results/recurrent_ig/`
(`seed_<seed>/ig_traj.parquet`, `stats/*.csv`, `plots/*.png`).

**File layout (new).** `cascadir/src/cascadir/dynamics.py` (+`tests/test_dynamics.py`);
`scripts/{run_recurrent_ig_oesinghaus,analyze_recurrent_ig}.py`;
`slurm/recurrent_ig/{train,analysis}.slurm` + `submit_recurrent_ig_dag.sh`;
`reports/recurrent_ig/{PRE_REGISTRATION,RECURRENT_IG_RESULTS}.md`. Edits (additive, opt-in):
`cascadir/src/cascadir/{config,train,types,pipeline,__init__}.py`, `cascadir/MANUAL.md`.

---

## 32. T-cell maturation cascade — cascade direction on a SARS-CoV-2 vaccination PBMC atlas (2026-06)

**Headline.** Extend the validated `cross_asym` cascade-**direction** method (§26/§28) from
cytokine perturbations and disease progression (§30) to a **cell-state differentiation
cascade**: recover the **naive → effector → memory** T-cell direction from a single
cross-sectional snapshot, validated against the known differentiation order. This is the §30
signature-coupling/progression machinery re-pointed at a *cell-state* axis instead of a
clinical-severity axis. It is **direction VALIDATION on a gold-standard order, not discovery**
(the state order is textbook; the novelty is recovering it from a snapshot, the same epistemic
status as the §26 labeled-pair result). The cross-cell-type **relay** question (one cell type's
state driving another's fate) is a *later* track this dataset also enables — out of scope here.

**Dataset.** Multimodal SARS-CoV-2 vaccination + infection PBMC atlas (Stephenson-adjacent;
*Nat Immunol* 2023, "Multimodal single-cell datasets characterize antigen-specific CD8+ T cells
across SARS-CoV-2 vaccination and infection"). **Whole-PBMC CITE-seq** (`PBMC_vaccine_CITE.rds`,
1.6 GB; 3′ RNA + 173 TotalSeq-A surface proteins), in vivo, **day 0/2/10/28**, ~6 donors,
author cell annotations. Processed data are **free/no-auth on Zenodo `7555405`** (raw is
dbGaP-gated; we never need raw). NOTE: Zenodo ships **R Seurat `.rds`**, so a `.rds → AnnData`
conversion is the one new front-end (cascadir needs AnnData; raw is dbGaP-gated and not used).
Cluster paths: `datasets/SARSCoV2_Vaccine/raw/` (rds + `vaccine_cite_raw.h5ad`),
`…/prepared/vaccine_tcell_prepared.h5ad`.

**Knob mapping (two framings, both run).**
- **Primary — STATE.** condition = T-cell maturation state `{Naive, Effector, Memory}` (from
  CITE **surface protein** gating — CD45RA/CCR7/CD27/CD95 — preferring an author state
  annotation if present; protein labels are *independent of the RNA we score on* → breaks the
  "states defined by the same genes" circularity); control = **day-0 cells relabeled
  `"Resting"`** (kept distinct from the state conditions, exactly as `Healthy ∉ grades` in §30);
  donor = `subject`; cell type = `tcell_lineage` (CD4/CD8); **oracle = Naive→Effector→Memory**.
- **Secondary — TIMEPOINT (corroboration).** condition = `{D2, D11, D28}`; control = `D0`;
  oracle = the clock (D0<D2<D11<D28; this atlas samples Day0/2/11/28). The *weaker*
  monotone-intensity framing (per §30's
  magnitude caveat), run only to corroborate that states emerge in time order.

**Method (reuse §30 verbatim).** `cd.CascadeDirection(condition_col=…, donor_col="subject",
celltype_col="tcell_lineage", control_label=…).fit(adata, assume="auto").direction_table()`,
then `cascadir.analysis.score_directions` (cross_asym vs the **symmetric `directional_score`
control** — the magnitude check), `cascadir.progression.{bootstrap_cross_asym (nested/donor
bootstrap CI + accuracy + Kendall τ), recover_order, kendall_tau}`, and the independent
synthetic **apparatus** gate (`scripts/apparatus_cross_asym_ladder.py`). No edits to `cascadir/`
or `cytokine_mil/`.

**Honest caveats.** mRNA-vaccine memory at day 28 is **early** memory → claim
naive→effector→**early**-memory, not the full central-memory arc. ~6 donors → rigor is the
**donor-bootstrap** + cell-level **degree-corrected** coupling, NOT the 8+-donor gate (§28.2).
Magnitude confound (activation is partly monotone-intensity) → the headline is
`cross_accuracy ≫ dirscore_accuracy`. Direction ≠ causation; PBMC blood only; single dataset;
validation not discovery.

**Pre-registration** (`reports/vaccine_progression/PRE_REGISTRATION.md`, committed BEFORE the
analysis job — §25.1): the state + timepoint oracles + expected signs; P1 apparatus
distinct-gate passes; P2 `cross_accuracy ≥ 0.8` AND `≫` symmetric control; P3 donor-bootstrap
accuracy CI excludes 0.5 AND Kendall τ ≥ 0.6. GREEN/AMBER/RED scale down. Honest results →
`reports/vaccine_progression/VACCINE_PROGRESSION_RESULTS.md`.

**File layout (new; clones §30 — no edits to existing COVID/cascadir/package files).**
`scripts/{download_vaccine_multimodal.sh, convert_vaccine_rds_to_h5ad.R,
assemble_vaccine_h5ad.py, prepare_vaccine_tcell.py, run_vaccine_cascadir.py,
analyze_vaccine_progression.py}`; `slurm/vaccine_progression/{download,convert,prepare,
fit_state,fit_timepoint,analysis_state,analysis_timepoint,apparatus}.slurm` +
`submit_vaccine_dag.sh`; `reports/vaccine_progression/{PRE_REGISTRATION,
VACCINE_PROGRESSION_RESULTS}.md`. Reuses `scripts/apparatus_cross_asym_ladder.py` and all of
`cascadir.{analysis,progression}` unchanged.

### 32.1 Results (2026-06-22) — timeline recovered, state direction REVERSED (boundary found)

Ran end-to-end on the cluster (jobs 30907134–30907138). Honest writeup:
`reports/vaccine_progression/VACCINE_PROGRESSION_RESULTS.md`.

- **Apparatus gate PASS**; **TIMEPOINT (corroboration): direction RECOVERED.** `cross_asym`
  recovers D2→D11→D28 at 100% (symmetric control 0%, τ=+1.0); AMBER only because the 6-donor
  bootstrap CI is wide (underpowered), not because direction failed. Reproduces §30's
  progression result on a vaccination time axis.
- **STATE (headline): direction REVERSED (RED).** `cross_asym` recovers Effector→Memory→Naive
  — the exact reverse — consistently across CD4/CD8/other T; the symmetric control scores
  100% here (inverse of the cytokine/COVID pattern).
- **Mechanism** (confirmed by a control-swap decomposition holding signatures fixed):
  differentiation has the **opposite** cross-engagement asymmetry from a cytokine cascade —
  `cross_asym` assumes the upstream cell *acquires* the downstream program, but the mature
  cell *retains* the progenitor's program (memory re-expresses naive IL7R/TCF7/SELL/CCR7), so
  the statistic points mature→naive regardless of control choice.
- **Boundary lesson:** `cross_asym` transfers to a temporal/progression axis but is
  fundamentally mis-signed on a cell-state *differentiation* axis — an acquisition-keyed
  statistic can't recover state direction; that would need a retention/loss-keyed one.

### 32.2 Tree-cascade synthetic follow-up (2026-06-25)

`scripts/apparatus_tree_cascade.py` adds a 4th synthetic scenario to the §30 apparatus
ladder: a partial-order TREE where each node carries its own block **+ half its parent's
program** (a RETENTION structure). Result (`reports/vaccine_progression/APPARATUS_TREE_RESULTS.md`):
`cross_asym` mis-signs all 4 ancestor→descendant edges while sibling pairs stay correctly
ambiguous — confirming the reversal is about acquisition-vs-retention asymmetry, not about
the chain being linear.

---

## 33. Attention training-dynamics — cell-type-resolved cascade readout (2026-06)

**Headline.** The direction call moved from training dynamics to **Integrated Gradients on
the *final* model** (`cross_asym` on discovered signatures, §26), which discards the *order*
in which signal is learned. §33 brings *actual* training dynamics back via the **attention
layer**: every attention token is a cell whose cell type is known post-hoc, so the trajectory
of **per-cell-type attention mass over training** is a learnability-ordering readout. A 24h
snapshot (Oesinghaus) has no real time axis; a frozen-encoder MIL learns easy/direct/
receptor-driven signal first and weak/secondary/cascade signal later, so recruitment order is
a **pseudo-time** the snapshot lacks, and attention pins it to cell types. This *adds to* the
Oesinghaus paper's Fig 4 (cell-type-resolved communication) **prior-free** — it replaces the
paper's receptor-expression primary/secondary-target prior (Fig 4h) with recruitment timing.

**Scope (Core):** Oesinghaus only, **multiclass** 91-class Stage-2 (multiclass is required:
attention over all stimuli encodes *specificity* vs other stimuli, and the relay statistic
needs all stimuli in one model), 3 seeds (42/123/7), encoder frozen. Direction-not-existence
(coupling is Path A's job) and not-causation caveats (§26.4) carry over. Deferred (NOT in this
build): the attention-vs-IG **specificity** contrast / §28 over-call tie-in, and porting to
`cascadir` (validate first per §28.2).

### 33.1 Three readouts (`cytokine_mil/analysis/attention_dynamics.py`, numpy-only)
From the per-cell-type attention trajectory `A_X(T,t)`:
1. **primary vs secondary responder map** (`classify_primary_secondary`) — primary = recruited
   in the first third with high final attention; secondary = recruited in the last third AND a
   coincident `p_correct` second-rise (the model needed an additional signal). The prior-free
   analog of Fig 4h.
2. **relay-recruitment-lag direction statistic** (`relay_recruitment_lag`) — for a coupled pair
   (A,B), relay `T_B` = B's data-driven attention-primary cell type; `lag = τ(A,T_B) − τ(B,T_B)`
   per donor (τ = first checkpoint reaching `rise_frac=0.5` of own final attention); `>0 ⇒ A
   upstream`. Donor-bootstrap CI; call only if CI excludes 0. An **IG-independent, temporal**
   corroboration of `cross_asym` (Fig 4f/i: IL-12/IL-2/IL-15 → IFN-γ → monocytes).
3. **intra-cell-type attention concentration** (`concentration_summary`) — within-type Gini of
   per-cell attention over training (rising ⇒ responding subpopulation; flat ⇒ whole-population).

Plus **P1** sanity (`attention_primary_vs_groundtruth`, reuses the `EXPECTED_DOMINANT` ground
truth from `scripts/check_attention_cell_types.py`) and **P3** primacy/subtlety
(`primacy_subtlety_correlation`: Spearman of primary-cell-type τ vs a directness proxy).

### 33.2 Pipeline (reuses existing machinery)
`scripts/extract_attention_trajectory.py` (extended: per-donor output + `--hvg_path` +
within-type Gini) loads `checkpoints/epoch_*.pt`, forwards every train tube, groups attention
`a_i` by cell type, donor-aggregates → `attention_trajectory.pkl`
(`trajectory` donor-mean, `trajectory_per_donor`, `concentration`, `epochs`).
`scripts/analyze_attention_dynamics.py` consumes it + the run's `dynamics.pkl` `records`
(p_correct), computes the readouts, evaluates P1–P4, renders 4 figures, writes the verdict.
The checkpointed multiclass run is just `train_oesinghaus_full.py --checkpoint_epochs
10,20,...,250` (no trainer edit; it already saves `checkpoints/`, `label_encoder.json`,
`manifest_train.json`, `dynamics.pkl`).

### 33.3 Pre-registration + verdict
`reports/attention_dynamics/PRE_REGISTRATION.md` (locked BEFORE the cluster analysis, §25.1)
fixes `rise_frac=0.5`, first/last-third bands, the known-cascade benchmark
(IL-12/IL-2/IL-15→IFN-γ), the negative control (IL-6 / TNF-α, §28 negative), donor-bootstrap
n, and the P1–P4 gates. **Overall GREEN iff P1 (attention-primary recovers known direct
responders) and P2 (relay-lag direction) are both GREEN.** Verdict →
`reports/attention_dynamics/ATTENTION_DYNAMICS_RESULTS.md`.

### 33.4 Honest caveats
- Attention is **task-driven (discriminative)**, not biology → validate attention-primary on
  held-out donors; **late recruitment ≠ secondary** unless a `p_correct` second-rise co-occurs
  (lazy/redundant attention can leave a sufficient cell type as the only one recruited).
- **Frozen-encoder representability:** a secondary program is visible only if it lives in the
  cell-type-pretrained embedding subspace (the real risk; budget a check).
- Direction ≠ existence ≠ causation; small donor N → relay-lag CIs are wide; multi-seed before
  trusting ordering (memory: the dynamics pipeline is seed-noisy → point estimates).

### 33.5 De-risk + cluster
Local de-risk (harness only, synthetic demo — NOT biology):
`scripts/run_demo_attention_dynamics.py` runs build→Stage1→Stage2(checkpoints)→extract→analyze
end-to-end; unit tests in `tests/test_attention_dynamics.py`. Cluster (separate approved step):
SLURM DAG `slurm/attention_dynamics/{train,extract,analysis}.slurm` +
`submit_attention_dynamics_dag.sh` (mirrors `slurm/recurrent_ig/`).

**File layout (new).** `cytokine_mil/analysis/attention_dynamics.py`;
`scripts/{analyze_attention_dynamics,run_demo_attention_dynamics}.py` (+ extended
`scripts/extract_attention_trajectory.py`); `tests/test_attention_dynamics.py`;
`reports/attention_dynamics/{PRE_REGISTRATION,ATTENTION_DYNAMICS_RESULTS}.md`;
`slurm/attention_dynamics/{train,extract,analysis}.slurm` + `submit_attention_dynamics_dag.sh`.
Reuses `scripts/check_attention_cell_types.py` (`EXPECTED_DOMINANT`), `analysis/dynamics.py`
§8.3, `analysis/confusion_dynamics.py` relay/temporal helpers, `train_mil.py`,
`train_oesinghaus_full.py` (`--checkpoint_epochs`) unchanged.

### 33.6 Results (2026-07-01) — collapse NOT fixable by regularization; keeper is P3

Ran three collapse interventions then a λ-sweep, all 3 seeds, vs the baseline. **All failed
to recover the biological readout.** The entropy penalty mechanically shrinks attention
collapse (top1_share 0.42→0.06 as λ: 0→100) but (1) does **not** recover P1 (stays well below
the 0.73 training peak) — attention's argmax simply isn't the biological responder in a
frozen-multiclass task except at a transient mid-training moment; (2) trades away
discriminability with no sweet spot (p_correct falls to ≈chance at high λ); (3) also degrades
P3. Unfreezing (Stage-3) was catastrophic; cell-type hygiene was marginal. **Resolution: the
collapse is a symptom of margin-maximizing softmax, not a fixable bottleneck — do not pursue
more attention regularization.** §33's direction layer (P2 relay-lag) is a negative (0/3
cascades at every λ). The keeper is the **baseline unregularized P3** (rho≈−0.40: direct
cytokines recruit their primary cell type earlier) — a learnability-timing result,
seed-noisy, NOT a cascade-direction tool. Reports:
`reports/attention_dynamics/{INTERVENTION_COMPARISON,LAMBDA_SWEEP_COMPARISON}.md`. New
(additive): `train_mil` `attn_entropy_lambda`/`exclude_cell_types`;
`scripts/{compare_attn_experiments,probe_attention_p1_over_epochs,plot_attention_by_celltype}.py`.

---

## 34. Self-attention over cells — cell↔cell interaction as a relay readout (2026-07)

**Headline.** §33 showed the AB-MIL attention layer is a *pooling* op with **no cell-cell
interaction** (`a_i = softmax(wᵀtanh(V h_i))` depends only on cell i; cells couple only through the
softmax denominator), so it structurally cannot express "cell A informs cell B". §34 replaces that
layer with **actual self-attention over cells**: each cell attends to every other cell, yielding an
N×N (→ cell-type × cell-type) "who-influences-whom" matrix — a **relay** signal that AB-MIL and the
IG `cross_asym` (§26) both miss. Same model/hyperparameters/dataset as §33; only the attention layer
changes, so it is an apples-to-apples add-on. **Additive, NOT a replacement:** IG `cross_asym`
(§26) stays the primary coupling+direction method; §34 also reports a direction accuracy on the same
audited benchmark so it is comparable to the standing Oesinghaus 88%.

**Scope.** Oesinghaus 24h PBMC only, 91-class **multiclass**, frozen encoder, 3 seeds (42/123/7),
every-epoch checkpoints (1..250) — identical to the §33 baseline (`results/attention_dynamics`).
Direction-not-existence and not-causation caveats (§26.4) carry over. Decisions (locked): SAB→keep
AB-MIL pooling; SGD (match §33); 1 SAB layer, 4 heads.

### 34.1 Architecture (`cytokine_mil/models/set_transformer_mil.py`)
`H = encoder(X)` (frozen) → `H' = SAB(H)` (Set-Transformer Set-Attention Block: pre-LN
`nn.MultiheadAttention(128, 4 heads)` + residual + FFN + residual; exposes per-head `A[i,j]`) →
`a = AttentionModule(H')` (the **existing** AB-MIL pooling) → `z = Σ aᵢ H'ᵢ` → `BagClassifier`.
`forward(X) -> (y_hat, a, H)` matches `CytokineABMIL` **exactly** (returns the original frozen `H`,
not `H'`, so `train_mil` centroid/PBS-RC logging keeps §33 semantics), and `.encoder`,
`.encoder_frozen`, `unfreeze_encoder()` mirror it — so `train_mil` runs with **no edits**
(`isinstance(model, CytokineABMIL_V2)` is False → standard branch). Extra methods for the
frozen-reconstruction extractor: `pool_from_H(H)->(a,H')`, `interaction_from_H(H)->A` (head-avg
N×N), `forward_with_interaction(X)`. Built via `experiment_setup.build_selfattn_model` (mirror of
`build_mil_model`); selected by `train_oesinghaus_full.py --model_type set_transformer`
(default `abmil` — baseline unchanged) with `--sab_heads`/`--sab_layers`.

### 34.2 Extraction (`scripts/extract_selfattn_trajectory.py`) — reuses the frozen-encoder trick
Encoder frozen ⇒ `H` fixed across all 250 checkpoints; cache `H` once, reconstruct attention from
each checkpoint's saved params. Emits TWO outputs: (1) `attention_trajectory.pkl` — **identical
structure** to §33 using `pool_from_H`'s pooling weights as `aᵢ`, so `analyze_attention_dynamics.py`
runs UNCHANGED (P1–P4 comparable to the §33 AB-MIL baseline); (2) `interaction_trajectory.pkl` —
`M[τ,σ] = mean_{i∈τ, j∈σ} A[i,j]` (head-averaged, row-normalised, reduced to 18×18 per tube on the
fly), donor-aggregated `{cytokine -> {(τ,σ) -> array(n_epochs)}}` (+ per-donor).

### 34.3 Readouts (`cytokine_mil/analysis/attention_interaction.py`, numpy-only)
On `interaction_trajectory.pkl`: **G0 go/no-go** off-diagonal (cross-cell-type) mass over training
(if ~0 the SAB collapsed to self/diagonal → cells don't interact → premise fails; hard gate first);
**interaction asymmetry** `Asym[τ,σ] = M[τ,σ] − M[σ,τ]` → directed cell-type graph per cytokine;
**known-cascade relay corroboration** (IL-12/IL-2/IL-15→IFN-γ relay=NK/mono; negative control
IL-6/TNF-α); **interaction-recruitment timing** (reuse `attention_dynamics.celltype_recruitment`).
Driver `scripts/analyze_selfattn_interaction.py` (verdict + ~4 figures).

### 34.4 Direction accuracy vs the 88% benchmark (`scripts/score_selfattn_direction.py`)
Score the self-attention direction calls — **relay-lag** (§33 `relay_recruitment_lag`) and the new
**interaction-asymmetry** — per audited labeled pair against `expected_sign` on the
`counts_in_benchmark=True` rows of `reports/cascade_pairs/cytokine_axes_audited.csv` (the exact 15/17
denominator behind `cross_asym`), reusing `scripts/retally_pipeline_against_audit.py`'s label-load +
sign-accuracy logic. Output: a comparison table with the IG `cross_asym` **88%** reference row
(reproduced via `retally --metric cross_asym` from existing signatures if present, else cited;
`cross_asym` direction is unchanged by the §28.2 panel/degree updates, which only touch the
*symmetric* coupling gate). Averaged over 3 seeds → `reports/selfattn_dynamics/SELFATTN_RESULTS.md`.

### 34.5 Pre-registration + verdict + de-risk
`reports/selfattn_dynamics/PRE_REGISTRATION.md` (locked BEFORE the cluster analysis, §25.1): fixes
the architecture, the G0 off-diagonal-mass gate, the pooling-head P1–P4 gates (inherit §33 verbatim),
and the interaction-direction predictions for the known cascades + negative control (GREEN/AMBER/RED).
Local de-risk `scripts/run_demo_selfattn.py` (harness-only synthetic demo) + `tests/
{test_set_transformer_mil,test_attention_interaction}.py`. Cluster: SLURM DAG `slurm/selfattn/
{train,extract,analysis}.slurm` + `submit_selfattn_dag.sh` (mirror `slurm/attention_dynamics/`;
analysis runs `analyze_attention_dynamics` + `analyze_selfattn_interaction` + `score_selfattn_direction`
+ the §33 probe/plot). Output `results/selfattn/seed_*/`.

**Honest caveats.** Same as §33.4 (task-driven not biology, frozen-encoder representability,
direction ≠ existence ≠ causation, small-N/multi-seed). Deferred: porting to `cascadir`
(validate first).

**File layout (new).** `cytokine_mil/models/set_transformer_mil.py`;
`cytokine_mil/analysis/attention_interaction.py`; `scripts/{extract_selfattn_trajectory,
analyze_selfattn_interaction,score_selfattn_direction,run_demo_selfattn}.py`;
`tests/{test_set_transformer_mil,test_attention_interaction}.py`;
`reports/selfattn_dynamics/{PRE_REGISTRATION,SELFATTN_RESULTS}.md`;
`slurm/selfattn/{train,extract,analysis}.slurm` + `submit_selfattn_dag.sh`. Edits (additive,
backward-compatible): `cytokine_mil/experiment_setup.py` (`build_selfattn_model`),
`scripts/train_oesinghaus_full.py` (`--model_type`/`--sab_heads`/`--sab_layers`). Reuses
`models/{attention,bag_classifier,instance_encoder}.py`, `train_mil.py`,
`analysis/attention_dynamics.py`, `scripts/{analyze_attention_dynamics,probe_attention_p1_over_epochs,
plot_attention_by_celltype,retally_pipeline_against_audit}.py` unchanged.

---

## 35. Thesis Document & Overleaf Sync (2026-07)

Pure infrastructure/tooling, no bearing on the research pipeline. The M.Sc. thesis source
lives in its own separate git repo, sibling to this one: `/Users/yam/my-packages/overleaf-thesis-cascades/`
(independent git history, synced via Overleaf's git bridge). Only `METHOD_GROUND_TRUTH.md`
and `WONDERINGS.md` remain in `cytokine_mil/thesis/` — the thesis prose itself is not here.
See `/thesis-sync` skill for the full location rationale, git-bridge auth, and sync workflow.


---

## 36. Oesinghaus full-90 — coupling + direction on a neutral background (2026-08)

**Motivation (the gap this closes).** Every published Oesinghaus coupling number is measured
on a **24-cytokine panel whose members were selected because they appear in a literature
benchmark pair** (`reports/coupling_figures_draft/donor_coupling_hub_IG_vsPBS.csv` = all
C(24,2)=276 pairs, 76 coupled; thesis Limitations, "Which cytokines enter each panel"). The
background is therefore non-neutral: the §28.2 enrichment and over-call figures (77% → 31%)
are computed against a family assembled from the prior itself. This section fits **all 90
cytokines + PBS** in one `cascadir` fit, giving coupling and `cross_asym` for all
**C(90,2) = 4005** pairs — the first neutral background for those figures, and the first
prior-free candidate list (3729 pairs never scored).

**This is a SEPARATE fit.** 66 of the 90 cytokines have no published signature, so mixing
its numbers with the published 24/45-cytokine results would be the mixed-provenance failure
§26.3 documents for Sheu. Never average or quote them together. Path A's 121 axes, §26's
88%, and §28.2's gate validation are all unaffected by this run.

### 36.1 Locked decisions (all taken with the user before the run)
| | |
|---|---|
| Stage-2 binary epochs | **250** — see §36.2; nothing plateaus earlier |
| Tube source | the **committed** Oesinghaus pseudo-tubes, loaded into cascadir's contract (not re-sampled) |
| Hyperparameters | the published **"wide"** config (embed=512, hidden=(512,512), attn=128, Stage-1 20@0.005, Stage-2 250@3e-5), i.e. `train_oesinghaus_binary_missing16.py:108-115`, **not** `cascadir.TrainConfig`'s packaged defaults |
| Audited-pair regression check | **reported, never aborting** (§36.3) |
| Coupling FDR | **BH over all 4005** on cascadir's `donor_sign_p`, primary q=0.05, secondary q=0.10 |
| Held-out donors | D2/D3 excluded **everywhere**, Stage-1 included (§16; stricter than some earlier runs) |

### 36.2 Epoch-count evidence (checked BEFORE the run, from §31's recurrent IG)
At epoch 50 only **81%** of each final top-50 signature has ever entered the band (⇒
Jaccard(top50@50, top50@250) ≤ 0.69), the top-50 **coupled-pair set** has Jaccard **0.52**
against the final one, and 9% of `cross_asym` signs differ. Even at 200 the coupled set is
at Jaccard 0.89 and still moving. A shorter schedule is a *different result*, not a cheaper
one. The **loss/p_correct plateau is unmeasured** — no binary-MIL per-epoch trajectory is
persisted anywhere in the repo (the §31 driver saves only `ig_traj.parquet`; the
`oesinghaus_full_*` pickles are the 91-class multiclass runs). Say so, don't infer it.

### 36.3 Why the regression check does not gate
Two independent *faithful* fresh fits have already under-reproduced the published 15/17:
§27.6 at 6/11, and **§31's recurrent-IG run at 11/17 = 0.65 despite a single shared Stage-1
encoder and the wide config**. So "separate encoder per chunk" is **not** established as the
sole cause of §27.6, and a hard ~88% gate would abort on run-to-run variance. The check runs
via `est.benchmark(...)` and is reported next to the symmetric `directional_score` control —
a shortfall matters only if the control rises with it.

### 36.4 The §27.6 guard, made structural
The encoder is trained **once** (stage 1) and its `state_dict` sha256 persisted; every
training chunk recomputes the digest and **refuses to run** on a mismatch. The pseudo-tubes
are materialised **once** (stage 0) as `(donor, condition)` shards with their own sha256,
verified per chunk and again at merge. Both are assertions, not conventions —
`cascadir.build_pseudotubes` advances a single RNG over the sorted (condition, donor) pairs,
so rebuilding from a condition subset would give different tubes for the same pair.

### 36.5 cascadir API addition
`CascadeDirection.from_artifacts(tube_set, signatures, ...)` (+ `cascadir/tests/
test_from_artifacts.py`, MANUAL.md §3.1) rebuilds exactly the state `fit()` leaves, for fits
staged across jobs. It is what keeps coupling/direction on the **orchestrator**: the bare
module-level `signature_coupling()` silently falls back to the over-powered cell-level null
while still returning a `coupled` column. **Never call the module-level functions.**

### 36.6 DAG + honest reporting
SLURM DAG `slurm/oes90/` + `submit_oes90_dag.sh` (`SUBMIT=echo` → readable dry run with
synthetic ids): `prepare`(CPU 64G) → `encoder`(GPU) → `train`(GPU array 0-8%3) →
`merge`(CPU) → `coupling`(CPU 260G) → `direction`(CPU 180G) → `analysis`(CPU), each with an
`afternotok` sentinel writing `results/oes_full90/STATUS.md`, plus a self-resubmitting
watchdog appending to `HEALTH.md` every 30 min. Memory asks follow from one full tube-set
copy ≈ 64 GB (9,100 tubes × ~438 cells × 4000 genes × 4 B): direction holds 2 copies,
coupling 3 (it adds the per-donor `cells_by_pair` dicts, `pipeline.py:411-414`).

**Reporting caveat that must survive into any writeup:** the *pair family* is neutral, but
the *literature labels* are not — only the 121 axes in `cytokine_axes.csv` carry an
adjudicated `literature_status`, and those were selected by Path A on the old panel.
`RESULTS.md` therefore reports enrichment **two ways** (full-4005, treating unlabeled as
unsupported — an explicit lower bound; and within the labeled subset) and never quotes
either as the §0 "~50% vs ~1%" number. `retally_pipeline_against_audit.py` is **not** usable
here (it consumes a per-cell-type CSV `direction_table` does not emit); `est.benchmark`
replaces it.

**File layout (new).** `scripts/{_full90_config,_full90_estimator,prepare_oesinghaus_full90,
train_oesinghaus_full90_encoder,train_oesinghaus_full90_chunk,merge_full90_signatures,
run_oesinghaus_full90_coupling,run_oesinghaus_full90_direction,analyze_oesinghaus_full90,
run_demo_full90_pipeline}.py`; `cytokine_mil/analysis/full90_tube_io.py`;
`tests/test_full90_tube_io.py`; `cascadir/tests/test_from_artifacts.py`;
`slurm/oes90/{prepare,encoder,train,merge,coupling,direction,analysis,sentinel,
watchdog}.slurm` + `submit_oes90_dag.sh`; `reports/oesinghaus_full90/RESULTS.md`.
**Edited (additive):** `cascadir/src/cascadir/pipeline.py`, `cascadir/MANUAL.md`.

---

## 37. Oesinghaus 90-cytokine PURE run — the cytokine-agnostic re-fit (2026-08)

**Motivation.** §36's full-90 run completed cleanly but its signatures came out **2.7×
less cytokine-specific** than the published ones (mean between-cytokine Jaccard 0.178 vs
0.065 on the *same* 24 cytokines; median Jaccard against the published top-50 = 0.064;
`IKZF2` in 73 of 90 signatures). Both downstream statistics inherit that: direction fell to
7/12 non-AMBIGUOUS (one-sided binomial **p = 0.39** — not distinguishable from chance) and
coupling reproduced only 17 of the published fit's 76 calls (Spearman ρ = 0.42).
Signature specificity is the method's linchpin (§26.4), so those numbers are a *consequence*
of the signatures, not an independent regression in `cross_asym`.

**The finding that redirected the work.** The published 88% anchor (`binary_ig_all24`) is a
**merge of two runs** (`results/oesinghaus_binary/` + `…_missing16/`) whose Stage-1 encoders
were each trained on **17–18 conditions** — subsets of the very cytokines the benchmark
scores — with **D2/D3 included** in Stage 1 (`train_oesinghaus_binary_missing16.py:150` applies
`VAL_DONORS` only to the Stage-2 binary split). Two consequences, both recorded here:
1. **§27.6's diagnosis is contradicted by its own anchor.** "Separate Stage-1 encoder per
   chunk" cannot be the sole cause of the 45-cytokine failure, because the 88% anchor has
   exactly that. §31's recurrent-IG run (single shared encoder, wide config) scoring 11/17
   already pointed the same way.
2. **Any diagnostic that reproduces the published Stage-1 setup reproduces its test-data
   leakage.** That killed the narrow-encoder comparison that was the obvious next step.

**What this run is.** A preregistration-style, **cytokine-agnostic** re-fit: every cytokine
carries equal weight at every stage, and no stage may read the audited pair list or the
published 24-cytokine panel. **There is deliberately no analysis stage** — it produces
artifacts only; scoring waits for a committed pre-registration (§25.1).

**This is a separate fit.** Never average or mix its numbers with the published 24/45-cytokine
results, nor with §36's `results/oes_full90/` (§26.3, the Sheu mixed-provenance lesson).

### 37.1 Locked decisions (taken with the user before the run)
| | |
|---|---|
| Encoder width | **2× the published wide config**: `embed_dim=1024`, `hidden_dims=(1024,1024)` ≈ 6.2 M params |
| Signature size | **`top_n=100`** (was 50) |
| Encoder early stopping | 10% of cells held out **stratified by cell type**; stop on val-loss patience but **keep running past the plateau** and record it; the encoder used downstream is the **best-val checkpoint** |
| Tube budget | **k=4 tubes per (donor, cytokine)** (`tube_idx 0–3`) = 3 640 tubes, plus a **disjoint reserve** (`tube_idx 4–7`) used *only* for the signature-stability check |
| Tube usage | **one fixed sample throughout** (§37.2) |
| Binary epochs | **250, unchanged** — a recorded deviation (§37.3) |
| Coupling | `signature_coupling(donor_level=True, degree_correct=True)` + BH-FDR over all 4005 |
| Held-out donors | D2/D3 excluded **everywhere**, Stage-1 included (§16) |
| Tube source | the **§36 shards reused read-only** and sha-verified — never rebuilt |

### 37.2 Why one fixed tube sample is enough (and what replaces a split)
`cross_asym` compares `s_T(a, S_b)` against `s_T(b, S_a)`. `S_b` comes from b's binary model,
trained on b's tubes plus PBS — **a's cells never entered S_b's derivation**, and symmetrically.
Same for coupling's `M[a,b]`. The cross terms are **structurally out-of-sample already**; the
only genuinely shared component is PBS (training negative, IG baseline, and engagement
normaliser alike), which a train/eval split would not remove either.

What in-sample derivation *does* leave open — whether `S_X` is memorised tube idiosyncrasy — is
measured directly instead: every signature is derived **twice**, once from the k=4 training
tubes and once from the disjoint reserve, and `signature_stability.csv` reports the per-cytokine
Jaccard. One extra IG pass over already-saved models; degrades nothing.

### 37.3 Recorded deviations (state these wherever the run is reported)
- **Gradient steps.** `_epoch_megabatches` (`cascadir/src/cascadir/train.py:241`) sets
  steps-per-epoch = tubes-per-class. Published: 100 tubes/condition × 250 epochs = **25 000
  steps**; here 40 × 250 = **10 000**. Epochs were deliberately not rescaled, so training
  length is a third changed variable alongside encoder width and `top_n`.
- **Encoder stopping point (added 2026-08-25).** §37 early-stops Stage 1 on val loss and
  restores the best checkpoint, which landed at **epoch 4** (val_loss 0.245, rising to 0.607
  by epoch 34 while train_acc hits 1.0). The published anchor and §36 both ran a **fixed 20
  epochs** with no validation split. This is a fourth changed variable and was missing from
  the original list.
- **Four** variables move at once (width, `top_n`, training length, encoder stopping point)
  — this run cannot attribute a change to any one of them, by design. It establishes what
  the method does under a leakage-free protocol, not why it differs from the published fit.

### 37.4 The agnosticism guard (mechanical, not a convention)
`scripts/_oes90_pure_config.py` is written from scratch and does **not** import
`_full90_config`, so `AUDITED_CSV` / `PUBLISHED_COUPLING_CSV` / `load_audited_labels` are not in
scope. `C.assert_agnostic()` runs at the top of every stage and fails if `_full90_config` is in
`sys.modules`. The condition list comes only from the tube manifest (sorted, 90 + PBS), and the
main/reserve split uses the **same** `tube_idx` values in every (donor, condition) group, so it
encodes no per-cytokine choice. `run_demo_oes90_pure.py` asserts the guard holds end-to-end.

### 37.5 Provenance guards (structural, not conventional)
Three digests, each an assertion that refuses to run on mismatch: the **encoder** state_dict
sha256 (§27.6 — the array can never shard encoder training), the **tube shard** digest, and the
**embedding cache** digest (so models are provably trained on the embeddings that were saved).
Saved model heads carry the encoder sha and refuse to be recombined with a different encoder.
`merge_oes90_pure_signatures.py` aborts unless all nine chunks agree on encoder and tubes.

### 37.6 cascadir changes (additive, opt-in, defaults bit-identical)
`cascadir/src/cascadir/train.py`:
- `train_encoder(..., val_fraction, patience, min_delta, extra_epochs_after_stop,
  return_history)` — cell-type-stratified validation split, early stopping that runs past the
  plateau, best-val restore, per-epoch history (`history.attrs` carries `best_epoch`,
  `stopped_epoch`, `last_state_dict`).
- `train_binary_mil(..., return_history)` — per-epoch mean mega-batch loss.
- `train_all_binary(..., embedding_cache, return_history)` — accept a **prebuilt** cache and
  forward the history flag.
Locked by `cascadir/tests/test_train_history.py` (defaults bit-identical; prebuilt cache ≡
internally built). `cytokine_mil/analysis/full90_tube_io.py` gains `tube_indices` on
`load_tube_set` plus `save_embedding_cache` / `load_embedding_cache` (pure I/O).

### 37.7 Hard-rule note
Coupling and direction go only through the orchestrator (`from_artifacts` →
`signature_coupling` / `direction_table`). One module-level exception, used to **persist
values, never to make a call**: `cascadir.cross_asym.directional_asymmetry_test` (a public
export) supplies the per-cell-type `sA_PB_norm` / `sB_PA_norm` engagement numbers that the
summary tables median away, and there is no other API for them.

### 37.8 DAG
`slurm/oes90_pure/` + `submit_oes90_pure_dag.sh` (`SUBMIT=echo` dry run): `prepare`(CPU 64G) →
`encoder`(GPU) → `encode`(GPU) → `train`(GPU array 0-8%3) → `ig`(GPU array 0-8%3) →
`merge`(CPU) → `coupling`(CPU 110G) → `direction`(CPU 80G), each with an `afternotok` sentinel
writing `results/oes90_pure/STATUS.md`, plus the 30-minute self-resubmitting watchdog appending
to `HEALTH.md`. Memory follows from one k=4 tube-set copy ≈ 25 GB (0.4 × §36's 63.5 GB):
direction holds ~2 copies, coupling ~3.

**Deliverables** (`results/oes90_pure/`, gitignored): `encoder.pt` + `encoder_last.pt` +
`encoder_history.csv`; `models/<cond>_head.pt` × 90 + `history/<cond>_train.csv` × 90;
`embeddings/` (the encoded pseudo-tubes); `signatures_{main,reserve}.parquet` (top-100) +
`signature_stability.csv`; `coupling_donor_degree.csv`; `direction_table.csv`;
`engagement_per_celltype.parquet`.

**File layout (new).** `scripts/{_oes90_pure_config,_oes90_pure_estimator,prepare_oes90_pure,
train_oes90_pure_encoder,encode_oes90_pure_tubes,train_oes90_pure_chunk,ig_oes90_pure,
merge_oes90_pure_signatures,run_oes90_pure_coupling,run_oes90_pure_direction,
run_demo_oes90_pure}.py`; `slurm/oes90_pure/{prepare,encoder,encode,train,ig,merge,coupling,
direction,sentinel,watchdog}.slurm` + `submit_oes90_pure_dag.sh`;
`cascadir/tests/test_train_history.py`.
**Edited (additive):** `cascadir/src/cascadir/train.py`, `cascadir/MANUAL.md`,
`cytokine_mil/analysis/full90_tube_io.py`, `tests/test_full90_tube_io.py`.

---

## 38. Why the Oesinghaus-90 fits degrade — the Stage-1 encoder breadth sweep (2026-08)

**The diagnosis.** §37's `cross_asym` collapsed into an additive potential `θ_a − θ_b` (81%
of ‖cross_asym‖², 88.8% of all 4005 signs from one scalar per cytokine), with θ turning out
to be response *amplitude*, not direction. Tracing that back gives a clean ladder, measured
on the **identical 24 cytokines at the identical top-50 cut**:

| fit | conds seen by encoder | enc epochs | width | k | meanJ | distinct/1200 | top-5 pool | worst shared gene |
|---|---|---|---|---|---|---|---|---|
| published anchor | 17–18 (+D2/D3) | 20 | 512 | 10 | **0.065** | 504 | 81 | LEF1 4/24 |
| §36 full90 | 90 | 20 | 512 | 10 | **0.178** | 307 | 54 | ANK3 10/24 |
| §37 PURE ep250 | 90 | 4 (best-val) | 1024 | 4 | **0.241** | 261 | 40 | SLC8A1 14/24 |
| §37 PURE ep50 | 90 | 4 (best-val) | 1024 | 4 | **0.301** | 217 | 33 | SLC8A1 20/24 |

Chance meanJ at top-50 is 0.006. **The largest step is published → §36, where the only
change was the Stage-1 encoder's training set.** All of §37's hyperparameter changes
together added only the smaller 0.178 → 0.241 step. This is causally tight: a binary model
trains on `{X tubes, PBS tubes}` alone (`cascadir/src/cascadir/train.py:469-470`), so
condition count cannot reach a signature except through the encoder.

**Mechanism.** Stage 1 optimises the encoder for **cell-type classification**, for which
cytokine-induced variation is nuisance variance *within* a cell type. The more conditions it
sees, the more perturbation diversity it is explicitly trained to be invariant to — so it
discards exactly the signal the method needs. Downstream, all binary heads sit on that frozen
representation and find the same few residual directions, and IG returns the same genes for
every cytokine. Only **557 of 4000 genes** ever enter a §37 top-100; the collapse is worst at
**ranks 0–4** (`SLC8A1` in the top-5 of 56 of 90), and the shared pool is myeloid/NK identity
and long-locus genes, not cytokine response. Those same genes sit at the *opposite* end of
the published ranking (median rank `SLC8A1` 3722/4000) — an attribution sign flip, not a
re-ordering.

**Ruled out** (each measured, not argued):
- **`top_n=100`** — truncating §37 to top-25 makes diversity *worse* (meanJ 0.254 vs 0.310).
- **Condition count as a direct effect** — restricting §37's own output to 24 cytokines still
  leaves 4.6× collapse vs published's 2.4×.
- **Over-training** — the epoch-50 re-run (`results/oes90_pure_ep50/`) is worse on every
  measure. The shared axis is what the model finds *first*; longer training partially dilutes
  it. A second-order version survives: low final loss predicts a worse signature
  (ρ(loss_final, frac_up) = +0.53), absent at epoch 50 (ρ = +0.11, p = 0.3).
- **Cell-level memorisation** — §37's reserve-tube signature stability is median Jaccard
  **0.942**. Any memorisation is at the well/donor level, where the reserve gives no
  independence.
- **An implementation or convention bug** — both paths rank with `np.argsort(-ig_mean)`
  (`cascadir/src/cascadir/signatures.py:135`, `scripts/run_binary_ig_probe.py:353`), and IG
  peakedness is identical (median rank0/rank99 13.15 vs 13.25).

### 38.1 The sweep
Four Stage-1 encoder arms, **nested** (`rand18 ⊂ rand45 ⊂ all90`), with everything else
pinned at the published values (512-wide, hidden (512,512), Stage-1 **20 epochs, no early
stopping**, lr 0.005, k=**10** tubes, top_n=**50**, Stage-2 250 @ 3e-5):

| arm | encoder trained on |
|---|---|
| `pbs_only` | PBS cells only — zero invariance pressure; precedent in §2.5/§2.7 |
| `rand18` | seeded-random 18 of 90 — published breadth, no benchmark knowledge |
| `rand45` | seeded-random 45 of 90 |
| `all90` | all 90 (= §36's encoder regime) |

**Total Stage-1 cell count is held fixed across arms** (target 36 000, split evenly over the
groups each arm uses; PBS is present in every arm and counts as a group). Without this,
breadth would be confounded with gradient exposure. If `pbs_only` cannot supply the target,
prepare lowers the budget for **all** arms and records it.

**Panel:** one seeded-random 24 of the 90 (`PANEL_SEED`), fixed across arms — never a
benchmark list. **Readout:** signature diversity only — top-5 gene-pool size and max shared
top-5 gene (primary; the collapse is worst at the top of the ranking, so this is the sharpest
measure), mean pairwise Jaccard and distinct-gene count at top-50 (secondary), plus `frac_up`
and final training loss. **No coupling, no direction, no benchmark scoring** — this decides
which encoder to fit with, not what the biology is.

**Decision rule.** Diversity rising monotonically as breadth falls confirms breadth is causal,
and the production re-run adopts the best arm — preferring `pbs_only` if it wins, since it is
canonical, uses zero cytokine cells, and avoids a random subset's arbitrariness. If all four
arms look like §37, breadth is not the lever and the remaining suspects are encoder width,
tube count k, and the published anchor's Stage-1 leakage.

### 38.2 Guards
`scripts/_encsweep_config.py` is standalone (does **not** import `_full90_config`) and reuses
only `_oes90_pure_config`'s file/digest plumbing, so `assert_agnostic()` stays meaningful. The
demo runner additionally enforces a **static AST check** that no stage references a benchmark
artefact, carrying a positive control so a broken check cannot pass silently. Each arm's
encoder is sha256-guarded, and a head refuses to be recombined with a different arm's encoder
(the §27.6 guard, now exercised across arms).

**Honest limits.** Even the published fit is 10× above chance overlap — this recovers a
known-good operating point, it does not make signatures clean in absolute terms. A `rand18`
production encoder makes the whole fit depend on which 18 were drawn (hence the `pbs_only`
preference, or several subset seeds). `pbs_only` risks distribution shift: an encoder that has
never seen a stimulated cell may extrapolate badly. And this explains the degradation *between
fits* — it does not establish that the published 88% is correct, since that anchor still has
Stage-1 leakage. Direction ≠ existence ≠ causation (§26.4) carries over.

**File layout (new).** `scripts/{_encsweep_config,prepare_encsweep,train_encsweep_encoder,
train_encsweep_chunk,ig_encsweep,analyze_encsweep,run_demo_encsweep}.py`;
`slurm/encsweep/{prepare,encoder,train,ig,sign,analysis,sentinel,watchdog}.slurm` +
`submit_encsweep_dag.sh`. **Edited (additive, backward-compatible):**
`scripts/prepare_oes90_pure.py` (`verify_shards(..., need_indices=)`),
`scripts/_oes90_pure_estimator.py` (`save_model_head` records the model's actual attention
width instead of a config constant), `scripts/diagnose_oes90_pure_signature_sign.py`
(`--conditions`). Reuses the §36 tube shards read-only.

### 38.3 Results (2026-08-26) — breadth FALSIFIED; the lever is Stage-1 *volume*, not breadth

Ran end-to-end (jobs 31378970–31378983). `prepare` confirmed the budget holds: 38,784
unique PBS cells available against a 36,000 target, so all four arms got 35,490–35,910
cells (within 1.2% of each other) over the same 18 cell types. The secondary `sign` stage
failed; `analysis` was wired `afterok:$IG,afterany:$SIGN` precisely so that could not cost
the primary readout, and it did not.

**The ladder is flat. There is no breadth effect.**

| arm | encoder conds | top-5 pool | worst top-5 gene | meanJ | collapse |
|---|---:|---:|---|---:|---:|
| `pbs_only` | 1 | 40 | DOCK4 13/24 | 0.257 | 4.3× |
| `rand18` | 19 | 39 | ZEB2 16/24 | 0.203 | 4.0× |
| `rand45` | 46 | 44 | ANK3 17/24 | 0.237 | 4.0× |
| `all90` | 91 | 44 | LRMDA 10/24 | 0.180 | 4.0× |

Not merely non-monotone — **inverted**: `all90` (the supposedly-broken regime) is the *most*
diverse by meanJ, and `pbs_only` (the candidate fix) the *least*. The §38.1 decision rule's
third branch fires: all four arms look like §37 (4.0–4.3× vs §37's 4.6×), none approaches
published's 2.4×. The `seen`/`unseen` contrast is contradictory across arms (rand18 seen
0.148 vs unseen 0.218; rand45 seen 0.264 vs unseen 0.216), so there is no familiarity
effect either.

**The gap to published is real, not a panel artifact.** The sweep's seeded-random 24 is
enriched for weak PBMC responders (GDNF, PRL, VEGF, IL-31, IL-17E …), which could inflate
overlap on its own. Restricting both sides to the **6 cytokines the two panels share**
(CD30L, GM-CSF, IL-10, IL-13, IL-6, VEGF) removes that confound and the gap survives:
published meanJ **0.058** (232 distinct genes, collapse 1.29×) against 0.123–0.285 for the
arms (139–197 genes, 1.52–2.16×). Computed with `analyze_encsweep.diversity`, same function
both sides.

**What this rules out.** The arms pinned embed 512, hidden (512,512), Stage-1 20 epochs with
no early stopping, k=10, top_n=50, Stage-2 250 @ 3e-5 — all at published values. So encoder
**width**, **tube count k**, **epochs**, **top_n**, and **condition breadth** are now all
controlled and none of them explains the collapse. Combined with §38's earlier eliminations
(over-training, cell-level memorisation, ranking-convention bugs), the surviving suspects
are few.

**What remains, largest first.** Published Stage-1 does not resemble any sweep arm in
*volume* or *donor structure*: `build_stage1_manifest` (`cytokine_mil/experiment_setup.py`)
takes **one tube per cytokine** at `tube_idx == 0`, rotating donors. For the missing16 run
that is 16 cytokines + PBS = **17 tubes ≈ 7.5K cells**, each cytokine contributed by exactly
**one** donor — against the sweep's 36K cells balanced over all 10 donors. Three uncontrolled
variables, in order of size:
1. **Stage-1 cell volume** — ~7.5K vs 36K, a 5× difference. The sweep pinned this at 36K
   deliberately (so breadth would not be confounded with gradient exposure) and in doing so
   pinned it far from the published value.
2. **Stage-1 donor confounding** — published entangles condition with donor by construction;
   the sweep balances them. A donor-balanced encoder can learn cell type *invariantly to
   donor*, which is a cleaner cell-type detector.
3. **D2/D3 in Stage-1** — published includes them (`train_oesinghaus_binary_missing16.py:150`
   applies `VAL_DONORS` only to the Stage-2 split); every sweep arm excludes them per §16.

(1) and (2) both point the same way as the already-measured second-order result
**ρ(loss_final, frac_up) = +0.53**: the *better* Stage 1 fits the cell-type task, the *worse*
the signature. A less-fit encoder retains more within-cell-type nuisance variance — which is
exactly the cytokine signal. That reframes the mechanism from "how many conditions the
encoder sees" to "how completely the encoder solves cell-type classification", and it is
still a settings knob, not a method change.

**Honest status.** This does not rescue the method and does not validate the published 88%
(that anchor still has Stage-1 D2/D3 leakage). It removes five candidate explanations and
leaves a sharper, cheaper one. Direction ≠ existence ≠ causation (§26.4) carries over.

### 38.4 The Stage-1 CONSTRUCTION sweep (2026-08-26) — volume, donor structure, leakage

§38.3 falsified breadth while holding Stage-1 cell count at 36K. What it never varied is
what the published anchor actually does differently: `build_stage1_manifest`
(`cytokine_mil/experiment_setup.py`) takes **one tube per condition** at `tube_idx == 0`,
rotating donors, so each condition is contributed by exactly **one** donor and the total is
~17 tubes (~7.5K cells) — not 36K balanced over ten donors. Three variables were riding on
that, and this sweep separates them. Everything else stays pinned at the published values
and the readout panel is the **same seeded-random 24** as §38.3, so the arms are directly
comparable to that table.

| arm | Stage-1 construction |
|---|---|
| `pub_replica` | one tube per condition, rotating donors, **D2/D3 INCLUDED** |
| `pub_replica_clean` | one tube per condition, rotating donors, D2/D3 excluded |
| `vol_small` | donor-balanced, ~7.5K cells (the published Stage-1 magnitude) |
| `vol_large` | donor-balanced, 36K cells (= §38.3's `all90` regime) |

Each pair moves exactly one variable:

| contrast | isolates |
|---|---|
| `pub_replica` vs `pub_replica_clean` | **D2/D3 leakage** — how much of the published anchor's advantage is held-out donors in Stage 1 |
| `pub_replica_clean` vs `vol_large` | **donor structure** — condition⊗donor entangled vs balanced, at comparable volume |
| `vol_small` vs `vol_large` | **Stage-1 volume** — 7.5K vs 36K, at matched structure |

The replica arms use **all 91 conditions**, not the published 16: breadth is settled
(§38.3), so spanning every condition keeps the run agnostic *and* decouples structure from
volume, which a 17-condition replica would confound.

**`pub_replica` deliberately violates §16** by putting the held-out donors into Stage 1.
That is the measurement, not an oversight — it is the only way to size the leakage in the
88% anchor. Its artefacts are diagnostic and **must never seed a production fit**; the
arm records `includes_val_donors: true` in the meta, and Stage-2 tubes still exclude D2/D3
in every arm.

**Hypothesis under test.** Volume and donor structure both act through the same mechanism
as the already-measured ρ(loss_final, frac_up) = +0.53: the *better* Stage 1 solves
cell-type classification, the *worse* the signature, because cytokine response is
within-cell-type nuisance variance to that objective. Less data, or donor-confounded data,
leaves a less complete cell-type detector and more residual perturbation signal. If the
volume and structure contrasts are both flat and only the leakage contrast moves, then the
published anchor's specificity was substantially leakage — a far more serious finding, and
one that bears directly on the 88%.

**Implementation.** New: `scripts/prepare_s1sweep.py`, `scripts/run_demo_s1sweep.py`,
`slurm/s1sweep/*` + `submit_s1sweep_dag.sh`. The encoder / train / IG / analysis stages and
`_encsweep_config.py` are **reused unchanged in substance** — three edits make them serve
both sweeps: `--arm` drops its `choices=` list, `arm_dir` validates the name as a safe
directory rather than against a fixed set, and the analyzer reads its arm list and its
header from the meta the prepare stage wrote. `sign.slurm` also gained the chunk-merge that
was missing in §38.3 (it ran before `analysis` had written `signatures.parquet`, which is
why that stage failed there).
