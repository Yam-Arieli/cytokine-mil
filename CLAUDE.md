# CLAUDE.md — Implementation Guide
# Systemic Mapping of Cytokine Signaling Cascades via MIL Dynamics

---

## 0. Project Overview

AB-MIL model classifies which cytokine was applied to a pseudo-tube of PBMCs.
Training dynamics (per-cytokine learning curves, attention entropy, instance-level
confidence) are analyzed to infer the temporal hierarchy of cytokine signaling cascades.

**Central hypothesis:** cytokines learned early induce strong, direct, canonical
responses; cytokines learned late induce subtle, pleiotropic, or multicellular cascades.
This is a hypothesis to be tested — not a prior injected into the model.

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

---

## 3. Preprocessing Decisions

**Applied (in order):**
1. **Doublet removal** — Scrublet before building pseudo-tubes. *(Skipped in actual run — `pseudo_tubes_creation.ipynb` built raw tubes without it. Future runs use `scripts/build_pseudotubes.py`.)*
2. **Total count normalization** — normalize each cell to fixed total count.
3. **Log1p transformation** — variance-stabilizing.
4. **HVG selection** — 4000 HVGs (hyperparameter; standard 2000–2500 is tuned for clustering, not neural nets).

Steps 2–4 applied post-hoc via `notebooks/preprocess_tubes.ipynb`. HVGs estimated from one tube per cytokine (~40k cells); all tubes normalized + log1p + filtered in-place. Raw tubes preserved as `pseudotube_N_raw.h5ad`.

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

**Motivation:** the single-layer attention model treats all cells equally as potential signal sources. Cytokine signaling operates in cascades: a cytokine directly activates certain cell types (primary responders), which then secrete secondary cytokines that activate other cell types (secondary responders). The two-layer architecture encodes this biological structure directly:
- **Layer 1 (SA)** learns to identify primary responders from their strong direct transcriptional signal.
- **Layer 2 (CA)** re-attends over all cells conditioned on what Layer 1 found, learning to identify secondary responders whose signal is only interpretable given the primary response context.

This specialization emerges purely from the classification objective — no prior knowledge about which cytokines cascade into which is required. SA dynamics track when the model learns direct signals; CA dynamics track when it learns cascade signals.

**Architecture** — given H ∈ R^(N×128) from `InstanceEncoder`:

**Layer 1 — Self-Attention (SA):** standard AB-MIL attention, identical to `AttentionModule`:
```
a_SA ∈ R^N  (sum to 1)
z_SA = sum_i( a_SA_i * h_i ) ∈ R^128
```

**Layer 2 — Cross-Attention (CA):** re-attends over H conditioned on z_SA:
```
Query: z_SA;  Keys/Values: H
a_CA_i = softmax( w^T * tanh(V * h_i + U * z_SA) )
z_CA = sum_i( a_CA_i * h_i ) ∈ R^128
```

**Classifier input:** concatenate z_SA and z_CA → linear classifier. *(Try both sum and concat; keep the one giving cleaner loss curves.)*

**Implementation files** (coexist with existing v1 files — nothing is replaced):
- `models/two_layer_attention.py` — `TwoLayerAttentionModule` (SA + CA)
- `models/cytokine_abmil_v2.py` — `CytokineABMIL_V2`

**Forward pass contract** — `CytokineABMIL_V2.forward(X)` returns:
```
y_hat ∈ R^K,  a_SA ∈ R^N,  a_CA ∈ R^N,  H ∈ R^(N×128)
```
Both attention vectors must be returned for dynamics tracking.

`CytokineABMIL_V2.forward_with_aux(X)` returns a 6-tuple for regularized training:
```
y_hat ∈ R^K,  a_SA ∈ R^N,  a_CA ∈ R^N,  H ∈ R^(N×128),  y_hat_sa ∈ R^K,  y_hat_ca ∈ R^K
```
`y_hat_sa` and `y_hat_ca` are produced by auxiliary classification heads (`sa_head`, `ca_head`: `nn.Linear(embed_dim, n_classes)`) that classify from the SA and CA aggregates alone. Used during training only; `forward()` and `_log_dynamics` are unchanged.

**KL Regularization (v2 only):**

*Motivation:* Without regularization, CA may learn to re-express the same signal as SA differently — not because the signal required contextual reasoning, but because the optimization landscape allows it. The KL regularization creates an asymmetry: CA deviating from SA has a cost, so it will only do so when the classification loss demands it. This makes the SA/CA divergence pattern interpretable: when CA diverges from SA, it is because the signal required contextual reasoning (cascade), not by chance. This strengthens the scientific claim — the specialization is not just emergent, it is robust under pressure to collapse.

**Full loss formula:**
```
L_total = L_class(z_combined) + α · L_class(z_SA) + α · L_class(z_CA) + λ · KL(a_CA || a_SA)
```
- `L_class(z_combined)` — cross-entropy on the concatenated SA+CA representation (main signal)
- `L_class(z_SA)` — auxiliary cross-entropy from SA aggregate alone (forces SA to be independently useful)
- `L_class(z_CA)` — auxiliary cross-entropy from CA aggregate alone (forces CA to add genuine signal)
- `KL(a_CA || a_SA)` — penalizes CA for deviating from SA; only when classification loss demands it
- `α` = `training.aux_loss_weight` (default 0.5), `λ` = `training.kl_lambda` (default 0.1)

Implementation: `F.kl_div((a_SA + 1e-8).log(), a_CA, reduction='batchmean')` = `KL(a_CA || a_SA)`.

**Loss component logging:** `train_mil` returns `loss_components` in the dynamics dict:
- `total`: total loss per epoch (all models)
- `main`, `sa_aux`, `ca_aux`, `kl`: per-component losses per epoch (v2 only, empty lists for v1)

**Constraints:**
- The two layers **must NOT share weights** — they must specialize independently.
- Zero dropout in both attention layers (same reasoning as current architecture).
- KL regularization is **only applied during v2 training** (`kl_lambda > 0`). The v1 training loop is completely unchanged.
- Controlled by config flag `model.use_two_layer_attention: false` in `configs/default.yaml`.

**Dynamics extension:** `compute_instance_confidence` in `analysis/dynamics.py` accepts an optional second attention weight vector. When provided:
```
C_SA_i(t) = a_SA_i(t) * P(t)(Y_correct)
C_CA_i(t) = a_CA_i(t) * P(t)(Y_correct)
```
Both trajectories stored per cell per logged epoch. See Section 8.3.

**Notebook:** `notebooks/experiment_v2_two_layer_attention.ipynb` mirrors `experiment_bootstrap.ipynb` but uses `CytokineABMIL_V2`. Additional analysis section: SA vs CA confidence trajectories side by side per cytokine; overlap/divergence between a_SA and a_CA weight distributions across cell types.

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

```
cytokine_mil/               <- repo root
│
├── CLAUDE.md
├── pyproject.toml
├── README.md
│
├── cytokine_mil/           <- importable package
│   ├── __init__.py
│   ├── experiment_setup.py
│   ├── data/
│   │   ├── dataset.py          <- PseudoTubeDataset, collate_fn
│   │   └── label_encoder.py    <- CytokineLabel, BinaryLabel
│   ├── models/
│   │   ├── instance_encoder.py
│   │   ├── attention.py
│   │   ├── bag_classifier.py
│   │   ├── cytokine_abmil.py
│   │   ├── two_layer_attention.py  <- TwoLayerAttentionModule (SA + CA)
│   │   └── cytokine_abmil_v2.py   <- CytokineABMIL_V2
│   ├── training/
│   │   ├── trainer.py          <- shared helpers, mega-batch logic
│   │   ├── train_encoder.py    <- Stage 1
│   │   └── train_mil.py        <- Stage 2/3
│   └── analysis/
│       ├── dynamics.py
│       ├── validation.py
│       ├── confusion_dynamics.py   <- confusion trajectory, asymmetry, cascade graph
│       └── latent_geometry.py      <- cytokine centroid geometry, directional bias, asymmetry (Section 20)
│
├── scripts/
│   ├── build_pseudotubes.py
│   ├── synthetic_cascade_control.py   <- Experiment 0 go/no-go gate (Section 19.5)
│   └── train_oesinghaus_full.py       <- full 91-class confusion dynamics training
├── configs/
│   └── default.yaml
├── notebooks/
│   ├── experiment.ipynb
│   ├── experiment_subset.ipynb     <- 10-cytokine subset (fixed EASY/HARD groups)
│   ├── experiment_binary.ipynb              <- binary experiment (one model per cytokine vs PBS, shared encoder)
│   ├── experiment_bootstrap.ipynb           <- bootstrap experiment (random 5+5 from SIMPLE/COMPLEX pools, see Section 18)
│   ├── experiment_v2_two_layer_attention.ipynb  <- v2 architecture experiment (CytokineABMIL_V2, SA vs CA analysis)
│   └── preprocess_tubes.ipynb
└── tests/
    ├── make_demo_data.py
    └── test_demo.py
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
pytest tests/test_demo.py -v
```

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
- **Hold out Donor2 and Donor3.** Never use val donors during training or optimizer steps.

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

**Purpose:** one binary AB-MIL per cytokine (cytokine vs PBS, n_classes=2) with a single shared frozen encoder. Isolates classification signal per cytokine without multi-class softmax competition.

**Cytokine subset** (same 10 as `experiment_subset.ipynb`):
```
IL-4, M-CSF, IL-10, TNF-alpha, IL-2, IL-22, VEGF, IL-12, OSM, HGF
```

### Training Protocol

**Stage 1 — Shared encoder** (identical to subset experiment): single `InstanceEncoder` trained via cell-type classification on all 10 cytokines + PBS.

**Stage 2 — Per-cytokine binary models:**
```python
bin_manifest, bin_label = make_binary_manifest(manifest_subset, target_cytokine)
# bin_label: positive→0 (cytokine), negative→1 (PBS), n_classes()==2
train_bin, val_bin = split_manifest_by_donor(bin_manifest, val_donors=VAL_DONORS)
model = build_mil_model(encoder, n_classes=2, encoder_frozen=True)
dynamics = train_mil(model, train_dataset, n_epochs=..., val_dataset=val_dataset)
```
Each model trained independently (no shared attention/classifier). `VAL_DONORS = ["Donor2", "Donor3"]`.

### Metrics

Both aggregated to donor level (median per donor, mean across donors).

**1. Normalized Trajectory AUC:**
```
normalized_AUC(C) = AUC( p_correct(t) / max_t(p_correct(t)) )
```
Divide trajectory by its own maximum; trapezoid-rule AUC normalized by `n_logged_epochs - 1`. Captures *shape* of learning independent of absolute confidence.

**2. Final Probability:**
```
final_p(C) = p_correct(t_final)
```
- `> 0.5` — better than chance; `≈ 1.0` — confident; `≤ 0.5` — no reliable signal.

Both metrics computed on train and val donors separately.

### Results Table

| Cytokine | Norm AUC (train) | Norm AUC (val) | Final P (train) | Final P (val) | Group |
|----------|-----------------|----------------|-----------------|---------------|-------|

**Group thresholds:**
- **EASY** — `final_p (train) > 0.75` AND `normalized_AUC (train) > 0.75`
- **HARD** — `final_p (train) <= 0.5` OR `normalized_AUC (train) <= 0.5`
- **MED** — everything else

Sort by `normalized_AUC (train)` descending. Column headers must use full metric descriptions, e.g. `AUC(norm_p_correct_trajectory, train donors)`.

**Precise output labels:**
```
Binary learnability — Normalized Trajectory AUC
Metric: AUC(p_correct(t) / max(p_correct(t))), aggregated to donor level
        (median across pseudo-tubes per donor, then mean across donors)

Binary learnability — Final Probability
Metric: p_correct(t_final), aggregated to donor level
        (median across pseudo-tubes per donor, then mean across donors)
```

**Relationship to multi-class:** binary framing removes cross-cytokine softmax competition. If a cytokine ranks high in multi-class but low in binary `final_p`, the model discriminated it *relative to other cytokines* but not vs resting state. Consistent rankings across framings indicate robust signal.

---

## 18. Bootstrap Experiment Cytokine Groups (`notebooks/experiment_bootstrap.ipynb`)

**Purpose:** test whether simple cytokines are learned faster than complex cytokines using a bootstrapped 5+5 subset design controlled by a single `BOOTSTRAP_SEED`.

### Pre-registered Hypothesis
> **Simple cytokines should rank higher in learnability AUC than complex cytokines.**
> Tested post-training with a one-sided Mann-Whitney U test (never repeated to avoid p-hacking).

### Top-level Controls
```python
BOOTSTRAP_SEED     = 42
N_SAMPLE_PER_GROUP = 5
```

### Cytokine Pools

**Name corrections vs user-facing notation:**
- IL-1β → `IL-1-beta`, IFN-β → `IFN-beta`, IL-32β → `IL-32-beta`, TGF-β → `TGF-beta1`

**Cytokine pools:** See `cytokine-pools` skill for full pool definitions, mechanisms, and rationale.

**Simple Pool** (9 candidates): `IL-4`, `IL-10`, `IL-2`, `M-CSF`, `TNF-alpha`, `IL-1-beta`, `IFN-beta`, `IL-7`, `G-CSF`

**Complex Pool** (8 candidates): `IL-12`, `IL-32-beta`, `OSM`, `IL-22`, `VEGF`, `HGF`, `TGF-beta1`, `IL-6`

### Sampling Logic
```python
_rng = random.Random(BOOTSTRAP_SEED)
SIMPLE_CYTOKINES  = sorted(_rng.sample(SIMPLE_POOL,  N_SAMPLE_PER_GROUP))
COMPLEX_CYTOKINES = sorted(_rng.sample(COMPLEX_POOL, N_SAMPLE_PER_GROUP))
SUBSET_CYTOKINES  = SIMPLE_CYTOKINES + COMPLEX_CYTOKINES
```

### Training Protocol
Identical to `experiment_subset.ipynb`: Stage 1 encoder on 10 sampled cytokines + PBS; Stage 2 AB-MIL frozen encoder; Stage 3 optional fine-tuning. `VAL_DONORS = ["Donor2", "Donor3"]`. Saved artifacts include seed in filename (e.g., `encoder_stage1_bootstrap_42.pt`).

### Hypothesis Test
One-sided Mann-Whitney U: simple AUC > complex AUC (5 vs 5, donor-aggregated). Report rank-biserial correlation alongside p-value (low power with n=5).

### Group Labels & Colors
`SIMPLE` (steelblue) and `COMPLEX` (tomato). Not `EASY`/`HARD`.

### Validation Checks
- **Generalization:** Spearman rho between train and val learnability rankings
- **Stage stability:** Spearman rho between Stage 2 and Stage 3 rankings
- **Seed robustness:** compare group-level Mann-Whitney p-values across different `BOOTSTRAP_SEED` values

---

## 19. Cascade Inference via Confusion Dynamics (`analysis/confusion_dynamics.py`)

### 19.1 Hypothesis

Each cytokine's pseudo-tube distribution is a mixture:
- Strong primary signal: direct transcriptional targets of that cytokine.
- Weaker secondary signal: cytokines produced downstream in the cascade (because activated cells begin responding to downstream cytokines).

If cascade A→B exists, A-tubes contain a weak B-like signal. A trained AB-MIL classifier will assign some softmax mass to class B when processing A-tubes, but not vice versa. This **asymmetric confusion** is the cascade direction signal.

**Temporal profile** distinguishes:
- Early-onset confusion → shared pathway (same TF, e.g. STAT3) — expected for IL-6/IL-10 (negative control)
- Late-onset confusion → cascade signal (after direct signatures learned, residual B-signal becomes the next thing to fit)

**Cell-type localization** of attention during confusion identifies the relay cell type (X in A→X→B).

### 19.2 Confusion Trajectory Tensor

At every logged epoch t, for each cytokine pair (A, B):
```
C(A, B, t) = mean over A-tubes of softmax_trajectory[B, t]
```
Yields a (K, K, T) tensor where K=91 (90 cytokines + PBS), T=n_logged_epochs.

Stored in `records` as `softmax_trajectory`: shape (K, n_logged_epochs) per tube (added to `_build_records` in `training/train_mil.py`).

### 19.3 Cascade Signal Extraction

**Signal 1: Asymmetry Score (directionality)**
```
Asym(A→B) = mean_{t ∈ late 30%}[ C(A,B,t) ] − mean_{t ∈ late 30%}[ C(B,A,t) ]
```
Positive asymmetry = evidence for A→B direction. Computed in `compute_asymmetry_score`.

**Signal 2: Temporal Profile (direct vs cascade)**
For each pair (A,B), characterize C(A,B,t):
- onset_epoch: first t where C > 0.05 * max(C)
- peak_epoch: argmax C
- profile_type: 'early' if peak < 30% of T, 'late' if peak > 70% of T, 'mid' otherwise

**Signal 3: Cell-type localization**
For pairs flagged as cascade candidates: extract per-cell attention weights from A-tubes, weighted by softmax[B], grouped by cell type. High attention in cell type X → X is the relay.

### 19.4 `analysis/confusion_dynamics.py` Functions

```python
def compute_confusion_trajectory(records, label_encoder) -> np.ndarray
# Groups records by true cytokine label.
# C[A, B, t] = mean over A-tubes of softmax_trajectory[B, t].
# Returns shape (K, K, T); row=true class, col=predicted class.

def aggregate_confusion_to_donor_level(records, label_encoder) -> np.ndarray
# For each (cytokine, donor): median across tubes → per-donor (K,T).
# Returns array (n_donors, K, K, T) for statistical analysis.

def compute_asymmetry_score(confusion_trajectory, late_epoch_fraction=0.3) -> np.ndarray
# Asym[A,B] = mean(C[A,B,t_late]) - mean(C[B,A,t_late]).
# Returns (K, K) matrix; positive = evidence for A→B.

def compute_temporal_profile(confusion_trajectory, a_idx, b_idx) -> dict
# Analyzes C[a_idx, b_idx, :] across epochs.
# Returns dict with onset_epoch, peak_epoch, profile_type, trajectory.

def extract_cell_type_attention_for_confusion(
    model, dataset, label_encoder, true_cyt, confused_cyt, device) -> dict
# For tubes labeled true_cyt, weight per-cell attention by softmax[confused_cyt_idx].
# Groups by cell type from tube .obs["cell_type"].
# Returns dict[cell_type -> mean_weighted_attention].

def build_cascade_graph(asymmetry_matrix, label_encoder,
    fdr_alpha=0.05, min_asymmetry=0.0) -> nx.DiGraph
# FDR correction (Benjamini-Hochberg) across all K*(K-1) pairs.
# Edge A→B if Asym[A,B] > 0 and FDR-significant.
# Returns nx.DiGraph with edge weights = asymmetry score.
```

### 19.5 Validation Experiments

All run before biological analysis. Sequential go/no-go gates.

**Experiment 0: Synthetic Positive Control (GO/NO-GO gate)**
- Construct synthetic A-tubes: replace alpha fraction (10–30%) of cells with B-tube cells.
- Expected: synthetic A-tubes show elevated C(A,B,t) with late onset.
- Script: `scripts/synthetic_cascade_control.py`. SLURM: `scripts/run_synthetic_cascade_control.slurm`.
- If fails: method cannot detect cascade signals → reconsider approach.

**Experiment 1: IL-12 → IFN-γ Biological Positive Control**
- Pre-registered: C(IL-12, IFN-γ, t) > C(IFN-γ, IL-12, t) at late epochs; late-onset; localized to NK cells.
- Documents recovery of best-documented PBMC cytokine cascade (Oesinghaus Fig. 4i).

**Experiment 2: IL-6 / IL-10 Negative Control**
- Shared STAT3 pathway but no documented cascade. Expect: symmetric confusion, early-onset, diffuse attention.
- Validates temporal separation claim.

**Experiment 3: Seed Stability**
- Run seeds 42, 123, 7. Spearman rho of asymmetry scores across seeds.
- Threshold: rho > 0.7 for reportable pairs. Unstable pairs excluded.

**Experiment 4: Cytokine Family Clustering**
- Cluster cytokines by confusion fingerprint. IFN family and IL-2/IL-15 should cluster.
- Sanity check before claiming novel cascade discovery.

**Experiment 5: 24h Kinetics Confound Check**
- After unblinding: test if cascade-source cytokines correlate with fast-acting kinetics.
- If yes → 24h timing is confound. If no → cascade-specific signal.

### 19.6 Config Parameters

Added to `configs/default.yaml` under `dynamics` (see Section 13):
```yaml
confusion_late_epoch_fraction: 0.3   # fraction of final epochs for asymmetry score
confusion_fdr_alpha: 0.05            # FDR threshold for cascade graph edges
cascade_graph_min_seed_rho: 0.7      # min Spearman rho across seeds for reportable pairs
```

### 19.7 Training

Full 91-class training script: `scripts/train_oesinghaus_full.py`.
SLURM: `scripts/run_oesinghaus_full.slurm` (array job, 3 seeds: 42, 123, 7).
Memory: 128GB (full 91-class ~1200 tubes). Time: 24h.

`softmax_trajectory` (shape K × n_logged_epochs) is stored in every record — added to `_build_records` in `training/train_mil.py`.

### 19.8 Precise Output Labels

```
Confusion trajectory
Metric: C(A,B,t) = mean over pseudo-tubes of true cytokine A of softmax[B] at logged epoch t
        Shape: (K, K, n_logged_epochs)

Asymmetry score
Metric: mean(C(A,B,t_late)) - mean(C(B,A,t_late)), where t_late = last 30% of training
        Positive = evidence for cascade direction A→B

Temporal profile
Metric: peak_epoch = argmax C(A,B,t); profile_type: early (<30% T), late (>70% T), mid
```

---

## 20. Latent Space Cytokine Geometry (`analysis/latent_geometry.py`)

### 20.1 Core Idea

After MIL training the encoder maps each cell to `h_i ∈ R^128`. We hypothesize that cells in cytokine-A tubes that are responding to endogenously produced cytokine B (a cascade secondary signal) will have their embedding displaced toward B's centroid in the learned representation space.

Cascade relationships are detected as **per-cell-type directional bias** of a cell-type subpopulation's mean embedding toward another cytokine's centroid, within a given cytokine's tubes — without using attention weights.

**Run on a 20-cytokine subset first** (must include IL-12, IFN-γ, IL-18, TNF-α, IL-6 as known cascade pairs), not all 91.

### 20.2 Experiment 0 — Does Cytokine Geometry Exist at the Cell Level? (GO/NO-GO Gate)

Everything in Sections 20.3–20.5 depends on this gate passing.

**Cytokine centroid:**
```
μ_C = mean over all cells in all C-tubes of h_i    (training donors only: D1, D4–D12)
```

**Cytokine affinity vector for cell i:**
```
f_i = softmax( -d(h_i, μ_C) for all C )    d = L2 distance
```

**Cytokine alignment score for cytokine A:**
```
cytokine_alignment_score(A) = mean over cells in A-tubes of f_i[A]
```

**Null distribution:** permute cytokine tube labels, recompute alignment score. Permutation test (1000 permutations).

**Gate criterion:** `cytokine_alignment_score` significantly above null (p < 0.05, permutation test) for the majority of cytokines.

**Also compute:** Spearman correlation between `cytokine_alignment_score(C)` and learnability AUC(C). Expect positive correlation.

**If gate fails:** encoder embeds in cell-type space only; cytokine geometry exists only at bag level. Proceed to Experiment 3 (auxiliary decoder — not yet implemented; see TODO in `analysis/latent_geometry.py`).

### 20.3 Experiment 1 — Per-Cell-Type Directional Bias

**Directional bias of cell type T in cytokine-A tubes toward cytokine B:**
```
bias(A, B, T) = (μ_{A,T} - μ_A) · (μ_B - μ_A) / ||μ_B - μ_A||_2
```
Where:
- `μ_{A,T}` = mean embedding of cells of type T within A-tubes
- `μ_A` = mean embedding of all cells within A-tubes
- `(μ_B - μ_A) / ||μ_B - μ_A||_2` = unit vector from A's centroid to B's centroid

Positive = cell type T is displaced toward B within A-tubes; negative = displaced away.

**Null distribution:** permute cell-type labels within tubes 1000 times, recompute bias.
```
z(A, B, T) = (bias(A, B, T) - mean(bias_null)) / std(bias_null)
```

**Multiple testing:** Benjamini-Hochberg FDR across all (A, B, T) triples.

**Positive control:** For IL-12→IFN-γ:
```
bias(IL-12, IFN-γ, NK)       > 0    # NK cells are primary IFN-γ responders
bias(IL-12, IFN-γ, CD14_Mono) > 0   # secondary
bias(IFN-γ, IL-12, NK)        ≈ 0   # asymmetry check
```

**Output:** Directed graph: edge A→B exists if `max_T z(A, B, T) > FDR threshold`, weighted by max z-score.

### 20.4 Experiment 2 — Asymmetry Test (Cascade Direction)

**Per-cell-type asymmetry:**
```
asym(A, B, T) = bias(A, B, T) - bias(B, A, T)
```

**Pair-level asymmetry (aggregate over cell types):**
```
ASYM(A, B) = max_T asym(A, B, T)
```

Positive `ASYM(A, B)` = evidence for cascade direction A→B.

**Output:** (K, K) asymmetry matrix. Cluster it. IFN family and IL-2/IL-15 should cluster.

**Seed stability:** run across seeds 42, 123, 7. Spearman rho of ASYM vectors across seeds. Threshold: rho > 0.7 for reportable pairs.

**Precise output labels:**
```
Cytokine alignment score
Metric: mean over cells in A-tubes of softmax(-||h_i - μ_C||_2 for all C)[A]
        Aggregated to donor level (training donors D1, D4–D12 only for centroid computation)

Directional bias
Metric: scalar projection of (μ_{A,T} - μ_A) onto unit vector (μ_B - μ_A)/||μ_B - μ_A||

Asymmetry score
Metric: max_T [ bias(A,B,T) - bias(B,A,T) ]
        Positive = evidence for cascade A→B
```

### 20.5 Experiment 3 — Auxiliary Decoder (Contingency)

**Not implemented.** Triggered only if Experiment 0 gate fails (alignment ≈ null, encoder embeds cell-type space only).

**Concept:** small MLP trained post-hoc on frozen encoder output, supervised by bag-level softmax `p_bag(C | tube)` via KL divergence. Transfers bag-level cytokine geometry to cell level without requiring cell-level cytokine labels.

See `# TODO: Experiment 3` comment in `analysis/latent_geometry.py`.

### 20.6 Implementation Notes

- Compute centroids `μ_C` using **training donors only** (D1, D4–D12); validate directional bias on D2, D3
- Cell-type labels reintroduced post-hoc from original h5ad metadata — never used during training
- Aggregate all statistics to donor level before significance testing (effective N=12, not N=120)
- All functions return a `metric_description` key per Section 14 convention
- Implementation file: `analysis/latent_geometry.py`

### 20.7 File

Add `analysis/latent_geometry.py` to the project file structure (Section 11):
```
cytokine_mil/analysis/
    dynamics.py
    validation.py
    confusion_dynamics.py
    latent_geometry.py     <- Experiments 0–2; Experiment 3 is a TODO
```
