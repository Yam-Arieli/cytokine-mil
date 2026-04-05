# Task: Implement the Oelen 2022 (1M-scBloodNL) Pathogen Experiment

## Context
This codebase already implements an AB-MIL experiment on the Oesinghaus cytokine 
dataset (90 cytokines, 12 donors, PBMCs). The goal now is to run the same 
experiment on a new dataset — Oelen et al. 2022 (1M-scBloodNL) — as a validation 
that the method generalizes beyond cytokine-specific biological assumptions.

The new experiment should:
1. Build pseudo-tubes from the Oelen dataset
2. Train the same SA+CA attention architecture as in `train_stage3_ca.py`
3. Visualize results the same way as in `inspect_stage2-sa_stage3-ca_results.ipynb`

Do NOT modify any existing files. Create new files only.

---

## The New Dataset: Oelen et al. 2022 (1M-scBloodNL)

### File locations
- `/cs/labs/mornitzan/yam.arieli/datasets/Oelen/1m_scbloodnl_v2.h5ad`
- `/cs/labs/mornitzan/yam.arieli/datasets/Oelen/1m_scbloodnl_v3.h5ad`

### USE v3 ONLY. Do not use v2.
v2 and v3 differ in 10x Genomics chemistry version and have different gene 
detection rates (~900 genes/cell for v2 vs ~1860 genes/cell for v3). Mixing them 
would introduce a strong technical confound. v3 is higher quality.

### adata.obs columns (both files have the same structure after metadata join):
- `assignment` — donor ID (integer as string, e.g. "1", "2", ...). This is the 
  analog of `donor` in the Oesinghaus dataset. Each donor's cells were processed 
  together as a biological system — this is the batch unit for pseudo-tube 
  construction.
- `timepoint` — perturbation condition. Values are: "UT" (unstimulated), "3hCA", 
  "24hCA", "3hMTB", "24hMTB", "3hPA", "24hPA". This is the analog of `cytokine` 
  in the Oesinghaus dataset.
- `cell_type` — fine-grained cell type annotation (e.g. "memory CD8T", "mono 2").
- `cell_type_lowerres` — coarse cell type annotation (e.g. "CD8T", "monocyte"). 
  USE THIS for pseudo-tube stratification, analogous to `cell_type` in Oesinghaus.
- `chemistry` — "v3" (all cells in the v3 file).

### Perturbation classes to use
Filter to 24h timepoints + UT only:
- "UT" (unstimulated control — analog of PBS)
- "24hCA" (C. albicans — intermediate cascade)
- "24hMTB" (M. tuberculosis — deep cascade, expected hardest to learn)
- "24hPA" (P. aeruginosa — most direct, expected easiest to learn)

This gives 4 classes. The biological ground truth ordering for validation is:
  UT ≈ 24hPA (easiest) → 24hCA (intermediate) → 24hMTB (hardest)
MTB is the most cascade-dependent: monocytes phagocytose it → secrete IL-12 → 
NK/T cells produce IFN-γ → macrophage reprogramming. Its PBMC signature at 24h 
is dominated by indirect, multicellular effects.
PA activates TLR4/TLR5 directly and simultaneously across multiple cell types — 
no cascade required.

### Raw data — preprocessing required
The h5ad files contain RAW unnormalized gene counts (integer counts, no 
normalization, no HVG selection applied). You must apply the standard preprocessing 
pipeline before building pseudo-tubes:

1. Filter cells: minimum 200 genes detected, maximum 5% mitochondrial content
2. Filter genes: minimum 10 cells expressing the gene
3. Normalize total counts per cell to 10,000
4. Log1p transform
5. Select HVGs: top 4000 highly variable genes
   (use sc.pp.highly_variable_genes, then subset adata to HVGs)
6. Do NOT z-score / scale
7. Do NOT compute log2FC vs control

### Number of donors
The v3 file contains cells from ~60-80 donors (v3 chemistry subset of the 120 
total). After filtering to 24h+UT only, check how many unique `assignment` values 
remain — expect ~40-70 donors with all 4 conditions present. Use only donors that 
have ALL 4 conditions (UT, 24hCA, 24hMTB, 24hPA) represented with sufficient cells.

---

## What to implement

### Step 1: Build pseudo-tubes
Create `scripts/build_pseudotubes_oelen.py`.

Pseudo-tube construction logic is IDENTICAL to the Oesinghaus version 
(`scripts/build_pseudotubes.py`), with these substitutions:
- `donor` column → `assignment`
- `cytokine` column → `timepoint`
- `cell_type` column → `cell_type_lowerres`
- Source file: v3 h5ad only
- Filter to 24h+UT conditions before building tubes
- Output directory: `/cs/labs/mornitzan/yam.arieli/datasets/Oelen_pseudotubes/`
- Manifest key `cytokine` should store the timepoint value (e.g. "24hMTB")
- Use the same parameters: N_PER_CELL_TYPE=30, MIN_CELLS_THRESHOLD=10, 
  N_PSEUDO_TUBES=10

The manifest.json structure must be identical to the Oesinghaus manifest so the 
existing PseudoTubeDataset and CytokineLabel classes can consume it without changes.

### Step 2: Training script
Create `scripts/train_oelen_stage2sa_stage3ca.py`.

Copy the logic of `train_stage3_ca.py` exactly — same SA attention stage, same CA 
attention stage, same training loop, same dynamics logging. The only changes are:
- Use the Oelen manifest path
- n_classes = 4 (UT, 24hCA, 24hMTB, 24hPA)
- Adjust label encoder for 4 classes (UT = class 3 or whichever index, not 90)
- Output/checkpoint directory: 
  `/cs/labs/mornitzan/yam.arieli/results/oelen_sa_ca/`
- Keep all other hyperparameters identical to the Oesinghaus experiment for 
  comparability (lr, optimizer, epochs, embedding_dim, etc.)
- No train/val donor split needed for this validation experiment — use all donors 
  for training (we are not measuring generalization, we are measuring learnability 
  ordering)

### Step 3: Visualization notebook
Create `notebooks/inspect_oelen_sa_ca_results.ipynb`.

Copy the structure of `inspect_stage2-sa_stage3-ca_results.ipynb` exactly. 
The only changes are:
- Load results from the Oelen output directory
- Class names are ["UT", "24hCA", "24hMTB", "24hPA"] instead of cytokine names
- The expected learnability ordering annotation should reflect the pathogen 
  cascade ground truth: PA easiest, MTB hardest
- All plots should be identical in structure to the Oesinghaus notebook

---

## Implementation plan
Before writing any code, read:
1. `train_stage3_ca.py` — understand the exact SA+CA training loop
2. `inspect_stage2-sa_stage3-ca_results.ipynb` — understand the exact 
   visualization structure and what result files it expects to load
3. `scripts/build_pseudotubes.py` — understand the exact pseudo-tube 
   construction logic to replicate

Then implement in order: Step 1 → Step 2 → Step 3.

Do not run any training — only write the scripts and notebook. The training will 
be submitted as a SLURM job separately.