# Cano-Gamez CD4 T-cell — coupling-gate validation (4th dataset, human)

**Goal.** A 4th, independent, **human** validation of this session's coupling-gate fix
(§28.2 degree/hub correction). Cano-Gamez et al. 2020 (Nat Commun) CD4+ T cells — a PBMC
subset, NOT whole PBMC — was chosen as the cascade-cleanest freely-downloadable human
option. The scRNA subset turned out to contain only **5 conditions** (no Th1/IFN-β; those
were bulk-only), so this is a **coupling + degree-correction** validation, not a
cascade-direction one.

**Data.** BioStudies S-BSST2978 (processed UMI, free; raw is EGA-gated). 43,112 cells ×
20,953 genes → 4000 HVG. Conditions: `PBS`(=resting/UNS), `Th0` (TCR/CD28 only), `Th2`
(IL-4→STAT6), `Th17` (IL-1β/IL-6/IL-23/TGF-β→STAT3), `iTreg` (TGF-β→SMAD). 4 donors
(D1–D4, all conditions), 2 cell types (Naive/Memory). 200 pseudotubes.

**Method (reuse).** Pseudotubes → 4 binary AB-MIL (condition vs PBS, shared Stage-1
encoder, WIDE config) → Integrated-Gradients signatures `S_X` → **cell-level
degree-corrected coupling** (`cell_coupling_degree`; 4 donors → donor-level null
inapplicable, p-floor 1/2⁴=0.0625, so the Sheu/ID cell-level path). Both signature
variants (IG_vsPBS, IG_vsPanel) × {raw, hub}; gene-set null n_perm=500.
**Benchmark** (`cano_axes_labeled.csv`): Th17↔iTreg = coupled (shared TGF-β; the data's
own clusters merge them, e.g. "TCM2/TEM (Th17/iTreg)"); Th2 = distinct lineage; Th0 =
activation hub (degree correction must suppress).

**Run.** Jobs 30885789 (build), 30885791 (binary, GPU 6 min), 30885792 (bridge/IG+coupling).
Outputs `results/cano_gamez/coupling_cell_degree/`.

---

## Headline

| variant | mode | coupled / 6 | known-pair (Th17-iTreg) | over-call |
|---|---|---|---|---|
| IG_vsPBS | raw | 6 (100%) | coupled | 100% — everything (shared activation) |
| **IG_vsPBS** | **hub** | **2 (33%)** | **#1, passes null** | cut 100%→33% |
| IG_vsPanel | raw | 6 (100%) | coupled | 100% |
| IG_vsPanel | hub | 3 (50%) | #1, passes null | cut 100%→50% |

**The degree-correction fix generalizes to a 4th dataset and a new cell type (human CD4
T).** Raw coupling calls every pair coupled (all conditions share the T-cell activation
program — the exact over-call failure §28.2 targets). Hub correction isolates the specific
coupling.

## IG_vsPBS — all 6 pairs by coupling_hub (the readout)

| rank | pair | label | coupling_hub | null_p_hub | call |
|---:|---|---|---:|---:|:--|
| **1** | Th17–iTreg | KNOWN (TGF-β) | **+0.084** | 0.00 | ✓ coupled — correct |
| 2 | Th0–Th2 | activation hub | +0.052 | 0.00 | ✗ residual hub FP |
| 3 | Th0–Th17 | activation hub | −0.005 | 1.00 | ✓ suppressed |
| 4 | Th0–iTreg | activation hub | −0.018 | 1.00 | ✓ suppressed |
| 5 | Th17–Th2 | distinct | −0.049 | 1.00 | ✓ suppressed |
| 6 | Th2–iTreg | distinct | −0.065 | 1.00 | ✓ suppressed |

The one **known coupled pair (Th17↔iTreg) is ranked #1 and clears the null**; 2/3 of the
Th0 activation-hub pairs and both distinct-lineage pairs are suppressed. One residual hub
false-positive (Th0–Th2).

## vsPBS beats vsPanel here too

IG_vsPBS+hub (over-call 33%, 2 coupled) is **cleaner** than IG_vsPanel+hub (50%, 3
coupled) — both rank Th17↔iTreg #1, but vsPanel lets more Th0-hub pairs through. This
matches the **Immune Dictionary** result (vsPBS > vsPanel for coupling) and adds a 3rd
dataset to the picture: **Oesinghaus is the exception that prefers vsPanel; ID and
Cano-Gamez both prefer vsPBS.** Consistent with the donor-count verdict
(`COUPLING_DONOR_COUNT_OES.md`) that the variant preference is **dataset-specific**, not
donor-count-driven.

## Direction (descriptive only — no clean cascades here)

The conditions are parallel differentiation fates, not cytokine→cytokine cascades, so
cross_asym is not a direction test here. Sensibly, the co-equal TGF-β partners Th17↔iTreg
have near-zero cross_asym (+0.006 vsPBS / −0.024 vsPanel) — no spurious "upstream" call.
The Th0 pairs have large |cross_asym| (e.g. Th0–iTreg −0.21), reflecting the
effectorness/activation gradient (more-polarized cells carry more program), a known
confound — NOT a cascade.

## Caveats

1. **Tiny benchmark** — 1 known coupled pair, 6 total pairs, only 4 stimulated nodes. The
   degree-centering (node-strength estimate from 3 off-diagonal entries/node) is
   underpowered at n=4; that it still ranked Th17↔iTreg #1 and suppressed 4/5 non-targets
   is the encouraging signal, but this is a focused check, not a powered sweep.
2. **CD4 T cells, not whole PBMC.** A PBMC subset; the cascade-rich conditions (Th1, IFN-β)
   are bulk-only, absent from this scRNA matrix.
3. **Cell-level null is permissive** → ranking is the readout, not the binary call.
4. One residual hub false-positive (Th0–Th2) — degree correction is partial at n=4.

## Bottom line

On a 4th, independent, human dataset and a new cell type (CD4 T), the degree-corrected
coupling gate **recovers the textbook Th17↔iTreg (TGF-β) coupling as the #1 pair while
cutting the activation-driven over-call from 100% to 33% and suppressing most of the
activation hub + the distinct lineages.** The §28.2 fix now holds across
Oesinghaus / Sheu / Immune Dictionary / Cano-Gamez (human PBMC, mouse BMDM, mouse in-vivo
lymph node, human CD4 T). vsPBS again beats vsPanel for coupling, reinforcing that Oes's
vsPanel preference is dataset-specific. Direction is not testable on this dataset's
parallel-fate conditions.
