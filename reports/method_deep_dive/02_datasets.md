# M2 — The datasets

We need *labeled* data to test the method. We use **two deliberately different datasets**
so a win on both can't be a fluke of one system.

## Vocabulary (ML ↔ this project)
- **example / sample** = one **pseudo-tube** (a bag of cells; built in M3).
- **label** = which **stimulus** was applied to that tube.
- **classes** = the set of stimuli, plus **PBS** (the resting/unstimulated control).
- **features** = **genes** (each cell is a vector over genes).
- **donor** = an independent biological replicate (a person, or a mouse context×replicate).
  Donors are the unit of independence — train/val split is **by donor**, never by tube
  (tubes from one donor are correlated; splitting on tubes would leak).

## Dataset 1 — Oesinghaus (broad, human, single late snapshot)
| property | value |
|---|---|
| system | human **PBMC** (mixed blood immune cells) |
| stimuli | **91 cytokines** + PBS |
| donors | **12** (people) |
| genes | whole transcriptome → **4000 HVGs** (highly-variable-gene selection) |
| time | **single snapshot, 24 h** post-stimulation |
| role | scale + human relevance; Path A validated here; **cross_asym = 88%** |
| weakness | 24 h is *late* (cascade saturated); direction labels were noisy (keyword-parsed literature → required the hand-audit `cytokine_axes_audited.csv`) |

## Dataset 2 — Sheu (deep, mouse, clean, time-resolved)
| property | value |
|---|---|
| system | mouse **BMDM** (bone-marrow-derived macrophages); subclusters `mac_c0..c3` |
| stimuli | **7** (LPS, LPSlo, P3CSK, PIC, TNF, CpG, IFNb) + PBS |
| genes | **targeted 500-gene immune panel** (BD Rhapsody) — no HVG step (already curated) |
| time | **time-course**: 0, 1, 3, 5 h (+ more) |
| donors | only 2 biological reps → **pseudo-donors** = (context × replicate); ~4 at 3 h |
| role | crystal-clear textbook cascade labels (the 21 pairs, M1); pick the best single frame |
| cross_asym | **86%** at 5 h |

The local file `results/hvg_list.json` **is** this 500-gene panel (mouse symbols:
`Acod1, Zfp36, ...` — all immune-response genes).

## Why two datasets
Different **species** (human/mouse), **system** (mixed blood / pure macrophages),
**scale** (91 / 7), **gene coverage** (4000 broad / 500 targeted), **time** (24 h / course).
cross_asym working on **both** (88% / 86%) shows the result is a property of the **method**,
not of one dataset, species, or platform.

## The no-cross-time-leakage constraint (Sheu-specific, central)
Sheu's time-course is a temptation: "learn the signature at 1 h, test direction at 3 h."
That would **cheat** — it uses the time axis, and the entire claim is *direction from a
single frame*. **Hard rule:** each time point is analyzed **completely independently**; the
method never sees more than one frame. We compare across time **only** in the final
analysis ("which frame scored best"), never inside the method. The 0 h Unstim baseline is
the one shared reference, and only because the direction metric is **PBS-invariant** (proven
in M7) so it cannot leak.

## The manifest (how a dataset is indexed)
A built dataset = a folder of pseudo-tube `.h5ad` files + a `manifest.json` listing them.
One entry (real schema):
```json
{ "path": ".../Donor1/IL-2/pseudotube_0.h5ad", "donor": "Donor1",
  "cytokine": "IL-2", "n_cells": 480,
  "cell_types_included": ["CD4_T","NK","CD14_Mono", ...], "tube_idx": 0 }
```
`cytokine` is the **label**; `donor` drives the split; `cell_types_included` supports the
per-cell-type stratification (M3/M6/M7). A separate `hvg_list.json` fixes the gene/feature
order.

## The 500-gene panel: help AND hazard (checkpoint topic)
- **Help:** every gene was hand-picked to be immune-informative → high SNR, no wasted
  features, no HVG step, far less noise than 4000 transcriptome-wide genes.
- **Hazard:** because the 500 are *co-regulated immune-response* genes, they move
  **together** on almost any activation. So *different* stimuli produce *similar-looking*
  top-gene signatures → it is harder to find a stimulus-**specific** signature; signatures
  **overlap** (the §22 finding). This overlap is part of why polyIC's signature collapses
  onto the IFN program (M5/M8).
