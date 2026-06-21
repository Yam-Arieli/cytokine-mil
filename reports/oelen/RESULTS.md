# Oelen 1M-scBloodNL — donor-level gate @128 donors + 3h→24h progression (5th dataset)

**Goal.** Two things on the largest, highest-donor dataset (Oelen 2022, human PBMC,
~911k cells, **128 donors**): (1) the **first run of the donor-level coupling gate at the
scale it was designed for** — every prior dataset had ≤12 donors, forcing the cell-level
path; (2) the **3h→24h progression-direction** (cross_asym vs the known time order, §30).
Pre-flagged caveat: the labels are 3 pathogens (CA/PA/MTB), which all engage the shared
innate program, so the coupling benchmark is weak — this is a **scale demonstration**, not
a clean discrimination.

**Data/method.** molgenis (free; raw EGA-gated). 2 chemistries merged (V2+V3, 23,269
genes → 4000 HVG), donor = composite `chem_assignment`, condition = pathogen×time
(3h/24h CA/PA/MTB), control = UT→PBS, cell types = `cell_type_lowerres` (CD4T/CD8T/
monocyte/NK/B/DC/…). 2370 pseudotubes. Binary AB-MIL (6 conditions vs PBS, shared
Stage-1, WIDE) → IG `S_X` → **donor-level null + degree correction**
(`run_donor_coupling_null --dataset oelen`, 104 usable donors, min_donors=8) AND
**cell-level cross_asym** (`run_cell_degree_coupling`, for the progression sign).
Benchmark `oelen_axes_labeled.csv`: 3 same-pathogen pairs coupled, expected_sign −1
(3h upstream). Jobs 30886152 (build), 30886157 (binary, GPU 1h16m), 30886158 (coupling).

---

## 1. Donor-level gate @128 donors — WORKS AT SCALE (the win), weak benchmark (pre-flagged)

| mode | variant | tested pairs | donor-coupled q<0.10 | same-pathogen recall |
|---|---|---|---|---|
| raw | IG_vsPBS | 15 | 0.33 | **1/3** |
| hub | IG_vsPBS | 15 | 0.47 | 1/3 |
| raw | IG_vsPanel | 15 | 0.20 | 1/3 |
| hub | IG_vsPanel | 15 | 0.47 | 1/3 |

- **Headline: the donor-level sign-flip null finally ran fully powered — 104 donors used,
  88–103 per pair** (vs ≤12 ever before), q-values down to 0.0019. The machinery scales;
  the donor-level gate is no longer power-starved. This was the session's central open
  thread (`COUPLING_DONOR_COUNT_OES.md`, §28.2 boundary "needs ~8+ donors").
- **Biological benchmark is weak (as pre-registered).** Only the **MTB** same-pathogen
  pair (24hMTB–3hMTB, q=0.0019) is recovered; CA and PA same-pathogen pairs are NOT
  coupled (24hCA–3hCA q=1.0). The top coupled set is MTB-progression + **shared-innate
  cross-pathogen pairs** (24hCA–24hMTB, 3hCA–3hMTB) — which *should* couple (shared
  PRR→NF-κB), so "1/3 recall on same-pathogen" understates: pathogens-all-couple means
  the 3-pair benchmark can't cleanly score this dataset. Exactly the limitation flagged
  before the run.
- **Degree correction did NOT cut over-call here** (raw 0.33 → hub 0.47). With only
  **6 condition-nodes**, degree-centering (node strength from 5 off-diagonal entries) is
  too thin to behave — the same n-too-small failure as Cano-Gamez (4 nodes). Degree
  correction needs a broad pair space (Oes: 24+ nodes) to work; it is not a fix at ≤6
  conditions.
- *Note:* the report's "How to read" lines (cell 77%→53%→hub lower; IL-15 disappears) are
  Oesinghaus boilerplate baked into the script, NOT Oelen findings.

## 2. Progression direction (3h→24h) — 2/3 correct

cross_asym for the same-pathogen pairs (canonical axis_a=24h, axis_b=3h; expected −1 = 3h
upstream):

| pathogen | cross_asym (IG_vsPBS) | expected | call |
|---|---:|:-:|:--|
| CA  | **−0.022** | − (3h up) | ✓ correct |
| PA  | **−0.018** | − (3h up) | ✓ correct |
| MTB | +0.003 | − | ✗ ambiguous (≈0) |

- **2/3 recover the known time order** (CA, PA: 3h upstream). MTB is a near-zero
  non-call — the §30 magnitude-confound (24h = "more of" the 3h program, so the
  antisymmetry can wash out). Magnitudes are small and several non-progression pairs have
  larger |cross_asym| (e.g. 24hMTB–3hPA −0.19), consistent with §26.4 (magnitude ≠
  coupling; read direction only on coupled pairs).
- Interesting asymmetry: **coupling recovered MTB; direction recovered CA+PA** — the
  per-pathogen signatures behave differently (MTB's 3h/24h cross-engage → coupling;
  CA/PA's carry a directional asymmetry → direction).

## Honest bottom line

- **The donor-level gate runs at full power at 128 donors** — the scale milestone this
  session was missing. The donor-level null is validated as *operable and discriminating
  at scale* (some pairs q≈0.002, others q=1.0).
- **Oelen's pathogen labels do not give a clean coupling OR direction validation** — as
  pre-flagged. Coupling: 1/3 same-pathogen (+ real shared-innate). Progression: 2/3.
  Degree correction is inapplicable at 6 nodes.
- **Net:** a successful *scale demonstration* of the donor-level machinery + a partial
  (2/3) progression result, with the biological readout limited by the dataset's design,
  not the method. For a clean donor-level *discrimination* test we still need a
  high-donor dataset with a discriminative benchmark (coupled AND uncoupled pairs) — which
  remains the open need. The §28.2 degree-correction fix's clean validations stand on
  Oes/Sheu/ID/Cano-Gamez; Oelen adds the scale milestone, not a 5th clean pass.
