# Pre-Registration — §25 Cascade Sweep on the Immune Dictionary

**Project:** cytokine-MIL Path B extension
**Dataset:** Cui A. et al., *Nature* 625, 377–384 (2024). "Dictionary of immune responses to cytokines at single-cell resolution." Mouse lymph node, 86 cytokines, 17+ immune cell types, 4 h post sc/id injection, n=3 mice per cytokine, 386,703 cells, 10x Genomics 3′ v3.
**Data deposit:** GEO `GSE202186` (88 GSMs, 7.6 GB processed counts) + Broad SCP `SCP2554` (annotated metadata).
**Methodology:** §24 Directional Asymmetry Cascade Test (`cytokine_mil/analysis/pathway_audit.py::directional_asymmetry_test`). API and math unchanged from the Sheu sweep.
**Commit lock:** this file + the pathway gene sets in `cytokine_mil/analysis/pathway_signatures.py` MUST be committed to `main` BEFORE `scripts/run_immune_dictionary_pathway_audit.py` is run for the first time. Running before commit is a protocol violation.

---

## 1. Hypothesis Under Test

The §24 directional-asymmetry method, which on Sheu 2024 recovered the textbook TLR3-TRIF and TLR4-TRIF → IFN-β cascades, generalises to a broader set of cytokine cascades across JAK-STAT, NF-κB, and SMAD pathway families — but only when the two paired pathways are transcriptionally distinct. Overlapping pathways are predicted to fail by the same mechanism that caused the NF-κB → TNF failure on Sheu.

## 2. Pathway Gene Sets (locked at commit)

Defined in `cytokine_mil/analysis/pathway_signatures.py::PATHWAY_SIGNATURES`. Pre-existing (locked from Sheu): `IRF3_direct`, `IFNAR_induced`, `NFkB_canonical`, `TNFR_autocrine`. New additions:

| Pathway | Genes (mouse symbols) | Primary literature |
|---|---|---|
| `IL12_STAT4_target` | Il2ra, Hlx, Furin, Tnfrsf25, Eomes, Tbx21, Ifng, Cxcr6, Il18r1, Il18rap, Nkg7 | Wei 2010 *Immunity*; Mullen 2001 *Science*; Hwang 2005 *Science*; Lawless 2000 *J Immunol* |
| `IFNg_STAT1_target` | Gbp2, Gbp5, Iigp1, Ifi47, Igtp, Tgtp1, Nos2, Cxcl9, Cd274, Irf1, Socs1, Ciita | Boehm 1997 *Annu Rev Immunol*; MacMicking 2012 *Nat Rev Immunol*; Garcia-Diaz 2017 *Cell Rep* |
| `IL6_STAT3_target` | Socs3, Bcl3, Cish, Sgk1, Junb, Pim1, Saa1, Saa3, Mcl1, Tnfrsf1b, Stat3, Osmr | Hutchins 2013 *Blood*; Hagihara 2005 *Genes Cells*; Carpenter & Lo 2014 *Trends Immunol* |
| `IL2_STAT5_target` | Il2ra, Cish, Socs2, Bcl2, Foxp3, Ikzf2, Bcl2l1, Cd69, Stat5a, Stat5b, Eomes, Il2rb | Lin 2012 *Immunity*; Liao 2011 *Immunity*; Villarino 2016 *J Exp Med*; Burchill 2007 *J Immunol* |
| `IL4_STAT6_target` | Arg1, Retnla, Chi3l3, Mrc1, Ccl17, Ccl22, Gata3, Il4ra, Cd200r1, Pdcd1lg2, Klf4 | Czimmerer 2018 *Immunity*; Piccolo 2017 *Cell Rep*; Zheng & Flavell 1997 *Cell*; Liao 2011 *JCI* |
| `SMAD2_3_target` | Foxp3, Smad7, Skil, Serpine1, Tgfbr1, Ccn2, Itgb6, Cdkn1a, Cdkn2b, Id3, Tgfb1i1, Ltbp1 | Mullen 2011 *Cell*; Massagué 2012 *Nat Rev Mol Cell Biol*; Tone 2008 *Nat Immunol* |

**Pairwise overlap (|i ∩ j| / min(|i|, |j|)) verified via `compute_pathway_overlap_matrix()`:**

For all MUST-PASS (A, B, P_A, P_B) pairs: overlap(P_A, P_B) = 0.00 (well under the 0.30 distinctness threshold).
For all MUST-FAIL pairs: overlap(P_A, P_B) = 1.00 by construction (P_A = P_B).
Designed cross-pathway overlaps that do NOT compromise any pre-registered test:
- `IL12_STAT4_target` ∩ `IL2_STAT5_target` = 18% via {Il2ra, Eomes} — real Th1/effector overlap
- `IL6_STAT3_target` ∩ `IL2_STAT5_target` = 8% via {Cish} — general JAK-STAT feedback
- `SMAD2_3_target` ∩ `IL2_STAT5_target` = 8% via {Foxp3} — Treg differentiation requires both

## 3. Pre-Registered Cascade List

### 3.1 MUST-PASS (5 cascades, predicted directional_score > +1.0 in ≥1 cell type)

Distinct upstream and downstream pathways. Failure here would falsify the §24 methodology's generalization claim.

| # | A → B | P_A | P_B | Biological rationale | Expected relay cell type |
|---|---|---|---|---|---|
| 1 | IL-12 → IFN-γ | `IL12_STAT4_target` | `IFNg_STAT1_target` | IL-12 drives STAT4 → Tbx21 → Ifng in NK/T; IFN-γ engages STAT1-GAS ISGs | NK, CD8+ T |
| 2 | IL-1β → IL-6 | `NFkB_canonical` | `IL6_STAT3_target` | IL-1R → MyD88 → NF-κB → IL-6 induction; IL-6 → JAK1/2 → STAT3 — distinct downstream | macrophage, monocyte |
| 3 | IFN-γ → IL-12 | `IFNg_STAT1_target` | `IL12_STAT4_target` | IFN-γ → STAT1 → Irf1 → Il12b transactivation (positive feedback) | cDC1, macrophage |
| 4 | TNF → IL-6 | `NFkB_canonical` | `IL6_STAT3_target` | TNF → TNFR1 → NF-κB → IL-6 induction in myeloid cells | macrophage, monocyte |
| 5 | IFN-β → IFN-γ | `IFNAR_induced` | `IFNg_STAT1_target` | Type-I IFN primes NK for IFN-γ production (Nguyen 2002 *J Immunol*) | NK |

### 3.2 MUST-FAIL (3 cascades, predicted |directional_score| < 0.5)

Overlapping pathways (P_A = P_B). These are positive controls for the failure mode — replicating the §24 NF-κB → TNF failure observed on Sheu confirms the methodology's known limitation rather than refuting it.

| # | A → B | P_A | P_B | Predicted failure mode |
|---|---|---|---|---|
| 6 | IL-2 → IL-15 | `IL2_STAT5_target` | `IL2_STAT5_target` | Both common γ-chain → STAT5; gene sets identical by design |
| 7 | IL-1β → TNF | `NFkB_canonical` | `NFkB_canonical` | Both engage NF-κB targets; replicates Sheu's known failure |
| 8 | IL-4 → IL-13 | `IL4_STAT6_target` | `IL4_STAT6_target` | Both signal STAT6 via IL-4Rα; identical downstream targets |

### 3.3 NEGATIVE CONTROLS (2 cascades, predicted directional_score ≤ 0)

Distinct pathways but no biological cascade. Test discriminates "distinct pathways + cascade" (MUST-PASS) from "distinct pathways + no cascade" (NEG_CONTROL).

| # | A → B | P_A | P_B | Reason |
|---|---|---|---|---|
| 9 | IL-4 → IFN-γ | `IL4_STAT6_target` | `IFNg_STAT1_target` | Th2 actively inhibits Th1 — antagonism, not cascade |
| 10 | IL-10 → IL-12 | `NFkB_canonical` | `IL12_STAT4_target` | IL-10 → STAT3 → suppresses Il12b transcription; reverse of any induction |

## 4. Scoring Protocol (locked)

Per cascade (A, B, P_A, P_B):

1. **Per-cell pathway scores** — `s_X_on_PY = cells[:, gene_idx_PY].mean(axis=1)` for X ∈ {A, B, PBS} on Y ∈ {A, B}.
2. **PBS-normalised** — subtract `s_PBS_on_PY` from each tube's score on P_Y.
3. **Per-cell-type asymmetries** —
   `asym_PA(T) = (s_A_on_PA_norm − s_B_on_PA_norm)(T)`
   `asym_PB(T) = (s_A_on_PB_norm − s_B_on_PB_norm)(T)`
4. **Per-cell-type directional score** — `directional_score(T) = asym_PA(T) − asym_PB(T)`.
5. **Per-mouse aggregation** — for each mouse_id, mean(directional_score) across informative cell types where both A and B have ≥10 cells.
6. **Donor-level statistic** — sign of mean(directional_score) across the 2 train mice + 1 val mouse (n=3, severely underpowered; magnitude is the primary readout, not the p-value).
7. **Reported per-cascade output** — table of (cell_type, asym_PA, asym_PB, directional_score) across all mice + a per-mouse mean.

## 5. Verdict Criteria (locked)

Pre-registered thresholds. No post-hoc adjustment.

**GREEN (§24 generalises):**
- 4 of 5 MUST-PASS cascades show directional_score > +1.0 in ≥ 1 cell type, AND
- 2 of 3 MUST-FAIL show |directional_score| < 0.5 (overlap-failure replicated), AND
- 2 of 2 NEG_CONTROL show directional_score ≤ 0 (no false-positive cascade calls)

→ Path B claim upgrades to "cascade-direction inference across JAK-STAT, NF-κB, and IFN pathway families."

**AMBER (partial):**
- 2-3 of 5 MUST-PASS pass, OR
- 1 of 3 MUST-FAIL miscalled, OR
- 1 of 2 NEG_CONTROL miscalled

→ Identify which pathway pairs caused failure; check the gene-set overlap matrix; consider literature-based gene-set refinement before re-running. Do NOT silently adjust pathway lists post-hoc — any change to gene sets is a new pre-registration.

**RED (§24 does not generalise):**
- 0-1 of 5 MUST-PASS pass

→ Report as TLR-IFN-specific result only. The §24 methodology's claim is restricted to the original 2 Sheu cascades. Conclude that the in-vivo 4 h time point or species-specific differences invalidate the broader application.

## 6. Validity Boundaries (declared up front)

The following caveats apply to ANY outcome and must be reported alongside the verdict:

- **Single 4 h time point.** No kinetic validation as in Sheu's 1h→3h→5h trajectory. The §24 method is single-time-point by construction, but cascade products that take longer than 4 h to develop will be missed.
- **In vivo design.** The relay cell type sometimes differs from the responder cell type (paracrine networks operate at the organism level). Relay cell-type calls are reported as informative, not causal.
- **Mouse only.** The Oesinghaus/Cytokine Dictionary human cross-check is independent (Path A axis-discovery, already complete on `main`). No cross-species generalization is claimed from §25.
- **n = 3 mice.** Donor-level Wilcoxon signed-rank p-values are reported but not interpreted as evidence — magnitudes and direction-of-effect are the primary readout.
- **Stimulus name resolution.** Pre-registered cascade list uses canonical cytokine names (e.g., `"IL-12"`, `"IFN-g"`). The actual ID manifest stimulus strings will be resolved at run time. Any stimulus that does not match the ID manifest is skipped with an explicit log line; this counts as a NULL outcome for that cascade and is reported separately from MUST-PASS / MUST-FAIL / NEG_CONTROL outcomes.

## 7. Outputs (locked)

The run script writes:

- `results/immune_dictionary_pathway/cascade_results.parquet` — long-form (cascade, cell_type, mouse_id, asym_PA, asym_PB, directional_score, n_cells_A, n_cells_B, n_cells_PBS)
- `results/immune_dictionary_pathway/per_cascade_summary.csv` — one row per cascade with mean(directional_score), sign-test stats, predicted outcome, observed outcome
- `results/immune_dictionary_pathway/verdict.json` — GREEN/AMBER/RED + counts per category
- `results/immune_dictionary_pathway/plots/cascade_<A>_<B>.pdf` — per-cascade strip plot of (cell_type × mouse) directional_score values
- `results/immune_dictionary_pathway/pathway_overlap_matrix.csv` — 10x10 confirmatory matrix at run time (must match this document's claims)
- `reports/immune_dictionary/CASCADE_SWEEP_RESULTS.md` — written by hand after the audit, summarizing the verdict and per-cascade interpretations

## 8. Signatures

- **Pre-registered by:** Yam Arieli
- **Date locked:** 2026-06-01
- **Methodology basis:** CLAUDE.md §24 (Directional Asymmetry Cascade Test), §25 (Phase 2 Cascade Sweep on Immune Dictionary)
- **Git commit hash at lock:** TBD (record in commit message when locking)
- **No post-hoc modification of cascade list, pathway gene sets, or verdict criteria is permitted after the audit script runs.** Any subsequent change constitutes a new pre-registration with a new audit run.
