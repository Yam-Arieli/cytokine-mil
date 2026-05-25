# Session summary — 2026-05-26

Continuation of the post-2026-05-25 cascade-direction push. The previous session
ended with eight cumulative failed checks of the encoder + PBS-RC + dot-product
method bundle and a decision to "rethink basic assumptions". This session
inverted the workflow — first explored, then proposed, then audited — and
ended with a narrower, audited result.

This document is self-contained so the next session can resume without
scrolling back.

---

## TL;DR

- **Discovered:** a methodology (paired-pathway directional asymmetry on
  literature-curated gene sets) that produces the predicted asymmetric
  signature for two textbook cascades — PIC→IFN-β and LPS→IFN-β — on
  Sheu 2024 BMDM 3hr data.
- **Retracted (after own audit):** the initial "4 of 5 pre-registered tests
  pass Bonferroni α=0.01" headline. Three of the four claimed-significant
  tests were artifacts of small N, random-gene-set non-specificity, or
  pseudo-replication.
- **What stands:** a methodology concept demonstrated on two textbook
  cascades with consistent +1.7 to +2.4 directional_score, NOT an
  established general-purpose cascade-direction tool.
- **Status:** promising lead, not validated method.

---

## What was built (artifacts on `main`)

### Code (new files committed in this session)

- `cytokine_mil/analysis/eda_pair_benchmark.py` — pair-level statistic battery
- `cytokine_mil/analysis/eda_pair_plots.py` — EDA plot helpers
- `cytokine_mil/analysis/pathway_signatures.py` — curated pathway library
  (mouse + human), penetration math, pre-registered binary IFNAR test,
  magnitude test, battery aggregator with Bonferroni
- `cytokine_mil/analysis/pathway_plots.py` — pathway plotting helpers
- `cytokine_mil/analysis/pathway_audit.py` — adversarial audit framework
  (4 falsification tests)
- `scripts/run_sheu_eda_benchmark.py` — EDA driver
- `scripts/run_sheu_pathway_signatures.py` — Sheu pathway driver
- `scripts/run_pathway_trajectory.py` — time-trajectory driver
- `scripts/run_oesinghaus_pathway_signatures.py` — Oesinghaus pathway driver
- `scripts/run_pathway_audit.py` — audit driver (the script that revised the
  conclusions)
- `scripts/_inspect_sheu_timepoints.py`, `scripts/_inspect_oesinghaus.py` —
  one-off data inspectors

### SLURM scripts

- `slurm/run_sheu_eda.slurm` — EDA benchmark
- `slurm/run_sheu_pathway.slurm` — pathway analysis 3hr
- `slurm/build_pseudotubes_sheu_0_5hr.slurm`, `slurm/build_pseudotubes_sheu_5hr.slurm`,
  `slurm/build_pseudotubes_sheu_8hr.slurm` — additional time-point builds
- `slurm/run_pathway_trajectory.slurm` — 5-time-point trajectory
- `slurm/run_oesinghaus_pathway.slurm` — Oesinghaus pathway
- `slurm/run_pathway_audit.slurm` — adversarial audit

### Reports / specifications

- `reports/sheu2024_pathway/cascade_direction_results.md` — Path B writeup
  (drafted 2026-05-25, fully revised 2026-05-26 after audit)
- `reports/sheu2024_pathway/figures/` — headline plots + audit plots + audit
  summary CSVs (self-contained for the writeup)
- `CLAUDE.md` §0 (status block) — fully revised 2026-05-26
- `CLAUDE.md` §22 — pair-level EDA benchmark spec
- `CLAUDE.md` §23 — pathway-signature methodology (with audit-revision warning
  added pointing readers to §24)
- `CLAUDE.md` §24 (new) — directional asymmetry methodology spec, the
  post-audit primary methodology

### Cluster job IDs (for archaeology)

| Job ID | Description | Outcome |
|---|---|---|
| 30658924 | Sheu EDA benchmark (3hr) | Done; nothing cleared the permutation null |
| 30659060 | Sheu pathway analysis 3hr (initial) | Done; revealed cell-type-resolved IFNAR signal |
| 30659444 | Sheu pathway analysis with extended battery + Bonferroni | Done; produced the inflated "4/5 Bonferroni" claim |
| 30660346 | Sheu trajectory 1h + 3h | Done; 2-point kinetics |
| 30660738/9/40 | Sheu pseudotube builds: 0.5h, 5h, 8h | All COMPLETED |
| 30660842 | Oesinghaus pathway analysis (first attempt) | FAILED (missing defaultdict import) |
| 30660867 | Oesinghaus pathway analysis (rerun) | Done; AUC ≥ 0.74-0.93 across 18 cell types |
| 30660873 | Sheu trajectory 5 time points | Done; full kinetic |
| 30662225 | Adversarial audit (4 falsification tests) | **Done — invalidated 3 of 4 original claims, validated the 4th** |

---

## Chronological narrative

### Phase 1: brainstorm + EDA

Started from the user's pushback against closing the cascade-direction line ("5
failed checks of one specific method doesn't mean signal doesn't exist"). The
decision was to invert the workflow:

1. Stop designing methods from cascade-signal assumptions.
2. Build a labeled-pair benchmark from Sheu §21 (5 positives + 3 negatives).
3. Compute a broad statistic battery on raw normalized expression.
4. See where (if anywhere) the cascade signature actually lives in the data.

Built `eda_pair_benchmark.py` with 22 statistics per ordered (A, B, cell_type)
triple — centroid distance, Wasserstein, log2FC correlation, DE Jaccard,
reciprocal-closer fractions, signature scores, KL asymmetry, variance ratios,
bimodality. 2000-permutation null. Plots: heatmap, AUC bars, per-pair signature
scatter, projection density.

**Result:** Nothing cleared the permutation null (best p ≈ 0.064). Visual
inspection revealed the confounder: in Sheu's 500-gene targeted immune-response
panel, top-DE signatures of every stimulus collapse onto a diagonal because
every gene moves under any immune activation. Empirical signatures aren't
pathway-specific in this panel.

### Phase 2: curated pathway signatures (§23, the initial "positive result")

Driven by the EDA finding, switched to literature-curated, signaling-adaptor-
specific gene sets:

| Pathway | Marker genes | Primary stimuli |
|---|---|---|
| IRF3_direct | Ifnb1, Ccl5, Cxcl10, Ifit2, Ifit3 | PIC, LPS (TRIF arm) |
| IFNAR_induced | Isg15…→ Mx1, Mx2, Ifit1, Ifit1bl1, Ifit3, Ifit3b, Rsad2, Irf7, Oasl1 | IFNb |
| NFkB_canonical | Tnf, Il1b, Il6, Nfkbia, Nfkbid, Tnfaip3, Cxcl1/2, Ccl3/4, Birc3 | LPS, LPSlo, P3CSK, CpG, TNF |
| TNFR_autocrine | Tnfaip3, Nfkbid, Nfkbie, Nfkbiz | TNF |

Defined cascade penetration: `pen(A→P, B) = (s_A − s_PBS) / (s_B − s_PBS)`.
Defined a pre-registered battery: one binary IFNAR test + four NF-κB magnitude
tests, Bonferroni α=0.01.

**Result on Sheu 3hr:** "4 of 5 tests pass Bonferroni":
- Binary IFNAR in mac_c2: AUC=1.00, clean separation, p=0.05 (on edge)
- NF-κB magnitude LPS/LPSlo/P3CSK/CpG > TNF > PBS: all 4 with per-cell
  Mann-Whitney p ≈ 10⁻⁷⁹ to 10⁻¹³¹ in best cell type

Celebrated as first positive cascade-direction result. Drafted full writeup at
`reports/sheu2024_pathway/cascade_direction_results.md`. Updated CLAUDE.md §0
to claim "two positive contributions".

### Phase 3: trajectory + Oesinghaus generalisation

To corroborate the result:

- **Time trajectory** (1hr + 3hr, then extended to 0.5h, 1h, 3h, 5h, 8h): in
  mac_c3, PIC and LPS penetration of IFNAR flipped from negative at 1h to
  positive at 3h, consistent with the IFN-β autocrine kinetic timescale. In
  mac_c2 the signal was already established at 0.5h and remained stable through
  8h.

- **Oesinghaus 24h PBMC** (10M cells, subsampled to ~437k cells via 1 tube per
  train donor per cytokine): single-pathway binary AUC test on IFNAR_induced,
  positives = 6 auto-detected IFN cytokines (IFN-α1, IFN-β, IFN-γ, IFN-λ1/2/3),
  negatives = the other 84. AUC = 0.74-0.93 across 18 cell types. The
  methodology "generalised" from mouse BMDM 3hr to human PBMC 24hr.

At this point the project status block in CLAUDE.md was rewritten to claim two
publication-grade contributions: Path A (axis discovery) and Path B (cascade
direction via pathway penetration).

### Phase 4: user-requested adversarial audit

User explicitly pushed back: *"I want to be cautious before celebrating the
good results, because it seems to be too easy. ... make an honest audit as a
peer review."*

This was the right call. Drafted four falsification tests, each targeting a
specific potential weakness:

1. **Cytokine-label permutation null** on the binary IFNAR test. Tests: is
   the observed AUC distinguishable from random label assignment?
2. **Random-pathway null.** Generate 200 random gene sets of the same size as
   the curated IFNAR signature, drawn from non-curated panel genes. Compute
   binary AUC for each. Tests: does the curated gene list carry pathway-
   specific information, or do random subsets give similar AUCs?
3. **Per-donor Wilcoxon signed-rank** replacing the per-cell Mann-Whitney for
   the four NF-κB magnitude tests. Tests: do the p-values survive when
   pseudo-replication is respected?
4. **Directional asymmetry test** — the test the original analysis silently
   skipped. Compute `s(A) on P_A`, `s(A) on P_B`, `s(B) on P_A`, `s(B) on
   P_B`. True cascade A→B predicts asym_PA = (sA−sB on P_A) positive AND
   asym_PB = (sA−sB on P_B) negative (B is the direct ligand on P_B, A is
   partial via cascade). Tests: does the data show cascade DIRECTION (vs
   pure pathway engagement)?

### Audit results (job 30662225)

| # | Audit | Outcome | Key number |
|---|---|---|---|
| 1 | Permutation null on binary IFNAR | **FAILED** | mac_c2: observed AUC=1.00, p_emp = **0.096** (does not clear α=0.05) |
| 2 | Random-pathway null | **FAILED** | mac_c2: random 9-gene-set AUC distribution has Q95 = **1.0** — curated IFNAR no better than random |
| 3 | Per-donor inference | **FAILED to reach significance** | mac_c3 LPS/LPSlo/P3CSK/CpG vs TNF: Wilcoxon p = **0.0625** (design floor for n=4); mean s_NFkB(TNF) < mean s_NFkB(PBS) in mac_c3 (baseline assumption broken) |
| 4 | Directional asymmetry | **PASSED** | PIC→IFNb mac_c2: directional_score = **+1.87**; mac_c3: **+1.67**. LPS→IFNb: **+2.35**, **+2.02**. NF-κB cascades cluster on diagonal (paired pathways overlap — methodology limitation) |

The audit invalidated the original headline. Random gene sets give similar AUC
to the curated one, the binary IFNAR test does not clear permutation null, and
the per-cell p-values were inflated by pseudo-replication. The one rigorous
test (Audit 4 — directional asymmetry on paired pathways) survived and showed
genuine cascade signature for the two textbook TLR-TRIF → IFN-β cascades.

### Phase 5: revise writeup, retract inflated claims

Following the audit:

- `reports/sheu2024_pathway/cascade_direction_results.md` fully rewritten.
  Original headline retracted. New defensible claim narrow: directional
  asymmetric signature for PIC/LPS → IFN-β cascades detectable on 3hr BMDM
  data via paired curated pathway signatures.
- CLAUDE.md §0 status block rewritten with the retraction + revised diagnosis.
- CLAUDE.md §23 marked with an audit-revision warning pointing readers to §24.
- CLAUDE.md §24 (new section) specifies the directional asymmetry methodology
  with construction, preconditions, public API, power discussion, and
  explicit non-claims.

---

## Honest current state

Asked "did we find an established method for directionality?" — answer: **no,
not yet.**

What we have:
- A methodology concept (paired-pathway directional asymmetry) that produced
  the predicted asymmetric signature for two textbook cascades across two
  cell types — 4 observations total.
- The numerical values are large and biologically consistent (+1.7 to +2.4).
- The audit invalidated the easier, less rigorous tests (binary AUC, magnitude
  tests with inflated per-cell p-values).

What's missing for "established":
- 4 observations is not statistical power. Need many more cascades tested.
- Cascades were chosen knowing the answer; pre-registration was effectively
  post-hoc.
- Fails entirely on overlapping paired pathways (NF-κB → TNFR test does not
  discriminate).
- No blind/independent validation, no negative-control cascade with predicted
  ≈ 0 score, no donor-level inference on directional_score.

**Honest framing:** the work has moved from "general cascade-direction
inference is recoverable" (the previous session's claim) to "a methodology
concept that worked on two textbook cases under retrospective inspection". A
promising lead, not a working tool.

---

## Path A status

Unchanged. Oesinghaus axis discovery (`reports/cascade_pairs/cytokine_axes_report.md`)
remains publication-grade and independent of all Path B revisions.

## Path B status

Revised down to a methodology demonstration on two textbook cascades. The
Oesinghaus generalisation claim (AUC ≥ 0.85 in most cell types) was a
single-pathway AUC test that Audit 2 invalidates as not gene-list-specific;
that result is retracted unless extended to a directional asymmetry test with
paired pathways (which is not always possible on the 91-cytokine panel because
upstream P_A is not defined for direct cytokine ligands).

---

## Next defensible steps from the revised writeup

Numbered as in `reports/sheu2024_pathway/cascade_direction_results.md` §6.4:

1. **Donor-level directional_score inference.** Compute `directional_score` per
   pseudo-donor (instead of pooled across donors), then Wilcoxon signed-rank
   across the 3-4 donors. With small N this is low-powered but rigorous, and
   replaces the current 4-observations-with-no-formal-statistics situation
   with honest donor-level p-values.

2. **Pre-register the methodology blind.** A colleague defines a paired-pathway
   test for a cascade not previously analysed, before seeing the data. If the
   `directional_score` recovers the expected sign for a held-out cascade, the
   methodology has actual information content beyond what's available to
   someone who already knows the answer.

3. **Negative-control cascade.** Define a non-cascade pair (e.g., IL-4 → IL-10,
   no documented cascade) and check that `directional_score ≈ 0`. This is the
   most direct falsification: if random non-cascade pairs also score positive,
   the method is not measuring what we think.

4. **Time-resolved directional_score.** Re-run the 5-time-point trajectory
   (0.5h/1h/3h/5h/8h pseudotubes already built) computing
   `directional_score` instead of single-pathway penetration. Cascade should
   "turn on" over time — predict score increases monotonically from 0.5h to a
   plateau by 5-8h. Failure of this prediction would weaken the cascade
   interpretation.

5. **Apply to additional textbook cascade pairs** where P_A and P_B are
   well-separated transcriptionally:
   - IL-12 → IFN-γ (STAT4 direct → STAT1 induced; well-distinct gene sets)
   - IL-6 → STAT3 → STAT3-target induction (less clear; primary engages
     STAT3 directly already)
   - LPS → IL-6 → STAT3 (chained, would test the methodology on a 3-step
     cascade rather than direct A→B)

These five steps roughly correspond to: (1) honest statistics, (2) defeating
the post-hoc concern, (3) negative control, (4) corroboration by time, (5)
breadth of applicability. None requires retraining a model; all are feasible
on existing pseudotube data plus modest new builds. The first three are the
most important for moving from "promising lead" to "validated methodology".
