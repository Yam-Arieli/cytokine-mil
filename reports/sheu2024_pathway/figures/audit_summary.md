# Pathway-Signature Cascade Methodology: Adversarial Audit

Four independent validations of the §23 cascade-direction methodology.
Each is designed to falsify a specific concern raised in peer review.

## Audit 1: Cytokine-label permutation null on binary IFNAR test

**Concern addressed:** Is the observed AUC structurally inflated by small-N selection effects, or is it distinguishable from null?

Per cell type:

| cell_type | observed AUC | null mean | null Q95 | p_emp |
|---|---:|---:|---:|---:|
| mac_c2 | 1.000 | 0.497 | 1.000 | 0.0960 |
| mac_c3 | 0.667 | 0.488 | 0.889 | 0.3300 |

**0 of 2 cell types have observed AUC above the permutation null at p<0.05.**

## Audit 2: Random-pathway null (curated genes vs random gene sets)

**Concern addressed:** Is the curated gene list carrying specific information, or do random gene sets of the same size give equally high AUCs (which would mean we are reading overall activation, not pathway specificity)?

Per cell type, the AUC distribution from 200 random gene sets of the same size as the curated IFNAR_induced signature:

| cell_type | random mean | random Q95 | random Q99 |
|---|---:|---:|---:|
| mac_c2 | 0.610 | 1.000 | 1.000 |
| mac_c3 | 0.487 | 1.000 | 1.000 |

Compare to observed curated AUC in `audit_1_permutation_null.csv`. If curated AUC > random Q95, the gene-list specificity matters.

## Audit 3: Per-donor (not per-cell) magnitude tests

**Concern addressed:** The original Mann-Whitney p-values (`p ≈ 10⁻¹³¹`) were per-cell, with hundreds of cells per tube sharing donors. These are fiction. The unit of replication is the donor (~3-4 per condition). Honest p-values via Wilcoxon signed-rank on donor-paired means.

| pathway | A | cell_type | n_donors | mean_A | mean_TNF | mean_PBS | Wilcoxon p (A>TNF) | sign-test p | order ok |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| NFkB_canonical | LPS | mac_c2 | 1 | 1.762 | 1.307 | 0.835 | 0.5000 | 0.5000 | True |
| NFkB_canonical | LPS | mac_c3 | 4 | 3.088 | 2.240 | 2.559 | 0.0625 | 0.0625 | True |
| NFkB_canonical | LPSlo | mac_c3 | 1 | 3.451 | 2.707 | 2.559 | 0.5000 | 0.5000 | True |
| NFkB_canonical | P3CSK | mac_c0 | 1 | 1.567 | 1.622 | 0.591 | 1.0000 | 1.0000 | False |
| NFkB_canonical | P3CSK | mac_c1 | 1 | 3.143 | 2.428 | 1.118 | 0.5000 | 0.5000 | True |
| NFkB_canonical | P3CSK | mac_c2 | 1 | 1.690 | 1.307 | 0.835 | 0.5000 | 0.5000 | True |
| NFkB_canonical | P3CSK | mac_c3 | 4 | 2.877 | 2.240 | 2.559 | 0.0625 | 0.0625 | True |
| NFkB_canonical | CpG | mac_c0 | 1 | 1.978 | 1.622 | 0.591 | 0.5000 | 0.5000 | True |
| NFkB_canonical | CpG | mac_c1 | 1 | 2.830 | 2.428 | 1.118 | 0.5000 | 0.5000 | True |
| NFkB_canonical | CpG | mac_c2 | 1 | 1.933 | 1.307 | 0.835 | 0.5000 | 0.5000 | True |
| NFkB_canonical | CpG | mac_c3 | 4 | 3.012 | 2.240 | 2.559 | 0.0625 | 0.0625 | True |

**0 of 11 (pathway × cell_type × pair) tests pass donor-level α=0.05 with the predicted ordering.**

With only ~3 donors per (A, B) pair, the minimum achievable Wilcoxon signed-rank p is ~0.125. So 'pass' here means: the donor-level means consistently follow the predicted ordering (which is the substantive claim), even if formal significance is unreachable at this donor count.

## Audit 4: Directional asymmetry (the test originally skipped)

**Concern addressed:** High penetration of pathway P by stimulus A proves A engages P. It does NOT prove A → B cascade through P_B. A genuine cascade A→B requires: A engages P_A (A's own pathway) AND A engages P_B (B's pathway, via autocrine cascade) AND B engages P_B but NOT P_A. This audit computes the four quantities and the directional asymmetry score.

**Predicted pattern for true cascade A→B:**
- `asym_PA` (= s_A_norm − s_B_norm on P_A) **positive** (A engages P_A; B doesn't)
- `asym_PB` (= s_A_norm − s_B_norm on P_B) **≈ 0** (both engage P_B)
- `directional_score` = asym_PA − asym_PB → **positive** for true A→B

| A | B | P_A | P_B | cell_type | asym_PA | asym_PB | directional_score | interpretation |
|---|---|---|---|---|---:|---:|---:|---|
| PIC | IFNb | IRF3_direct | IFNAR_induced | mac_c2 | 0.784 | -1.090 | 1.874 | A->B (cascade A drives B's pathway, not vice versa) |
| PIC | IFNb | IRF3_direct | IFNAR_induced | mac_c3 | 0.511 | -1.155 | 1.666 | A->B (cascade A drives B's pathway, not vice versa) |
| LPS | IFNb | IRF3_direct | IFNAR_induced | mac_c2 | 0.830 | -1.519 | 2.349 | A->B (cascade A drives B's pathway, not vice versa) |
| LPS | IFNb | IRF3_direct | IFNAR_induced | mac_c3 | 0.704 | -1.313 | 2.017 | A->B (cascade A drives B's pathway, not vice versa) |
| LPS | TNF | NFkB_canonical | TNFR_autocrine | mac_c2 | 0.456 | 0.548 | -0.093 | B->A (reverse direction would be implied) |
| LPS | TNF | NFkB_canonical | TNFR_autocrine | mac_c3 | 0.835 | 0.331 | 0.504 | A->B (cascade A drives B's pathway, not vice versa) |
| LPSlo | TNF | NFkB_canonical | TNFR_autocrine | mac_c1 | 0.666 | 0.156 | 0.509 | A->B (cascade A drives B's pathway, not vice versa) |
| LPSlo | TNF | NFkB_canonical | TNFR_autocrine | mac_c3 | 1.197 | 0.355 | 0.842 | A->B (cascade A drives B's pathway, not vice versa) |
| P3CSK | TNF | NFkB_canonical | TNFR_autocrine | mac_c0 | -0.055 | -0.223 | 0.168 | A->B (cascade A drives B's pathway, not vice versa) |
| P3CSK | TNF | NFkB_canonical | TNFR_autocrine | mac_c1 | 0.780 | 0.276 | 0.503 | A->B (cascade A drives B's pathway, not vice versa) |
| P3CSK | TNF | NFkB_canonical | TNFR_autocrine | mac_c2 | 0.383 | 0.433 | -0.050 | B->A (reverse direction would be implied) |
| P3CSK | TNF | NFkB_canonical | TNFR_autocrine | mac_c3 | 0.649 | 0.413 | 0.237 | A->B (cascade A drives B's pathway, not vice versa) |
| CpG | TNF | NFkB_canonical | TNFR_autocrine | mac_c0 | 0.357 | 0.403 | -0.046 | B->A (reverse direction would be implied) |
| CpG | TNF | NFkB_canonical | TNFR_autocrine | mac_c1 | 0.402 | 0.499 | -0.097 | B->A (reverse direction would be implied) |
| CpG | TNF | NFkB_canonical | TNFR_autocrine | mac_c2 | 0.626 | 0.601 | 0.026 | A->B (cascade A drives B's pathway, not vice versa) |
| CpG | TNF | NFkB_canonical | TNFR_autocrine | mac_c3 | 0.759 | 0.500 | 0.260 | A->B (cascade A drives B's pathway, not vice versa) |

**12 of 16 (pair, cell_type) observations show positive directional_score (consistent with A→B cascade direction).**

---

## Honest reading

All four audits are diagnostics, not validations. Specifically:

- Audit 1: tells us whether random label assignment can reproduce the observed AUC. Strong if p_emp < 0.01.
- Audit 2: tells us whether the curated gene list is doing work, or whether any random gene set would yield similar discrimination.
- Audit 3: replaces the fictional per-cell p-values with honest donor-level statistics. With only ~3 donors per condition this is low-powered but rigorous.
- Audit 4: tests whether the data shows directional cascade evidence (vs pure pathway engagement). Positive directional_score is necessary but not sufficient evidence for cascade direction; interventional follow-up (e.g., IFNAR-KO) would close the loop.

Combining the four: if Audit 1 and Audit 2 both reject null at p<0.01 AND Audit 3 shows consistent donor-level ordering AND Audit 4 shows positive directional_score, the result is robust against the most likely peer-review concerns. If any one fails, that's the specific weakness to address before publication.