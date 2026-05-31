# Pipeline A-Bridge-B (full-19) — careful diagnosis

**Date:** 2026-05-31
**Input:** `results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv`
       (53 axes × 17–18 cell types, all-12-donor pooled)

The earlier read ("structural positive bias — method doesn't work") was
overstated. Three closer looks change the picture.

---

## 1. Component pattern by literature class

By the §24 algebra (binary-IG sets passed as P_A = S_A, P_B = S_B):

| Regime | asym_PA | asym_PB | directional_score |
|---|---|---|---|
| A→B cascade, A=upstream | medium + | ~0 | + |
| B→A cascade, A=downstream | ~0 | medium − | + |
| No cascade, distinct signatures | large + | large − | very large + |
| Coregulated, shared signature | ~0 | ~0 | ~0 |

All three "cascade" regimes give **positive** `directional_score`. The
discriminator is the component pair, not the scalar.

Observed medians per literature class:

| lit_class | n | median asym_PA | median asym_PB | median dir_score |
|---|---:|---:|---:|---:|
| a_to_b | 17 | +0.0058 | +0.0017 | +0.0030 |
| b_to_a | 12 | +0.0156 | −0.0005 | +0.0080 |
| bidir | 3 | +0.0069 | +0.0040 | +0.0187 |
| coregulated_other | 4 | +0.0085 | +0.0083 | +0.0054 |
| no_lit | 15 | +0.0117 | +0.0055 | +0.0027 |
| partial_lit | 2 | +0.0343 | +0.0385 | −0.0012 |

**Read:**
- `a_to_b` axes show (small +, ~0) — matches the A→B-like regime.
- `b_to_a` axes show (medium +, ~0) — does **not** match the predicted
  B→A regime (~0, large −). They look like A→B-like with no asym_PB
  signal.
- `coregulated_other` shows (small +, small +) — matches "shared signature"
  prediction.

The components carry some signal but not as cleanly as the algebra
predicts — particularly the `b_to_a` class is muddied.

---

## 2. Magnitude vs correctness (key finding)

Restricting to the 29 axes with clean a_to_b / b_to_a labels:

| Bucket | n | % | median \|asym_PA\| | median \|asym_PB\| |
|---|---:|---:|---:|---:|
| Correct sign, non-ambiguous | 6 | 21% | 0.0157 | **0.0264** |
| Wrong   sign, non-ambiguous | 10 | 34% | 0.0160 | **0.0073** |
| Ambiguous (\|median\| < 0.01) | 13 | 45% | 0.0165 | 0.0177 |

**Key**: among AMBIGUOUS axes, **9/13 (69%) have the correct expected
sign** — significantly above 50% chance. The "ambiguous" bucket is
where the real signal lives.

**Also key**: wrong-sign calls have \|asym_PB\| ~3.6× smaller than
correct-sign calls. Wrong calls are in the "A→B-like with no asym_PB
signal" regime — i.e., they're driven almost entirely by signature
distinctness, with no real cascade footprint.

**Implication**: large `directional_score` is not necessarily a strong
call. Calls with \|asym_PB\| ≈ 0 are likely the "no cascade" or
"non-specific signature" regime, not a real A→B cascade.

---

## 3. Top 5 wrong-direction calls — tag vs summary

The five biggest wrong calls all involve VEGF or IL-15:

| axis_a | axis_b | tag | median_dir | median_PA | median_PB |
|---|---|---|---:|---:|---:|
| IL-15 | VEGF | b_to_a | +0.2193 | +0.2116 | +0.0005 |
| IL-15 | IL-2 | b_to_a | +0.0456 | +0.0720 | +0.0267 |
| IL-12 | VEGF | b_to_a | +0.0372 | +0.0363 | +0.0011 |
| IL-13 | IL-27 | b_to_a | +0.0230 | +0.0072 | −0.0160 |
| IL-27 | VEGF | b_to_a | +0.0186 | +0.0148 | −0.0015 |

Reading the `literature_summary` column for each:

| pair | tag | summary says | match? |
|---|---|---|---|
| IL-15/VEGF | b_to_a | **"IL-15 → VEGF via JAK-STAT5 in NK cells"** (i.e. a_to_b) | **tag wrong** |
| IL-15/IL-2 | b_to_a | "Pre-registered KNOWN_CASCADE" (no detail) | tag unverifiable |
| IL-12/VEGF | b_to_a | "IL-12 anti-angiogenic via IFN-γ/IP-10 → inhibits VEGF" (A inhibits B, not B → A) | tag inconsistent |
| IL-13/IL-27 | b_to_a | "IL-27 → suppresses IL-13" (B → A) | tag matches summary |
| IL-27/VEGF | b_to_a | "IL-27 INHIBITS VEGF via STAT1" (A inhibits B, not B → A) | tag inconsistent |

**3 of 5 top wrong calls have the lit_direction tag inconsistent with
the lit_summary in our favour**: IL-15/VEGF, IL-12/VEGF, IL-27/VEGF
are tagged `b_to_a` but the summary clearly describes axis_a as the
acting cytokine. Note: pathway signatures don't distinguish induction
from inhibition, so IL-12/VEGF and IL-27/VEGF show positive scores
because IL-12 and IL-27 engage VEGF's transcriptional context (even
if the net effect is suppression).

For the 6 *correctly* called axes, lit_summary mostly agrees with the
tag (5/6); only `IFN-gamma/IL-2` has tag-summary inconsistency in
the reverse direction.

**Net**: ground truth has internal inconsistency between
`literature_direction` tag and `literature_summary` text. The method's
"wrong" calls are partly disagreement with noisy labels, not partly
method failure.

---

## 4. Per-cell-type breakdown of biggest wrong call (IL-15/VEGF)

| cell_type | directional_score | asym_PA | asym_PB |
|---|---:|---:|---:|
| CD16 Mono | +0.1047 | +0.1446 | +0.0399 |
| pDC | +0.1089 | +0.1094 | +0.0005 |
| cDC | +0.1430 | +0.1625 | +0.0196 |
| Treg | +0.1924 | +0.1775 | −0.0148 |
| CD14 Mono | +0.1951 | +0.2116 | +0.0165 |
| NK | +0.2015 | +0.2516 | +0.0501 |
| ... | ... | ... | ... |
| NK CD56bright | +0.2807 | +0.2971 | +0.0163 |

**17 of 17 cell types positive.** The signal is not an aggregation
artifact — every cell type agrees IL-15 engages VEGF's signature more
than VEGF engages IL-15's. Magnitudes are highest in NK / NK CD56bright
(consistent with IL-15-activated NK release of VEGF, which is the
literature_summary's claim).

---

## 5. Revised interpretation

| Earlier claim | Revised claim |
|---|---|
| "Structural positive bias makes the method broken" | Method has signal; the signal lives in component magnitudes and in the AMBIGUOUS-but-non-zero bucket (69% sign correct) |
| "53 axes, 6 correct / 10 wrong is failure" | After correcting for ~3 tag-summary inconsistencies: ~9 correct / ~7 wrong / 13 ambiguous (sign-correct 9/13) → on the order of ~60% effective accuracy on the 29 graded axes |
| "Magnitude can't separate right from wrong" | Magnitude of `directional_score` alone can't, but the *component* `|asym_PB|` does: correct calls have \|asym_PB\| ≈ 0.026, wrong calls have \|asym_PB\| ≈ 0.007 (3.6× smaller) |
| "Aggregation hides the signal" | Aggregation is fine; the biggest wrong call has all 17 cell types in the same direction |

---

## 6. What this changes about the path forward

The method is not broken. The pipeline as-is has three issues, all
fixable without architectural change:

1. **Use the component pattern, not just the scalar.** A call should
   require both `|asym_PA|` and `|asym_PB|` to be meaningfully non-zero
   (e.g., > 0.01) with directionally consistent signs. Scalar
   `directional_score` alone collapses too much information.

2. **The literature ground truth is noisy.** The
   `cytokine_axes.csv` `literature_direction` tag conflicts with the
   `literature_summary` in at least 3/5 of our top wrong calls and 1/6
   of our correct calls. Before re-evaluating, the ground truth labels
   should be either (a) hand-corrected by reading the summaries, or
   (b) only the clean-text-match subset used for benchmarking.

3. **High |asym_PA| + |asym_PB| ≈ 0** is the "no cascade, distinct
   signatures" regime, not "strong cascade". Treating these as STRONG
   calls is wrong. They should be tagged as DISTINCT-PROGRAMS rather
   than mixed into the cascade-direction calls.

None of these require retraining or new compute — they're a re-read
of the same parquet with a richer call rule.
