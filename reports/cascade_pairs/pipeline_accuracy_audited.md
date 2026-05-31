# Pipeline accuracy re-tallied against strict audited labels

- Pipeline output: `results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv`
- Audited labels: `reports/cascade_pairs/cytokine_axes_audited.csv`

## Per-axis aggregation rule
Median of `directional_score` across cell types, plus sign-consensus fraction. Matches the pipeline's median+consensus aggregator.

## Headline accuracy (strict audited benchmark)

- Axes in benchmark: **17** (DIRECTIONAL_a_to_b + DIRECTIONAL_b_to_a)
- Sign-correct calls: **8 / 17 = 47%**

### Per status sub-tally

| pair_status | n | n_correct | accuracy |
|---|---:|---:|---:|
| DIRECTIONAL_a_to_b | 10 | 6 | 60% |
| DIRECTIONAL_b_to_a | 7 | 2 | 29% |

### Per-axis detail (strict benchmark)

| axis | pair_status | expected | observed_sign | median | consensus | tag_changed | correct |
|---|---|:-:|:-:|---:|---:|:-:|:-:|
| CD30L / IL-17A | DIRECTIONAL_a_to_b | + | − | -0.0149 | 1.00 |  | ✗ |
| GM-CSF / TL1A | DIRECTIONAL_a_to_b | + | + | +0.0002 | 0.50 |  | ✓ |
| IFN-lambda1 / IL-6 | DIRECTIONAL_a_to_b | + | − | -0.0003 | 0.53 |  | ✗ |
| IFN-omega / IL-15 | DIRECTIONAL_a_to_b | + | + | +0.0546 | 0.89 |  | ✓ |
| IL-15 / VEGF | DIRECTIONAL_a_to_b | + | + | +0.2193 | 1.00 | ✓ | ✓ |
| IL-16 / IL-6 | DIRECTIONAL_a_to_b | + | + | +0.0234 | 0.94 |  | ✓ |
| IL-36-alpha / IL-9 | DIRECTIONAL_a_to_b | + | − | -0.0056 | 0.78 | ✓ | ✗ |
| IL-36-alpha / VEGF | DIRECTIONAL_a_to_b | + | + | +0.0014 | 0.59 |  | ✓ |
| IL-6 / VEGF | DIRECTIONAL_a_to_b | + | + | +0.0133 | 0.76 |  | ✓ |
| IL-9 / VEGF | DIRECTIONAL_a_to_b | + | − | -0.0037 | 0.65 | ✓ | ✗ |
| IFN-gamma / IFN-omega | DIRECTIONAL_b_to_a | − | − | -0.0161 | 0.83 | ✓ | ✓ |
| IFN-gamma / IL-2 | DIRECTIONAL_b_to_a | − | + | +0.0327 | 1.00 | ✓ | ✗ |
| IL-13 / TL1A | DIRECTIONAL_b_to_a | − | + | +0.0253 | 0.89 |  | ✗ |
| IL-13 / VEGF | DIRECTIONAL_b_to_a | − | + | +0.0074 | 0.76 | ✓ | ✗ |
| IL-15 / IL-2 | DIRECTIONAL_b_to_a | − | + | +0.0456 | 1.00 |  | ✗ |
| IL-17A / IL-36-alpha | DIRECTIONAL_b_to_a | − | − | -0.0182 | 0.88 | ✓ | ✓ |
| IL-6 / TNF-alpha | DIRECTIONAL_b_to_a | − | + | +0.0024 | 0.53 |  | ✗ |

## Weak benchmark (reported separately)

Axes with WEAK_* or DIRECTIONAL_*_NOISY status. Sign has direction but evidence is weak.

- Axes in weak benchmark: **7**
- Sign-correct: **4 / 7 = 57%**

| axis | pair_status | expected | observed_sign | median | tag_changed | correct |
|---|---|:-:|:-:|---:|:-:|:-:|
| IFN-beta / IL-2 | WEAK_a_to_b | + | + | +0.0349 |  | ✓ |
| IL-12 / IL-9 | WEAK_a_to_b | + | − | -0.0117 |  | ✗ |
| IL-35 / VEGF | WEAK_a_to_b | + | + | +0.0030 |  | ✓ |
| IL-36-alpha / IL-6 | WEAK_a_to_b | + | − | -0.0065 |  | ✗ |
| Decorin / IL-6 | WEAK_b_to_a | − | − | -0.0007 |  | ✓ |
| IL-16 / IL-9 | WEAK_b_to_a | − | + | +0.0063 |  | ✗ |
| IL-9 / TNF-alpha | WEAK_b_to_a | − | − | -0.0154 |  | ✓ |

## Excluded from accuracy (no graded sign)

| pair_status | n | n_positive_score | n_negative_score | n_zero |
|---|---:|---:|---:|---:|
| LOW_CONFIDENCE | 2 | 1 | 1 | 0 |
| PARTIAL_INHIBITORY | 7 | 6 | 1 | 0 |
| UNKNOWN | 20 | 15 | 5 | 0 |

## Delta vs original keyword-parsed tags

Original cytokine_axes.csv tags (any axis with original_direction ∈ {a_to_b, b_to_a}):
- Axes graded by original tag: **29**
- Sign-correct vs original tag (all): **15 / 29 = 52%**
- Of those AMBIGUOUS (|median| < 0.01): **13**
- Sign-correct among non-ambiguous: **6 / 16 = 38%**

- Original tags flipped by the audit: **7** (of 29 graded)

Interpretation: the audit reassigns ~7 of the 29 original tags. The strict benchmark on audited labels (8 / 17 = 47%) is the headline number — the original-tag accuracy is biased by mislabels and not directly comparable.
