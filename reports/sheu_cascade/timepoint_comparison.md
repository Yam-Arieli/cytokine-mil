# Sheu single-frame cascade-direction — cross-time comparison

Primary metric: **cross_asym** (antisymmetric cross-engagement; sign encodes direction). Each time point is a self-contained single-frame detection — no cross-time data used in the method.

## Directional accuracy by class (sign vs expected)

| time | IFN_MUST | IFN_SHOULD | NFKB_SHOULD | all directional |
|---|---|---|---|---|
| 1hr | 1/2 | — | 3/3 | 4/5 |
| 3hr | 1/2 | 1/1 | 3/4 | 5/7 |
| 5hr | 1/2 | 1/1 | 4/4 | 6/7 |

## Negative-pair specificity (|cross_asym|, lower ⇒ cleaner null)

| time | median \|cross_asym\| NEGATIVE | median \|cross_asym\| IFN benchmark |
|---|---|---|
| 1hr | 0.1070 | 0.1026 |
| 3hr | 0.1167 | 0.1366 |
| 5hr | 0.2427 | 0.1595 |

## Per-pair cross_asym kinetics (signed; correct sign in parens)

| pair | expected | 1hr | 3hr | 5hr |
|---|---|---|---|---|
| IFNb/LPS | − | -0.0933 (✓) | -0.0728 (✓) | -0.0208 (✓) |
| IFNb/PIC | − | +0.1119 (✗) | +0.1366 (✗) | +0.1595 (✗) |

## Headline

- Best single time frame for directional accuracy: **5hr** (86% on the 7 directional pairs).
- IFN cascades (MUST) are the clean test (distinct pathways, §24 precondition holds). NF-κB cascades (SHOULD) are expected weaker (pathway overlap). Negatives should stay near zero.
- Reminder: cross_asym (not the symmetric directional_score) is the direction-bearing quantity — on Oesinghaus 24h it scored 88% vs 47%.
