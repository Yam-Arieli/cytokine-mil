# Two controls on the direction statistic

Source record for two numbers that had been quoted in working notes with no script behind
them. Produced by `scripts/compute_direction_controls.py`; re-run it to reproduce every value
below. All inputs are sanctioned cascadir outputs listed in
`.claude/skills/cascadir-values/references/result_files.md` — nothing here recomputes coupling
or `cross_asym` by hand.

Computed 2026-08-21.

## Control 1 — the asymmetric magnitude baseline

**What it is.** `sign( median_T [ |s_T(a, S_a)| − |s_T(b, S_b)| ] )` — which of two conditions
responds more strongly *to its own signature*, aggregated across cell types exactly the way
`cross_asym` is.

**Why it is the right control.** The symmetric control (`directional_score`) is provably
invariant under swapping `a` and `b`, so it could never have scored above chance no matter
what it measured. The magnitude baseline is antisymmetric, so it *could*; it is the
"`a` simply responded harder" explanation of the direction recall made into a statistic.

**Result — Oesinghaus, the 16 audited benchmark pairs the thesis headline is scored on:**

| statistic | correct | recall |
|---|---:|---:|
| `cross_asym` | 14 / 16 | **88%** |
| asymmetric magnitude baseline | 7 / 16 | **44%** |

The two agree on the sign of only 7 of the 16 pairs. Response magnitude does not account for
the direction recall.

Per-pair detail: `oesinghaus_magnitude_baseline.csv`. Scored on Oesinghaus only — Sheu's
cascadir-native fit and the Immune Dictionary have no local per-cell-type file carrying the
self-engagement terms, and the legacy Sheu per-cell-type file belongs to a different fit
(86% direction recall) from the one the thesis quotes (100%), so pairing a magnitude row with
it would mismatch the fits.

## Control 2 — the additive-potential (rank-1) fit

**What it is.** Fit one scalar `theta` per condition minimising
`sum over observed pairs of (cross_asym(a,b) − (theta_a − theta_b))^2`, and report the
fraction of `sum(cross_asym^2)` it explains. A high fraction means the pairwise direction
calls are close to a single global ordering of conditions — a ladder — rather than
pair-specific judgements.

**The null matters and was missing from the earlier working note.** The fit has
`n_conditions − 1` free parameters against `n_pairs` observations, so a panel whose pair count
barely exceeds its condition count scores high on random values too. The floor column is that
null, measured over 400 draws of random values on the same edge set.

| panel | conditions | pairs | explained | null floor | excess |
|---|---:|---:|---:|---:|---:|
| Oesinghaus, 53 axes | 21 | 53 | 90.7% | 37.8% | **+52.9** |
| Sheu 5 h, cascadir-native | 7 | 21 | 95.1% | 28.4% | **+66.6** |
| Sheu 5 h, legacy fit | 7 | 21 | 74.6% | 29.0% | **+45.6** |
| Immune Dictionary | 12 | 12 | 96.0% | 82.9% | +13.1 |

**Reading.** The ladder structure is real on Oesinghaus and on both Sheu fits. The Immune
Dictionary panel is 12 pairs over 12 conditions, which leaves the fit one degree of freedom —
its 96% is arithmetic, not structure, and it should not be quoted. *(This corrects the earlier
working note, which listed ID's 96% as the strongest of the four.)*

On the Oesinghaus benchmark, the residual after removing the fitted ordering orients 8 of 16
pairs — chance. So the direction signal travels with the global ordering, not with the
pair-specific remainder.

**What this does and does not imply.** It bounds how much independent evidence any single
unlabelled pair carries: most of what the method says about a pair is implied by where its two
conditions sit in one ranking. It does *not* show the ranking is an artifact — the mechanism a
sceptic would propose for it, response amplitude, is exactly what Control 1 rules out, and a
near-total order is also what real cascade biology would produce.

## Files

| file | contents |
|---|---|
| `oesinghaus_magnitude_baseline.csv` | per-pair: expected direction, `cross_asym`, magnitude baseline, both correctness flags |
| `potential_fit.csv` | per-panel explained fraction, null floor, residual benchmark recall |
| `potential_theta.csv` | the fitted per-condition `theta` for each panel |
| `summary.json` | the Control 1 headline counts |
