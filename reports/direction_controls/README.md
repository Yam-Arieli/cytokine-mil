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

## Control 2b — is the ordering emergent, and is it response strength?

**Why the ordering is not forced by construction.** Each pair is scored on *its own* two gene
sets: `cross_asym(X,Y)` uses `S_X` and `S_Y`, `cross_asym(X,Z)` uses `S_X` and `S_Z`. Nothing
makes `A[X,Y] + A[Y,Z] + A[Z,X]` come out near zero, because the three terms do not share a
scale. Transitivity across independently-derived gene sets is therefore an emergent
consistency, and it is measurable directly:

- **1 of 38 closed triangles among the 53 Oesinghaus axes is cyclic — 2.6%.** A random
  tournament is cyclic in 25% of triangles.

**The ordering recovered.** Top: IFN-ω, IFN-β, IFN-γ, IL-2, IL-15. Bottom: VEGF, IL-16, IL-9,
TNF-α, IL-6. Canonical upstream drivers above canonical downstream effectors.

**Is it just response strength?** Score the 16 benchmark pairs from each candidate ordering:

| ranking by | correct | recall |
|---|---:|---:|
| response strength, signed | 9 / 16 | 56% |
| response strength, absolute | 8 / 16 | 50% |
| the fitted ordering `theta` | 12 / 16 | **75%** |
| `cross_asym` itself | 14 / 16 | **88%** |

`theta` does correlate with response strength (Spearman +0.684, p = 0.0006, n = 21), but
ranking *by* strength sits at chance while the ordering reaches 75%. The discriminating part of
`theta` is orthogonal to amplitude.

> **Discrepancy reported, not reconciled.** An earlier working note put this correlation at
> +0.484 (p = 0.026, n = 21). This script measures +0.684 using median self-engagement across
> all cell types and axes against the 53-axis `theta`; the earlier note does not state its
> aggregation or panel. The sign and the conclusion are unaffected.

**What this does and does not imply.** `cross_asym` adds two pairs over the pure ordering
(88% against 75%), so most of what the method says about a given pair is implied by where its
two conditions already sit — an unlabelled candidate pair is not independent evidence, and the
candidate lists should be read with that in mind. It does *not* show the ordering is an
artifact: the mechanism a sceptic would propose for it, response amplitude, scores at chance;
the ordering is consistent across triangles measured on different gene sets; and a near-total
order is what real cascade biology would produce.

## Files

| file | contents |
|---|---|
| `oesinghaus_magnitude_baseline.csv` | per-pair: expected direction, `cross_asym`, magnitude baseline, both correctness flags |
| `potential_fit.csv` | per-panel explained fraction, null floor, residual benchmark recall |
| `potential_theta.csv` | the fitted per-condition `theta` for each panel |
| `ordering_comparison.csv` | benchmark recall from each candidate ordering (strength, `theta`, `cross_asym`) |
| `summary.json` | the Control 1 headline counts |
