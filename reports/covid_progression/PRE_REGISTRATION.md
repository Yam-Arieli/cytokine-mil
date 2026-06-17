# Pre-registration — COVID severity-progression cascade-direction test (§30)

**Locked before the analysis job runs** (per CLAUDE.md §25.1). Method, oracle, verdict
thresholds, and calibration predictions are fixed here; the analysis script
(`scripts/analyze_covid_progression.py`) scores against exactly this file.

## Hypothesis

From a SINGLE cross-sectional snapshot of PBMC scRNA-seq, the antisymmetric
`cross_asym` statistic recovers the DIRECTION of COVID-19 disease progression
(less-severe = upstream), validated against the known clinical severity order. This is
the disease-progression analog of the cytokine cascade-direction result (§26), and a
direct test of the project thesis ("programs that span time, read from one snapshot").

## Dataset & design (locked)

- Stephenson 2021 Haniffa COVID-19 PBMC atlas (`haniffa21.processed.h5ad`).
- Conditions = 5 ordered severity grades; control = `Healthy`; drop `LPS`/`Non_covid`.
- Method = `cascadir.CascadeDirection.fit(adata).direction_table()` (encoder → per-grade
  binary models → IG signatures → `cross_asym`), fit-from-h5ad.
- Donor = patient (NESTED in grade) → donor-level rigour via the donor-bootstrap
  (`cascadir.progression.bootstrap_cross_asym`), NOT within-donor pairing.

## Oracle (locked ordered pairs, less-severe upstream)

Ordered severity: **Asymptomatic < Mild < Moderate < Severe < Critical**. All C(5,2)=10
ordered pairs (upstream, downstream):

| upstream | downstream |    | upstream | downstream |
|---|---|---|---|---|
| Asymptomatic | Mild | | Mild | Severe |
| Asymptomatic | Moderate | | Mild | Critical |
| Asymptomatic | Severe | | Moderate | Severe |
| Asymptomatic | Critical | | Moderate | Critical |
| Mild | Moderate | | Severe | Critical |

The 4 **adjacent** pairs (Asymp–Mild, Mild–Moderate, Moderate–Severe, Severe–Critical)
are the cleanest. (If a grade is absent after QC, the oracle is restricted to grades
present in `direction_table`; this is reported, not retrofitted.)

## Calibration predictions (locked)

- **P1 (apparatus power).** The synthetic **distinct-program** ladder is recovered at
  cross_accuracy = 100% while its symmetric `directional_score` control is ≈ chance
  (`scripts/apparatus_cross_asym_ladder.py`; verified locally before submission — see
  `APPARATUS_GATE_RESULTS.md`). HARD GATE for trusting the method at all.
- **P2 (confound characterization).** The synthetic **monotone-intensity** ladder
  flips/degrades `cross_asym` without a seed, and is rescued by a seed — fixing the
  regime the COVID severity axis must be read against.
- **P3 (headline, real data).** On COVID, `cross_asym` accuracy ≫ the symmetric
  `directional_score` control (which sits ≈ chance because the grades' alphabetical
  order ≠ severity order). This is what distinguishes a genuine progression seed from
  mere severity magnitude.
- **P4 (donor robustness).** The donor-bootstrap 95% CI of sign-accuracy excludes 0.5,
  and Kendall τ(recovered order, true order) ≥ 0.6.

## Verdict thresholds (locked)

- **GREEN** — `cross_accuracy ≥ 0.80` (10 pairs) AND `dirscore_accuracy ≤ 0.60`
  (clearly below cross) AND donor-bootstrap accuracy 95% CI lower bound > 0.50 AND
  Kendall τ ≥ 0.60 AND the apparatus distinct-gate passed. ⇒ a single snapshot encodes
  progression direction beyond severity magnitude.
- **AMBER** — `cross_accuracy` in [0.60, 0.80), OR `dirscore_accuracy ≈ cross_accuracy`
  (magnitude not separated), OR the bootstrap CI includes 0.50. ⇒ signal present but
  confounded / underpowered; report honestly, do not claim direction.
- **RED** — `cross_accuracy ≤ 0.60`, OR the apparatus distinct-gate fails. ⇒ direction
  not recoverable on this axis with this method.

## Validity boundaries (declared up front, apply to any outcome)

Cross-sectional (direction, NOT per-subject forecast); nested donors (bootstrap, not
within-donor pairing); severity = one-disease monotone-intensity axis (the magnitude
confound — this is "state-of-health → state-of-health", NOT "disease A → disease B");
PBMC blood only; direction ≠ causation; single dataset.

## Locked outputs

`results/covid_progression/{summary.json, benchmark.csv, bootstrap_per_pair.csv,
plots/*.pdf}`; `reports/covid_progression/{APPARATUS_GATE_RESULTS.md,
COVID_PROGRESSION_RESULTS.md}`.

_Pre-registered before submission of the fit/analysis DAG; method basis = CLAUDE.md §26/§28/§30._
