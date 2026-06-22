# Pre-registration — vaccine T-cell maturation cascade-direction test (§32)

**Locked before the analysis jobs run** (per CLAUDE.md §25.1). Method, oracle, verdict
thresholds, and calibration predictions are fixed here; the analysis script
(`scripts/analyze_vaccine_progression.py`) scores against exactly this file. Two framings
are pre-registered: **STATE (headline)** and **TIMEPOINT (corroboration)**.

## Hypothesis

From a SINGLE cross-sectional snapshot of whole-PBMC CITE-seq, the antisymmetric
`cross_asym` statistic recovers the DIRECTION of T-cell maturation
(**naive → effector → memory**), validated against the textbook differentiation order.
This is the cell-state-differentiation analog of the cytokine cascade-direction result
(§26) and the disease-progression result (§30). It is **direction VALIDATION on a
gold-standard order, not discovery** — the order is known; the claim is that one snapshot
recovers it (same epistemic status as the §26 labeled-pair benchmark).

## Dataset & design (locked)

- Multimodal SARS-CoV-2 vaccination + infection PBMC atlas (Nat Immunol 2023; Zenodo
  `7555405`, `PBMC_vaccine_CITE.rds`). Whole-PBMC CITE-seq, in vivo, day 0/2/10/28, ~6 donors.
- **Subset to T cells.** Maturation `state` assigned from CITE **surface protein** gating
  (CD45RA/CCR7 quadrant; CD27/CD95 if present) — *independent of the RNA the signatures are
  built from* (breaks the "states defined by the same genes" circularity) — else the fine
  author annotation.
- Method = `cascadir.CascadeDirection.fit(adata).direction_table()` (encoder → per-condition
  binary models → IG signatures → `cross_asym`), fit-from-h5ad.
- Donor = `subject`; cell type = `tcell_lineage` (CD4 T / CD8 T). ~6 donors → donor-level
  rigour via the donor-bootstrap (`cascadir.progression.bootstrap_cross_asym`), NOT the
  8+-donor coupling gate.

## Oracles (locked)

**STATE (headline).** Ordered: **Naive < Effector < Memory**; control = `Resting` (day-0
T cells). C(3,2)=3 ordered pairs (upstream, downstream): (Naive, Effector), (Naive, Memory),
(Effector, Memory). Naive→Effector and Effector→Memory are the adjacent (cleanest) pairs.

**TIMEPOINT (corroboration).** Ordered: **D2 < D10 < D28**; control = `D0`. C(3,2)=3 pairs:
(D2, D10), (D2, D28), (D10, D28). The weaker monotone-intensity framing (§30 caveat), run to
confirm states emerge in time order.

(If a condition is absent after QC, the oracle is restricted to conditions present in
`direction_table`; reported, not retrofitted.)

## Calibration predictions (locked)

- **P1 (apparatus power).** The synthetic **distinct-program** ladder is recovered at
  cross_accuracy = 100% while its symmetric `directional_score` control is ≈ chance
  (`scripts/apparatus_cross_asym_ladder.py`). HARD GATE for trusting the method at all.
- **P2 (confound characterization).** The synthetic **monotone-intensity** ladder
  flips/degrades `cross_asym` without a seed and is rescued by a seed — the regime the
  maturation/time axis must be read against (activation is partly monotone-intensity).
- **P3 (headline, real data — STATE).** `cross_asym` accuracy ≫ the symmetric
  `directional_score` control. This distinguishes a genuine maturation seed from mere
  activation magnitude.
- **P4 (donor robustness).** The donor-bootstrap 95% CI of sign-accuracy excludes 0.5, and
  Kendall τ(recovered order, true order) ≥ 0.6.
- **P5 (timepoint corroboration).** The TIMEPOINT framing recovers D0<D2<D10<D28 with
  cross_accuracy ≥ 0.6 (states tracking the clock) — supportive, not a gate.

## Verdict thresholds (locked; applied to the STATE run as the headline)

- **GREEN** — `cross_accuracy ≥ 0.80` AND `dirscore_accuracy ≤ 0.60` (clearly below cross)
  AND donor-bootstrap accuracy 95% CI lower bound > 0.50 AND Kendall τ ≥ 0.60 AND the
  apparatus distinct-gate passed. ⇒ a single snapshot encodes maturation direction beyond
  activation magnitude.
- **AMBER** — `cross_accuracy` in [0.60, 0.80), OR `dirscore_accuracy ≈ cross_accuracy`
  (magnitude not separated), OR the bootstrap CI includes 0.50. ⇒ signal present but
  confounded / underpowered; report honestly, do not claim direction.
- **RED** — `cross_accuracy ≤ 0.60`, OR the apparatus distinct-gate fails. ⇒ direction not
  recoverable on this axis with this method.

## Validity boundaries (declared up front, apply to any outcome)

VALIDATION not discovery (textbook order); cross-sectional (direction, NOT per-cell
forecast); **early memory only** (mRNA vaccine, day 28 → naive→effector→**early**-memory,
not the full central-memory arc); ~6 donors (donor-bootstrap, underpowered); maturation/time
is a monotone-intensity-prone axis (the magnitude confound, addressed by the cross-vs-control
gap + per-cell-type consensus); PBMC blood only; direction ≠ causation; single dataset. The
cross-cell-type **relay** question (cell type A's state → cell type B's fate) is OUT OF SCOPE
here (a later track this dataset also enables).

## Locked outputs

`results/vaccine_progression/{state,timepoint}/{summary.json, benchmark.csv,
bootstrap_per_pair.csv, plots/*.pdf}`; `results/vaccine_progression/apparatus/`;
`reports/vaccine_progression/{VACCINE_PROGRESSION_RESULTS.md (state, headline),
VACCINE_PROGRESSION_RESULTS_timepoint.md}`.

_Pre-registered before submission of the fit/analysis DAG; method basis = CLAUDE.md §26/§28/§30/§32._
