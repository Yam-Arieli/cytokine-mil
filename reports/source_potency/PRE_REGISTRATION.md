# Pre-registration — Source-potency ranking from training dynamics

**Locked before the validation run (§25.1 discipline).** Objective/spec: the approved plan +
CLAUDE.md dynamics conventions. This fixes the score, the confound guard, and the GREEN/AMBER/RED
gates so the verdict is not chosen after seeing the result. This is a per-cytokine **magnitude**
(source-potency) claim, NOT a direction/existence/causation claim.

## Objective
A validated per-cytokine **source-potency** score — how strongly a cytokine acts as a cascade
*source* (depth×width) — read from the SHAPE of its multiclass learning curve, complementing
cascadir's directed edges. Deliverable = the ranked table + validation.

## Data (locked)
Oesinghaus 24h PBMC, multiclass 91-class, seeds 42/123/7 — reuse existing `dynamics.pkl`
`p_correct_trajectory` records (no new training). Donor-level aggregation: median across tubes per
donor, then mean across donors (project convention). Seed-averaged.

## Score (locked)
Per cytokine, on the donor-mean `p_correct(t)`:
- `P_max` = max over epochs (ceiling).
- `normalized_auc` = trapz(traj/P_max)/(n−1) — HIGH ⇒ plateaus early ⇒ shallow.
- `plateau_epoch` = first epoch ≥ **0.9**·P_max — LATE ⇒ deep.
- `late_gain` = rise over the final **1/3** (§8.3 secondary rise).
- **`source_potency = z(1 − normalized_auc) + z(late_gain)`**, z-scored over INCLUDED cytokines.

**Confound guard (single most important):** a merely *hard/unlearnable* cytokine (low `P_max`)
also plateaus late. Score is read **only** among cytokines with `P_max ≥ 0.1` (`ceiling_floor`),
and shape is always reported vs ceiling (the 2-axis figure).

## Ground truth (locked)
- **DEEP pool:** IL-12, IL-32-beta, OSM, IL-22, VEGF, HGF, TGF-beta1, IL-6.
- **SHALLOW pool:** IL-4, IL-10, IL-2, M-CSF, TNF-alpha, IL-1-beta, IFN-beta, IL-7, G-CSF.
  (provenance: `scripts/run_bootstrap.py` SIMPLE/COMPLEX = the early figure's blue/red.)
- Directed **out-degree**: source end of `counts_in_benchmark=True` edges in
  `cytokine_axes_audited.csv` (`expected_sign` +1⇒axis_a source, −1⇒axis_b source).
- **Coupling degree**: undirected degree in `cytokine_axes.csv` (121 axes).

## Predictions and gates
| ID | prediction | GREEN | AMBER | RED |
|---|---|---|---|---|
| **P1** (headline: dynamics⇔graph source) | deeper-learning ⇒ more a graph source | Spearman(potency, out-degree) > 0 at p<0.05 | ρ > 0 | ρ ≤ 0 |
| **P2** (width) | source cytokines are coupling hubs | Spearman(potency, coupling-degree) > 0 at p<0.05 | ρ > 0 | ρ ≤ 0 |
| **P3** (pre-registered pools) | DEEP > SHALLOW potency | one-sided perm p < 0.05 | Δ > 0 | Δ ≤ 0 |
| **P4** (literature) | master-regulators rank high | descriptive; ≥ half in top third | — | — |

**Overall GREEN** iff **P1 and P3** are both GREEN (the dynamics source score is validated against
the *directed* graph AND the pre-registered pools). AMBER if one is GREEN / both AMBER. RED if P1 or
P3 is RED (learning-curve lateness does not track cascade source).

## Honest caveats
The two assumptions (ML late=weak-secondary; bio deep-cascade=more-secondary) are logical but
unproven — this validates them, it does not assume them. Learnability is confounded by intrinsic
cytokine detectability (mitigated by the ceiling floor + reporting vs P_max, not removed). Small
pools (8–9). Dynamics pipeline is seed-noisy — seed-averaged, report per-seed spread if borderline.
