# Thesis Wonderings

This file collects open questions and methodological wonderings that come up while
writing the thesis — points worth revisiting, challenging, or validating that are not yet
resolved in the document itself. Each entry records the concern, why it matters, and the
current understanding or status, so nothing gets lost between writing sessions. These are
working notes, not claims made in the thesis.

---

## 1. Is the median the right cross-cell-type aggregation for direction?

**Concern.** The direction statistic aggregates the per-cell-type asymmetry
`cross_asym_T = s_T(a, S_b) − s_T(b, S_a)` across cell types by the **median**. But most
cytokines appear to interact through only a **few** cell types. If a pair couples in fewer
than half the cell types, the median sits on a non-responding cell type and reads about 0,
so on the face of it the statistic should be uninformative for exactly the
cell-type-restricted cascades we might most want to find.

**Why challenge it.** In practice the method works (direction accuracy 88/86/83% across
Oesinghaus/Sheu/ID), which is in tension with the above. Either (a) the signal is broader
than "few cell types" suggests, or (b) the median is silently costing us recall on
narrow-signal pairs. Worth deciding whether median is the right choice or whether a less
conservative aggregation would recover cell-type-restricted cascades.

**Current understanding (2026-07).**
- Implementation (`cascadir/src/cascadir/cross_asym.py`): `median` of per-cell-type
  `cross_asym_T`, plus a magnitude threshold and a sign-consensus. Narrow or inconsistent
  pairs fall below the magnitude threshold and are classified **AMBIGUOUS** (abstain), not
  miscalled. So the median makes the method *conservative*, not wrong.
- The **relay** (where the downstream cytokine is produced) is narrow, but the downstream
  **signature** (the response program to that cytokine) is broad, since a cytokine acts on
  many cell types. `cross_asym_T` reads signature presence, so genuinely-coupled pairs tend
  to have consistent signal across many cell types, which the median then captures.
- **Open:** does the median cost recall on truly cell-type-restricted cascades? Candidate
  alternatives to test against the same audited labels: a signed/trimmed statistic, or a
  "fraction of cell types with a large consistent asymmetry" score.

**Status:** open — method validated empirically; the median's conservatism-vs-recall
trade-off is not yet quantified.

---

## 2. Should the direction call use the donor as the statistical unit?

**Concern.** The coupling gate treats the donor as the unit of independence (a sign test
across donors), but the direction statistic `cross_asym` is computed on cells **pooled
across all donors**, and its random-gene-set null (`null_p`) is **cell-level**. This does not
follow the project's own principle (CLAUDE.md §5/§15/§16): "aggregate to donor level before
any statistical comparison; effective N = donors, not cells."

**Why it matters.** CLAUDE.md §27.6 found the cell-level direction-permutation null
over-powers (thousands of cells per (cytokine, cell-type) makes ~every pair significant;
Storey π₀ ≈ 0.038 is "the null being trivially beatable at large n, not biology") and
concluded "**the null must be donor-level**." So a cell-level significance readout for
direction is a known weakness; a donor-level null/bootstrap is the documented fix, and it is
already implemented for the progression extension (`cascadir.progression.bootstrap_cross_asym`,
donor = unit of independence).

**Why the current thesis claim still holds.** The headline direction accuracy (88/86/83%) is
sign-match against externally known cascades, not an internal significance test, so it does
**not** depend on the cell-level null. The statistical-unit question bites only when we want
per-pair direction significance or an FDR over unknown (Group-U) pairs — which §27.6 lists as
still OPEN.

**Status:** open — the direction point estimate is cell-pooled by design; whether direction
*significance* should move to the donor level (bootstrap/sign test, as in coupling and the
progression extension) is a candidate change, not yet run on the core Oesinghaus/Sheu/ID
direction benchmark. No new experiment planned for now; thesis describes the method as-is.

