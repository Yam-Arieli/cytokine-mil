# Vaccine T-cell cascade — direction results (§32)

**Headline.** On the SARS-CoV-2 vaccination whole-PBMC CITE-seq atlas (Zenodo 7555405;
day 0/2/11/28; 6 donors; 61,911 T cells), `cross_asym` **recovers the vaccination-response
*timeline* (D2→D11→D28) perfectly from a single snapshot, but *reverses* the T-cell
maturation *state* order (Naive→Effector→Memory).** The reversal is not noise or a bug — it
is a **mechanistically-understood boundary of the method**, and the apparatus gate + the
symmetric control pin down exactly why.

This is **direction VALIDATION, not discovery** (the orders are textbook). Method basis:
CLAUDE.md §26/§28/§30. Pre-registration: `reports/vaccine_progression/PRE_REGISTRATION.md`.

---

## 1. Apparatus gate (synthetic, independent) — PASS

| scenario | cross_accuracy | Kendall τ |
|---|---:|---:|
| **distinct-program** (hard gate) | **1.00** | **+1.0** |
| monotone-intensity, no seed | 0.00 | −1.0 |
| monotone-intensity, seeded | 0.67 | +0.33 |

The method **works** on a clean distinct-program ladder (P1 hard gate passed) and **inverts**
on a pure monotone-intensity ladder with no distinct seed. This is the lens for the two real
framings below.

## 2. TIMEPOINT framing (corroboration) — direction RECOVERED (verdict AMBER)

condition = day {D2, D11, D28}; control = D0; oracle D2<D11<D28.

| metric | value | pre-reg GREEN |
|---|---:|---|
| cross_asym accuracy | **100%** (2/2 scored) | ≥ 80% |
| symmetric `directional_score` control | **0%** | ≤ 60% (≪ cross) ✓ |
| Kendall τ (recovered vs true) | **+1.0** | ≥ 0.60 |
| donor-bootstrap accuracy 95% CI | [0.33, 1.0] | lower > 0.50 ✗ |

Recovered order **D2 → D11 → D28** = truth. `cross_asym` ≫ symmetric control (100% vs 0%),
so the direction is real beyond magnitude. **AMBER only because** the 6-donor bootstrap CI is
wide (n=6 donors, 2 scored pairs → underpowered), not because direction failed. → *The
vaccination-response timeline direction is recoverable from one cross-sectional snapshot*, the
disease-progression result (§30) reproduced on a vaccination time axis.

## 3. STATE framing (headline) — direction REVERSED (verdict RED)

condition = maturation state {Naive, Effector, Memory} (CITE protein gating, CD45RO/CD27,
RNA-independent); control = `Resting` (day-0 T cells); oracle Naive<Effector<Memory.

| metric | value |
|---|---:|
| cross_asym accuracy | **0%** (0/2 scored) |
| symmetric `directional_score` control | **100%** |
| Kendall τ | **−1.0** |
| recovered order | **Effector → Memory → Naive** (truth: Naive → Effector → Memory) |

The direction is **exactly inverted**, and the inversion is a **robust, consistent seed**, not
noise: `cross_asym(Memory, Naive)` is positive in **all three** lineages (CD4 T +0.14,
CD8 T +0.23, other T +0.06); same for `cross_asym(Effector, Naive)`.

### Why it reverses (the mechanism)

`cross_asym(a,b) = [s(a,S_b) − ctrl_{S_b}] − [s(b,S_a) − ctrl_{S_a}]`. The control here is
`Resting` ≈ **Naive** (day-0 T cells are naive-like). So:

- `s(Naive, S_Effector) − Resting` and `s(Naive, S_Memory) − Resting` are **NEGATIVE** in the
  data (e.g. CD8: −0.19 and −0.17): naive cells engage the activation/differentiation programs
  *below* the resting baseline — Naive is an even "cleaner" baseline than Resting.
- Effector/Memory cells, being transcriptionally active, carry **residual naive-program
  expression above baseline** (`s(Memory, S_Naive) − Resting > 0`).

⇒ the differentiated state "carries the other's signature" more than Naive does, so the
antisymmetric statistic calls the **differentiated** state upstream. The tell-tale signature of
this regime is **symmetric control = 100% while cross_asym = 0%** — the inverse of the healthy
cytokine/COVID pattern (cross ≫ control), and exactly the apparatus **monotone-noseed** regime
(cross 0.0). The "upstream carries the autocrine downstream program" biology that powers
`cross_asym` is a *cytokine-cascade* mechanism; it does **not** hold for a differentiation axis
whose upstream root (Naive) coincides with the resting control.

## 4. The boundary lesson

> `cross_asym` direction transfers to a **temporal / disease progression** axis (timepoint
> here, severity in §30) but **inverts** on a **cell-state differentiation** axis whose
> upstream state coincides with the control baseline (Naive ≈ Resting). Coupling/existence and
> the timeline are fine; the *state* direction sign is dominated by the baseline-coincidence
> (magnitude / no-seed) regime that the apparatus monotone-noseed scenario predicts.

This is a precise, useful negative: it maps where the method's core assumption applies. A
differentiation cascade is **not** the cytokine-autocrine setting, so the earlier intuition
("differentiation is the cleanest next target") is only half right — it is clean for the
*timeline* but not for the *naive-rooted state* direction.

## 5. Honest caveats

- **Validation, not discovery** (orders are textbook); cross-sectional (direction, not
  forecast); **early memory only** (day-28 mRNA-vaccine, not the full central-memory arc).
- **~6 donors** → the donor-bootstrap is underpowered (this is why the otherwise-perfect
  timepoint run is AMBER, not GREEN). Effective N is donors, not the 61,911 cells.
- The state-axis reversal is **mechanistic, not stochastic** — re-running with more donors
  will not "fix" it; it is the method meeting a genuine boundary.
- Direction ≠ causation; PBMC blood only; single dataset.

## Figures

- state: `results/vaccine_progression/state/plots/` — `direction_accuracy_bar.pdf`
  (cross 0% vs symmetric 100% — the reversal signature), `state_order_recovery.pdf` (τ=−1),
  `per_celltype_sign_consensus.pdf` (reversal consistent across CD4/CD8).
- timepoint: `results/vaccine_progression/timepoint/plots/` —
  `direction_accuracy_bar.pdf` (100% vs 0%), `timepoint_order_recovery.pdf` (τ=+1).
- apparatus: `results/vaccine_progression/apparatus/apparatus_accuracy.pdf`.

Auto-generated per-run docs: `VACCINE_PROGRESSION_RESULTS_timepoint.md` (timepoint). This file
is the synthesized headline writeup (state + timepoint + apparatus + mechanism).
