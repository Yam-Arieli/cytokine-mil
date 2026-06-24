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

### Why it reverses (the mechanism) — control-swap decomposition

`cross_asym(a,b) = raw_engagement(a,b) − control_term`, where `raw = s(a,S_b) − s(b,S_a)` and
`control_term = ctrl_{S_b} − ctrl_{S_a}`. We held the signatures FIXED and swapped only the
control (`scripts/analyze_vaccine_state_control_decomp.py`, run on the cluster) to separate
**(a) control-composition** from **(b) retention-biology**:

| control | Eff–Nai | Mem–Nai | recovered order | τ |
|---|---:|---:|---|---:|
| Resting (orig) | +0.107 | +0.140 | Mem>Eff>Nai | −1.0 |
| **ZERO = raw engagement (b)** | **+0.253** | **+0.588** | Mem>Eff>Nai | **−1.0** |
| Balanced (removes naive/mem imbalance → tests a) | +0.121 | +0.146 | Mem>Eff>Nai | −1.0 |
| Naive-as-control | +0.138 | +0.147 | Mem>Eff>Nai | −1.0 |
| Memory-as-control | +0.159 | +0.161 | Mem>Eff>Nai | −1.0 |
| Effector-as-control | +0.122 | +0.077 | Eff>Mem>Nai | −0.33 |

(correct sign for an X–Naive pair is NEGATIVE = Naive upstream; every control gives POSITIVE.)

**The inversion is effect (b), the retention biology — control-independent.** The RAW
cross-engagement with NO control (ZERO) is the *most* inverted (Mem–Nai = +0.588), and the
Resting control *reduces* it (to +0.140), not amplifies it. **No control choice un-inverts the
Naive pairs** (Balanced, Naive-, Memory-as-control all keep Naive last, τ=−1). So my first-pass
attribution to "Naive ≈ Resting baseline" (effect a) was wrong: a naive-like control actually
*mitigates* the inversion; it is not the cause.

The fundamental driver: **differentiation has the *opposite* cross-engagement asymmetry from a
cytokine cascade.** `cross_asym` assumes "upstream *acquires* the downstream program" (a source
cell gains the target's autocrine program). In differentiation, the *mature* cell *retains* the
*progenitor's* program — memory T cells re-express the naive program (IL7R/TCF7/SELL/CCR7), so
`s(Memory, S_Naive) ≫ s(Naive, S_Memory)` raw — pointing mature→naive. The statistic faithfully
reads that retention asymmetry and so calls the differentiated state "upstream." The tell-tale
is **symmetric control = 100% while cross_asym = 0%** (inverse of the cytokine/COVID pattern),
the apparatus **monotone-noseed** fingerprint.

## 4. The boundary lesson

> `cross_asym` direction transfers to a **temporal / disease progression** axis (timepoint
> here, severity in §30) but **inverts** on a **cell-state differentiation** axis — and the
> inversion is **control-independent** (the decomposition above: even raw cross-engagement with
> no control is fully inverted). The cause is that differentiation has the *opposite*
> cross-engagement asymmetry from a cytokine cascade: `cross_asym` assumes the upstream cell
> *acquires* the downstream program, but in differentiation the **mature cell *retains* the
> progenitor's program** (memory re-expresses the naive program), so the statistic points
> mature→naive.

This is a precise, useful negative that maps where the method's core assumption holds. A
differentiation cascade is **not** the cytokine-autocrine setting — and no control fix repairs
it (a different control just shuffles the Effector/Memory tie; Naive stays last). So the
earlier intuition ("differentiation is the cleanest next target") is only half right: it is
clean for the **timeline** (the right axis for an acquisition-based direction statistic) but
fundamentally mis-signed for the **naive-rooted state** direction. Getting *state* direction
would need a statistic keyed to program *retention/loss*, not acquisition.

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
