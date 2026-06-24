# §32 confirmatory — STATE-axis inversion: control-composition (a) vs retention-biology (b)

Holds the discovered signatures FIXED (state-vs-Resting binary IG) and varies ONLY the
`cross_asym` control, so `cross_asym = raw_engagement − control_term` is decomposed: ZERO
control = raw engagement = **effect (b)**; the control term = **effect (a)**. True order
Naive<Effector<Memory ⇒ for the X–Naive pairs the CORRECT sign is **negative** (Naive
upstream); the observed run had them positive (inverted), τ=−1.
(Generated on the cluster by `scripts/analyze_vaccine_state_control_decomp.py` from the saved
`fit_state` artifacts — no GPU re-fit.)

## Verdict

**Effect (b) RETENTION-BIOLOGY is the DOMINANT, sufficient cause:** even with NO control (raw
cross-engagement) the direction is fully inverted (τ_ZERO=−1.00), in fact *more* inverted than
with the day-0 control. The control (effect a) keeps the same sign (τ_Resting=−1.00) but
**REDUCES** the magnitude (its control term is subtracted) — it **mitigates, it does not
cause**. No control choice un-inverts the Naive pairs (Balanced τ=−1.00).

## cross_asym per control (correct sign for X–Naive is NEGATIVE; τ: +1 correct, −1 inverted)

| control | Effector–Naive | Memory–Naive | Effector–Memory | recovered order | τ |
|---|---:|---:|---:|---|---:|
| Resting (orig) | +0.107 | +0.140 | −0.050 | Memory > Effector > Naive | −1.00 |
| ZERO = raw engagement (effect b) | +0.253 | +0.588 | −0.353 | Memory > Effector > Naive | −1.00 |
| Balanced (removes naive/mem imbalance; tests effect a) | +0.121 | +0.146 | −0.055 | Memory > Effector > Naive | −1.00 |
| Naive (as control) | +0.138 | +0.147 | −0.073 | Memory > Effector > Naive | −1.00 |
| Memory (as control) | +0.159 | +0.161 | −0.010 | Memory > Effector > Naive | −1.00 |
| Effector (as control) | +0.122 | +0.077 | +0.006 | Effector > Memory > Naive | −0.33 |

## Effect decomposition (inverted X–Naive pairs)

`cross_asym_Resting = raw_engagement(ZERO) − control_term`

| pair | raw engagement = effect (b) | control term = effect (a) | observed (Resting) |
|---|---:|---:|---:|
| Effector–Naive | +0.253 | +0.146 | +0.107 |
| Memory–Naive | +0.588 | +0.447 | +0.140 |

The control term is **positive** and **subtracted**, so it pulls the (inverted, positive)
`cross_asym` *toward* the correct negative sign — i.e. the control *reduces* the inversion but
nowhere near enough to flip it. Raw effect (b) is what drives the reversal.

## Signature sizes / overlap

- |S_Naive| = 50 · |S_Effector| = 50 · |S_Memory| = 50  (S_Naive is **not** degenerate)
- Jaccard(S_Naive, S_Effector) = 0.52  (heavy overlap → explains the Effector–Memory ambiguity)
- Jaccard(S_Naive, S_Memory) = 0.18
- Jaccard(S_Effector, S_Memory) = 0.18

## Conclusion

The STATE-axis reversal is a **control-independent** property of the differentiation axis:
`cross_asym` reads a program-*acquisition* asymmetry (upstream gains the downstream program),
but differentiation is a program-*retention* asymmetry (the mature cell retains the
progenitor's program — memory re-expresses naive IL7R/TCF7/SELL/CCR7), so `s(Memory, S_Naive) ≫
s(Naive, S_Memory)` and the statistic points mature→naive. The earlier "Naive≈Resting baseline"
attribution is falsified. Method basis: CLAUDE.md §26/§32.
