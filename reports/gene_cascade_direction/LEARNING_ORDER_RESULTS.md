# Gradient Learning-Order — Results (Sheu 3hr) → AMBER/RED

**Verdict: the hypothesis "cascade-SOURCE genes are learned before DOWNSTREAM genes" does NOT survive
the confound controls.** Jobs: train 30830386 (3 seeds, COMPLETED) → analyze 30830578 (COMPLETED).
Pre-registration: `LEARNING_ORDER_PREREGISTRATION.md`. Apparatus validated on synthetic first
(controls proven to separate real cascade-order from SNR artifact).

## Setup actually run
Per-stimulus binary AB-MIL (PIC vs PBS, LPS vs PBS) on Sheu 3hr, 60 epochs, per-epoch checkpoints;
per-gene raw-gradient attribution at every epoch → emergence epoch; effect size = log2FC vs PBS;
real-time emergence from raw Sheu time course (0.25–8hr). 500-gene panel; source = IRF3_direct,
downstream = IFNAR_induced (Ifit3 dropped — overlap).

## Result (3 seeds 42/123/7, both stimuli)
- **H1 (source-first, effect-size-MATCHED permutation):** FAILS.
  - LPS p_matched = 0.059 / 1.000 / 1.000 ; PIC p_matched = 0.670 / 0.675 / 0.778.
  - Not significant and not seed-stable. At matched effect size, source genes are NOT learned before
    downstream genes.
- **H2 (partial Spearman training-emergence vs real-time, controlling effect size):** FAILS.
  - LPS partial = +0.065 / −0.014 / +0.001 (p ≥ 0.17) ; PIC partial = +0.038 / −0.074 / +0.038 (p ≥ 0.18).
  - Learning order does NOT track real biological timing beyond effect size.

## Interpretation
This is the pre-registered **AMBER** outcome: any raw "source-learned-first" appearance is explained by
**learnability/SNR** (higher-effect genes are learned first), not by cascade source→downstream structure.
The synthetic self-test proved the controls correctly *pass* a real cascade-order signal and *reject* an
SNR confound — here they reject. **H1 alone (which does not use real-time) rejects the hypothesis**, so the
conclusion is robust even though the real-time emergence metric looks noisy (down-regulated genes, e.g.
Adrb2 log2fc=−3.0, were flagged as "early" — a first-crossing-detection weakness that would only matter
for H2, which also fails).

## Honest status of the gene-cascade line
- Relay (population→cell snapshot, ID): **RED** (tube-level collapse).
- Gradient learning-order (training dynamics, Sheu): **AMBER/RED** (SNR confound, not cascade order).
- Both negative results were obtained with controls/apparatus validated on synthetic data first.
- Standing cytokine results are unaffected (Path A 121 axes; cross_asym 88/86/83%).

## Possible follow-ups (if continuing)
- The raw (uncontrolled) H1 may be significant — but that is exactly the SNR artifact; not a result.
- A cleaner real-time metric (peak-time / monotone-rise on up-regulated genes only) would firm up H2,
  but cannot rescue H1.
- The deeper issue is shared with the relay line: in 3hr BMDM snapshots, source and downstream IFN genes
  are both already expressed and co-learnable; the *order* the model learns them is set by effect size,
  not cascade depth.
