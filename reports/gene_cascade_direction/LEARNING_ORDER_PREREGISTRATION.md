# Pre-registration — Gradient Learning-Order (cascade source vs downstream)

**Locked before any real run (per CLAUDE.md §25.1). Commit to `main` first.** Date: 2026-06-14.

## Hypothesis
In a gene cascade, **SOURCE genes are learned earlier in training than DOWNSTREAM (cascade-induced)
genes.** Tested by following the **per-gene gradient** of a per-stimulus binary AB-MIL classifier across
training epochs (Sheu 3hr; polyIC vs PBS and LPS vs PBS), and comparing each gene's **emergence epoch**
(first epoch its |attribution| reaches 50% of its final value) to known source/downstream labels.

## Ground-truth gene sets (mouse, from pathway_signatures.py; Ifit3 dropped — appears in both)
- **SOURCE** (`IRF3_direct`, primary_for polyIC/LPS, directly induced via TLR3/4-TRIF):
  `Ifnb1, Ccl5, Cxcl10, Ifit2`.
- **DOWNSTREAM** (`IFNAR_induced`, cascade_from polyIC/LPS, induced via autocrine IFN-β):
  `Mx1, Mx2, Ifit1, Ifit1bl1, Ifit3b, Rsad2, Irf7, Oasl1`.
- Only genes present in the 500-gene Sheu panel are used (reported at runtime).

## The confound this is designed around
Training-order ≈ **learnability/SNR** (high-effect, clean genes learned first), NOT necessarily causal
source — the toy showed emergence-order ≈ SNR. Per-gene **effect size = log2FC vs PBS**. H1 must survive
**effect-size matching**; H2 (real time) must survive **partialling out effect size**.

## Pre-registered tests / gates (seeds 42/123/7; polyIC AND LPS)
- **H1 (source-first, effect-size-controlled):** median emergence(source) < median(downstream), AND the
  difference clears an **effect-size-matched permutation null** (permute source/downstream labels within
  effect-size quantile bins), one-sided p ≤ 0.05.
- **H2 (decisive — real-time external validation):** **partial Spearman** between training-emergence and
  **real-time emergence** (raw Sheu 0.25/0.5/1/3/5/8hr per-gene first-crossing time), **controlling for
  effect size**, > 0 with p ≤ 0.05.
- **Stability:** sign/direction of H1 and H2 agree across the 3 seeds.

## Verdict
- **GREEN:** H1 (effect-size-controlled) AND H2 hold, seed-stable → training dynamics encode cascade order.
- **AMBER:** H1 holds only *without* the effect-size control, or H2 fails → it is a learnability/SNR
  artifact; report honestly (NOT evidence of cascade order).
- **RED:** no source-before-downstream ordering at all.

## Discipline
No tuning of gene lists, conditions, emergence threshold, or gates after seeing results. The
`--synthetic` smoke (cascade-order vs SNR-confound regimes) validates the apparatus + that the controls
behave correctly BEFORE the real run; it is a code check, not part of the registered claim.
