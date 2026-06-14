# Pre-registration — Time-Resolved Gene Cascades (Sheu time course)

**Locked before the run (per CLAUDE.md §25.1). Commit to `main` first.** Date: 2026-06-14.

## Goal
Characterize real gene cascades for **PIC** and **LPS** from the Sheu time course (0.25→8hr): order genes
by **activation time**, infer **directed gene→gene edges by temporal precedence**, and validate against the
known IFN cascade. Direction comes from real biological time (no SNR-confound issue, unlike the prior
learning-order experiment).

## Ground truth (mouse, pathway_signatures.py)
- **SOURCE / EARLY** (`IRF3_direct`, directly induced via TLR3/4-TRIF): `Ifnb1, Ccl5, Cxcl10, Ifit2, Ifit3`.
  Literature kinetics: peak ~0.5–1hr.
- **DOWNSTREAM / LATE** (`IFNAR_induced` ISGs, induced via autocrine IFN-β): `Mx1, Mx2, Ifit1, Ifit1bl1,
  Ifit3b, Rsad2, Irf7, Oasl1`. Literature kinetics: peak ~3–5hr.
- Ifit3 appears in both lists → dropped from the comparison (ambiguous). Only panel-present genes used.

## Method (locked)
- Per stimulus, per pseudo-donor: PBS-corrected per-gene per-time mean (`_compute_time_series`) → g×t
  trajectory. Quality filter: max above-baseline ≥ floor (induced genes only).
- Activation time = first time crossing 50% of the gene's max above-baseline.
- Directed edge A→B: activation precedence (t_act(A) < t_act(B) − margin) AND lead-lag cross-correlation
  shows A leads B. Stability = sign agreement across the available pseudo-donors.

## Pre-registered validation gates
- **V1 (source-before-downstream):** median activation-time(`IRF3_direct`) < median(`IFNAR_induced`),
  one-sided permutation p ≤ 0.05, in **PIC AND LPS**, consistent across pseudo-donors.
- **V2 (directed edges):** among edges between IRF3-direct and IFNAR-induced genes, the fraction running
  IRF3-direct → IFNAR-induced ≥ 0.8 (i.e. cascade runs source → downstream, not reverse).
- **V3 (absolute kinetics):** IRF3-direct median activation ≤ ~1hr; IFNAR-induced median activation ≥ ~3hr.

## Verdict
- **GREEN:** V1 + V2 hold (donor-stable) → the time-resolved method recovers the real IFN cascade →
  trust it to characterize the full panel / other stimuli.
- **AMBER/RED:** V1 or V2 fails → even real time doesn't resolve the cascade in this 500-gene BMDM panel;
  the bottleneck is the panel/biology (stated honestly).

## Discipline
No tuning of gene lists, frac threshold, margin, or gates after seeing results. A `--synthetic` smoke
(planted early→late two-wave cascade) validates the apparatus first; it is a code check, not the claim.
**Caveat (reported, not a gate):** temporal precedence is necessary-but-not-sufficient for causation
(a common upstream driver can create apparent edges) — this is "cascade order from real kinetics," not
proven causation. Coarse time grid resolves *waves* (early/mid/late), not fine per-gene order.
