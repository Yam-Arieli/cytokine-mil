# M8 — Experiments & results

## 1. How accuracy is computed
`scripts/retally_pipeline_against_audit.py --metric cross_asym`: for each benchmark axis,
aggregate `cross_asym` (median across cell types) → take its **sign**, compare to the
`expected_sign` ground truth (audited Oesinghaus CSV / hand-labeled Sheu CSV). Accuracy =
fraction of benchmark axes whose sign matches (`a_to_b` → expect `+`, `b_to_a` → expect `−`).

## 2. Oesinghaus 24 h (human PBMC) — `reports/cascade_pairs/oes_crossasym_audited.md`
- **cross_asym: 15/17 = 88%** on the audited directional benchmark
  (DIRECTIONAL_a_to_b 9/10 = 90%; DIRECTIONAL_b_to_a 6/7 = 86%).
- **`directional_score`: 8/17 = 47% ≈ chance** — *same data, same signatures*. The symmetric
  metric is direction-blind (M7). This contrast is the headline experimental proof.
- **34/53 axes beat the random-gene-set null** (p < 0.05): discovered `S_X` carry
  cytokine-specific *direction* information, not just activation level.
- **Benchmark label-permutation null: p = 0.003.**
- The **2 misses both involve VEGF** (IL-6/VEGF, IL-13/VEGF) — a VEGF-signature weakness (M9).

## 3. Sheu single-frame, 1/3/5 h (mouse BMDM) — **no cross-time leakage**
`reports/sheu_cascade/timepoint_comparison.md`. Each time point fully self-contained (only the
0 h Unstim baseline is shared, and that is safe — M7 §5).

| time | IFN_MUST | IFN_SHOULD | NFKB_SHOULD | all directional |
|---|---|---|---|---|
| 1 h | 1/2 | — | 3/3 | 4/5 |
| 3 h | 1/2 | 1/1 | 3/4 | 5/7 |
| **5 h** | 1/2 | 1/1 | **4/4** | **6/7 = 86%** |

- **NF-κB → TNF cascades recover well (4/4 at 5 h)** — the *opposite* of the original
  §24-with-curated-pathways result (which failed on NF-κB/TNFR pathway overlap). Binary-IG
  self-signatures separate TNF from the TLR ligands cleanly enough.
- **polyIC → IFNb fails at every time point** (ISG-domination, M5). LPS → IFNb works at all
  times (LPS carries a broader non-ISG program → `S_LPS ≠ S_IFNb`).

## 4. Honest limitations
- **Direction, not existence.** Negative pairs also have large `|cross_asym|` (≈0.11–0.24).
  By design: Path A (M6) says *whether* a pair is coupled; cross_asym only assigns *direction*
  to coupled pairs. Magnitude is **not** a coupling gate.
- **Small n.** 17 (Oes) and 7 (Sheu) directional axes. The Sheu IFN-MUST class is effectively
  1 clean win (LPS→IFNb) + 1 mechanistic failure (polyIC→IFNb).
- **88% is vs a hand-audit** of noisy keyword-parsed literature labels (conservative, but a
  judgment call — `reports/cascade_pairs/audit_log.md`).
- **The polyIC wart** (M5/M9) is a real, understood failure, not noise.

## 5. Two-dataset takeaway
88% (human PBMC, 24 h) **and** 86% (mouse BMDM, 5 h) — different species, system, gene panel,
and time regime. cross_asym working on both is evidence it is a property of the **method**, not
an artifact of one dataset. The honest framing: a methodology demonstration on *known* cascades
with small n — strong signal, modest sample size.
