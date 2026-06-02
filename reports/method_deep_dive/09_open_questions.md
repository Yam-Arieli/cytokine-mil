# M9 — Consolidation & open questions

## The whole pipeline in one breath
**Raw cells** →(**M3** normalize → log1p → HVG(Oes) → stratified, shuffled, variable-size
**bags**)→ **pseudo-tubes** →(**M4** AB-MIL = encoder → attention → linear classifier; Stage 1
cell-type pretrain, Stage 2 frozen-encoder MIL)→ **trained models**, which split into two
branches:
- **Path A (M6):** the **multiclass** model → latent-geometry centroids + PBS-RC + Wilcoxon/FDR
  → **coupling axes** (*which* pairs are linked; direction-blind). → `cytokine_axes.csv`.
- **Bridge (M5):** per-stimulus **binary** models → **Integrated Gradients** (PBS baseline,
  20-step midpoint) → **signatures `S_X`** (top-50) → **Path B (M7):**
  `cross_asym = s(a,S_b) − s(b,S_a)`, **antisymmetric** → **direction** (who is upstream).
- **Eval (M8):** sign(median cross_asym) vs `expected_sign` → **88% Oes / 86% Sheu-5h / 83% ID**;
  `directional_score` (symmetric) = 47% / – / 33% on the same data.

One-line thesis: **`directional_score` is symmetric (direction-blind); the antisymmetric
`cross_asym` recovers cascade direction from a single snapshot — 88%/86%/83% on three datasets.**

## Open threads / next experiments
1. **polyIC ISG-exclusion (cheap, high-value).** Re-derive `S_polyIC` excluding ISGs (or
   up-weight IRF3-direct genes); predicted to flip polyIC→IFNb to the correct sign → a clean
   "we understand our one failure" result. (Mechanism: M5 §5.)
2. **VEGF signature.** The 2 Oesinghaus misses are both VEGF — inspect `S_VEGF` (why is it weak?).
3. **End-to-end Path A → Path B.** Gate the direction call on Path A's coupled pairs so
   coupling (existence) + direction are reported together (currently Path B is evaluated on
   pre-registered labeled pairs, decoupled from Path A recall).
4. **`top_n` sensitivity.** cross_asym uses top-50; sweep {20, 50, 100, 200} to characterise
   robustness.
5. **ID Path A geometry rerun.** The geo job emitted nothing this run; rerun
   `run_latent_geometry.py` on the 3 trained ID stage12 seeds to get coupling axes (and an
   ID Path-A ↔ Path-B comparison).
6. **IL-1β→IL-6 on ID (the ID wart).** A near-zero non-call while the *parallel* TNF→IL-6
   recovered cleanly; `S_IL1b` is NF-κB-dominated. Same family as the polyIC fix: re-derive
   `S_IL1b` to carry the IL-6/STAT3 program (or test whether 4 h in vivo is too early for the
   autocrine IL-6 loop in IL-1β-stimulated cells).

## Where we are (honest status)
The `cross_asym` direction method works single-frame, no-leakage, on three datasets (88% / 86% / 83%).
**Direction-not-existence is by design** (Path A handles existence). It is a methodology
demonstration on *known* cascades with small n; the one consistent failure (polyIC→IFNb) is
mechanistically understood and has a predicted fix. Path A (coupling, 121 axes on Oesinghaus)
is the independent, publication-grade standing result.

## Cross-references
- Audience-facing talk report: `reports/group_talk_2026-06/cascade_direction_report.{tex,pdf}`.
- Session narrative: `reports/SESSION_SUMMARY_2026-05-31.md`.
- Spec: `CLAUDE.md` §20 (Path A), §23–§24 (pathway/asymmetry lineage of Path B).
