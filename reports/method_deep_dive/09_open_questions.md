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
   pre-registered labeled pairs, decoupled from Path A recall). *(Attempted 2026-06 as the §27
   Group-U run: ran end-to-end over all 121 axes, but the direction null was over-powered at
   the cell level and the re-trained signatures regressed (P4) — see #7, `GROUP_U_RESULTS.md`.)*
4. **`top_n` sensitivity.** cross_asym uses top-50; sweep {20, 50, 100, 200} to characterise
   robustness.
5. **ID Path A geometry rerun.** The geo job emitted nothing this run; rerun
   `run_latent_geometry.py` on the 3 trained ID stage12 seeds to get coupling axes (and an
   ID Path-A ↔ Path-B comparison).
6. **IL-1β→IL-6 on ID (the ID wart).** A near-zero non-call while the *parallel* TNF→IL-6
   recovered cleanly; `S_IL1b` is NF-κB-dominated. Same family as the polyIC fix: re-derive
   `S_IL1b` to carry the IL-6/STAT3 program (or test whether 4 h in vivo is too early for the
   autocrine IL-6 loop in IL-1β-stimulated cells).
7. **Donor-level nulls — the headline fix (blocks both coupling AND direction).** The 2026-06
   Group-U FDR (§27) and the Oes signature-coupling gate (§28) both over-call because their
   null/gate is computed at the **cell** level (thousands of cells → ~everything significant,
   `dir_p_emp=0.000`, 894/1128 "coupled"). Recompute at the **donor** level (aggregate per
   donor, then permute / signed-rank across ≈10 donors). This is the prerequisite for *any*
   valid Group-U discovery claim and for a usable signature-coupling gate.
8. **Degree/hub-corrected coupling gate.** Oes signature coupling is hub-dominated (IL-15 in
   11/20 top pairs, CD40L in 5/20) — z-score `M` per cytokine or subtract row+column (hub)
   effects before the null, so coupling reflects *specific* engagement not signature magnitude.
9. **Reproduce the §26 signatures (P4 regression).** The Group-U re-run trained a separate
   encoder per 8-way chunk and scored 6/11 labeled (vs 15/17); retrain the affected cytokines
   with one shared encoder (like missing16) and re-check P4.
10. **Run signature coupling on ID + synthesize the two paths.** Coupling has never been run on
    ID (latent-geometry Path A emitted nothing there either). And Path 1 (latent) vs Path 2
    (signature) coupling are complementary (M6 §7) — combine signature specificity with
    donor-level discipline.

## Where we are (honest status)
The `cross_asym` direction method works single-frame, no-leakage, on three datasets (88% / 86% / 83%).
**Direction-not-existence is by design** (Path A handles existence). It is a methodology
demonstration on *known* cascades with small n; the one consistent failure (polyIC→IFNb) is
mechanistically understood and has a predicted fix. Path A (coupling, 121 axes on Oesinghaus)
is the independent, publication-grade standing result.

**Update (2026-06).** A signature-space coupling reframe (M6 §7, `SIGNATURE_COUPLING_RESULTS.md`)
recovered the Sheu cascades latent geometry missed (**2/2**) — a real win that confirms the
"shared-activation confound" diagnosis. But both that gate (Oes) and the first end-to-end
Group-U direction-FDR over-call at the **cell** level, so the Group-U discovery claim is **not
yet supported**. The open headline is donor-level nulls (#7) — the same statistical discipline
(§16) the standing results already respect.

## Cross-references
- Audience-facing talk report: `reports/group_talk_2026-06/cascade_direction_report.{tex,pdf}`.
- Session narrative: `reports/SESSION_SUMMARY_2026-05-31.md`.
- Spec: `CLAUDE.md` §20 (Path A), §23–§24 (pathway/asymmetry lineage of Path B).
