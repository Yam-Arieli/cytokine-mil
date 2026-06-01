# M6 — Path A: latent geometry (coupling — the standing result)

Real code: `scripts/run_latent_geometry.py` + `cytokine_mil/analysis/latent_geometry.py`.
Spec: CLAUDE.md §20–§20.2. **This is the separate "standing result", not the cross_asym spine.**

## 1. The job (different question from Path B)
Path A answers **which cytokine pairs are *coupled*** (share signaling biology) — *unordered*
axes, **not** direction. It is the project's publication-grade result (121 coupling axes on
Oesinghaus, ~50% literature-supported vs ~1% chance). Path B (cross_asym, M7) then assigns
**direction** to coupled pairs.

## 2. The idea: coupling = directional bias in embedding space
Using the **multiclass** model's frozen encoder (M4): embed all cells, compute per-(cytokine,
cell-type) **centroids** `µ` in embedding space. If cytokine A's cells (in cell type T) are
pulled **toward cytokine B's centroid** (relative to A's own resting state), that's evidence A
and B share a transcriptional program in T → a coupling axis, with T the candidate **relay**.

## 3. PBS-RC + directional bias (the refined readout, §20.1)
- **PBS-RC space:** subtract the per-cell-type PBS centroid from embeddings,
  `h̃ = h − µ_PBS,T` (training donors only) → everything is "deviation from resting".
- **Directional bias** per donor `d`: `b(A→B,T) = (µ_{A,T}^{(d)} − µ_A) · û_{A→B}`, where
  `û_{A→B}` is the unit vector from A's to B's pooled centroid and `µ_A` is A's pooled
  cross-cell-type centroid (subtracted to remove A's generic signal).
- **Significance:** two **independent** one-sided **Wilcoxon signed-rank** tests across donors
  (crucially **no `b_fwd − b_rev` subtraction** — that subtraction was the old symmetric bug),
  **Bonferroni** across cell types, then **BH-FDR** across pairs.

## 4. Why Path A is direction-BLIND (this motivates M7)
The legacy asymmetry score `max_T[bias(A→B) − bias(B→A)]` is **antisymmetric by construction**
— it cannot tell a genuine directional cascade from an algebraic sign flip, and the 2026-05-20
literature check found directional inference at **chance (49/51%)**. So Path A's output is
reframed as **unordered axes** (a coupling call + a relay cell type), *not* cascades. Getting
direction needs a different, antisymmetric-**yet-meaningful** statistic → that is cross_asym (M7).

## 5. Output
`cytokine_axes.csv`: `axis_a, axis_b` (unordered, `a < b`), `axis_strength`,
`relay_T_candidates`, `literature_status`, `literature_direction`, … — one row per coupled pair.

## 6. Relationship to Path B (the division of labour)
- **Path A = existence / coupling** (which pairs are linked).
- **Path B = direction** (who is upstream).
cross_asym gives **direction, not existence** — negatives can still have large `|cross_asym|`;
deciding *whether* a pair is coupled is Path A's job (the honest caveat in M8).
