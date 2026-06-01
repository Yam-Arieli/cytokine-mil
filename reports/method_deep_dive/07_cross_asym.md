# M7 — Path B: the `cross_asym` crux (the core discovery)

Real code: per-cell-type scores in `cytokine_mil/analysis/pathway_audit.py::directional_asymmetry_test`
(lines 457–491); `cross_asym` derived in `scripts/run_pipeline_a_bridge_b{,_sheu}.py`; eval in
`scripts/retally_pipeline_against_audit.py`. In the pipeline the "pathways" are the **discovered
signatures** `S_X` from M5, with `P_A = S_a`, `P_B = S_b` (each stimulus scored on its own signature).

## 1. The signature scores (per pair `(a,b)`, per cell type T)
Mean signature-gene expression:
`s(a,S_a), s(a,S_b), s(b,S_a), s(b,S_b)` and PBS rows `s(PBS,S_a), s(PBS,S_b)`.
**PBS-normalise** each (subtract the matching PBS score) → `sa_Sa_norm, sa_Sb_norm, sb_Sa_norm,
sb_Sb_norm` (code names `sA_PA_norm, sA_PB_norm, sB_PA_norm, sB_PB_norm`, lines 464–467).

## 2. The two candidate metrics
- `directional_score = asym_PA − asym_PB`, where
  `asym_PA = sa_Sa_norm − sb_Sa_norm`, `asym_PB = sa_Sb_norm − sb_Sb_norm` (lines 468–473).
- `cross_asym = sa_Sb_norm − sb_Sa_norm = [s(a,S_b) − s(PBS,S_b)] − [s(b,S_a) − s(PBS,S_a)]`.

## 3. The crux — symmetry vs antisymmetry under swapping `a ↔ b`
Within each `asym`, the PBS term cancels, so:
```
directional_score(a,b) = s(a,S_a) − s(b,S_a) − s(a,S_b) + s(b,S_b)
```
Swap `a↔b` (so `S_a↔S_b`):
```
directional_score(b,a) = s(b,S_b) − s(a,S_b) − s(b,S_a) + s(a,S_a)  ==  directional_score(a,b)
```
**SYMMETRIC** — identical value. Its sign does **not** depend on which stimulus we label `a`, so
it cannot encode direction; it measures *coupling distinctness*. → this is why it scored
**47% ≈ chance** on direction (M8).

```
cross_asym(a,b) = s(a,S_b) − s(b,S_a) − s(PBS,S_b) + s(PBS,S_a)
```
Swap `a↔b`:
```
cross_asym(b,a) = s(b,S_a) − s(a,S_b) − s(PBS,S_a) + s(PBS,S_b)  ==  − cross_asym(a,b)
```
**ANTISYMMETRIC** — flips sign on swap. So its **sign encodes direction**: `+` ⇒ `a` engages
`b`'s signature more than `b` engages `a`'s ⇒ **`a` upstream** (`a_to_b`); `−` ⇒ `b` upstream.

## 4. Why antisymmetry is the right shape (the biology, M1)
The **asymmetric fingerprint**: an upstream stimulus's cells carry **both** programs (their own
+ the downstream one they induce via autocrine); the downstream ligand's cells carry **mainly
their own**. So `s(upstream, S_downstream) > s(downstream, S_upstream)` ⇒ `cross_asym > 0` when
`a` is upstream. cross_asym is literally *"does `a` carry `b`'s program more than `b` carries
`a`'s?"* — the "X-tube has `Sx+Sy`, Y-tube has only `Sy`" intuition.

## 5. PBS handling (precise — don't over-claim)
cross_asym is **PBS-normalised** (each score baseline-subtracted) but **not PBS-vanishing**:
the terms `s(PBS,S_b)` and `s(PBS,S_a)` remain (they are on *different* signatures). The
antisymmetry holds **including** them (they flip sign on swap), so a shared resting baseline
**cannot leak direction** — it is a per-signature constant offset. (Contrast `directional_score`,
where the PBS term cancels entirely *within* each `asym`.)

## 6. Aggregation — per-cell-type → one call per pair
Each cell type yields one `cross_asym`. The pair's call = **median across cell types** + a
**sign-consensus** (fraction of cell types agreeing with the median sign) → robust to one noisy
cell type. (`run_pipeline_a_bridge_b.py`: `_cross_asym_series`, `_aggregate_metric`.)

## 7. Null control
**Random gene-set null:** replace `S_a, S_b` with random same-size gene sets drawn from disjoint
HVGs, recompute cross_asym; the real signatures must beat this null. On Oesinghaus, **34/53 axes
beat it at p < 0.05** (M8) — the discovered signatures carry cytokine-specific *direction*
information, not merely activation level.

## 8. Worked numbers (hands-on)
*(Filled in the interactive M7 walkthrough: open `results/sheu_cascade/5hr/pathB/per_celltype.csv`,
take the `LPS/TNF` axis, pull `sA_PB_norm` & `sB_PA_norm`, compute `cross_asym` by hand, confirm
sign `= +` vs `expected_sign = +1`; then recompute `directional_score` on the same row to feel
it is magnitude-only; then `IFNb/PIC` to see the polyIC sign flip from M5.)*
