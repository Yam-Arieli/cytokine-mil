# Session summary — 2026-05-31

**One line:** the binary-IG → Path B cascade-direction chain *works* once you use
the right aggregation. The §24 `directional_score` is direction-blind for
self-signatures; the antisymmetric **`cross_asym`** is the fix. It scores **88%
on Oesinghaus 24h** and **86% on Sheu 5h** against clean labels.

---

## Arc of the session

1. Re-examined the full-19 Oesinghaus pipeline result (was reading 47% ≈ chance,
   "structural positive bias").
2. **Strict literature audit** of the Oesinghaus labels — the `cytokine_axes.csv`
   `literature_direction` tags are produced by a keyword parser and are noisy.
   Built a per-direction audit from receptor biology (17-axis benchmark, 7/29
   tags flipped). See `reports/cascade_pairs/audit_log.md`.
3. **Found the real bug** while wiring up Sheu: `directional_score` is algebraically
   symmetric → its *sign cannot encode direction* when P_A/P_B are the cytokines'
   own discovered signatures.
4. **Fix = `cross_asym`** (the user's own "X-tube carries both Sx and Sy; Y-tube
   only Sy" model). Validated on existing data (no retraining): 47% → 88%.
5. Built + ran the **Sheu single-frame multi-time-point experiment** (1/3/5h, no
   cross-time leakage) as a clean-label confirmation on a second dataset.

---

## The metric fix (the core contribution)

For an unordered pair with discovered signatures S_a, S_b:

```
directional_score = (s_a·S_a + s_b·S_b) − (s_a·S_b + s_b·S_a)      # SYMMETRIC in (a,b)
cross_asym        =  s(a, S_b) − s(b, S_a)   (PBS-normalised)       # ANTISYMMETRIC
```

`directional_score` is invariant under swapping a↔b → it measures *coupling
distinctness*, not direction. `cross_asym` flips sign on swap → positive means
a engages b's program more than vice versa ⇒ **a is upstream (a_to_b)**.

Implemented as the **primary** call in `scripts/run_pipeline_a_bridge_b.py`
(directional_score kept as a secondary coupling reference);
`retally_pipeline_against_audit.py --metric cross_asym` is the default.

---

## Result 1 — Oesinghaus 24h PBMC (cluster-confirmed + null-validated)

`reports/cascade_pairs/oes_crossasym_audited.md`

- **cross_asym: 15/17 = 88%** on the strict audited benchmark
  (DIRECTIONAL_a_to_b 9/10 = 90%; DIRECTIONAL_b_to_a 6/7 = 86%)
- `directional_score`: 8/17 = 47% (chance) — same data, same signatures.
- **34 axes beat the random-gene-set null** (p<0.05): the discovered S_X carry
  cytokine-specific direction information, not just activation level.
- 2 misses both involve **VEGF** (IL-6/VEGF, IL-13/VEGF) — a VEGF-signature weakness.
- Label-permutation null on the benchmark: p = 0.003.

The "24h saturates the cascade" diagnosis from the prior session was **secondary** —
the dominant problem was the metric, not the time regime.

---

## Result 2 — Sheu BMDM single-frame, 3 time points (clean TLR-biology labels)

`reports/sheu_cascade/timepoint_comparison.md`. Each time point fully
self-contained — **no cross-time data used in the method** (the user's hard
constraint); 0h Unstim is the only shared reference (resting baseline, and
cross_asym only PBS-normalises with it).

| time | IFN MUST | IFN SHOULD | NF-κB SHOULD | all directional |
|---|---|---|---|---|
| 1hr | 1/2 | — | 3/3 | 4/5 |
| 3hr | 1/2 | 1/1 | 3/4 | 5/7 |
| **5hr** | 1/2 | 1/1 | **4/4** | **6/7 = 86%** |

Two findings:

- **NF-κB → TNF cascades recover well (4/4 at 5h)** — the *opposite* of the original
  §24-with-curated-pathways result (where NF-κB failed on pathway overlap).
  Binary-IG self-signatures separate TNF from the TLR ligands cleanly enough.
- **polyIC → IFNb fails at every time point** (cross_asym +0.11→+0.16, expected −).
  Mechanistic reason: polyIC's binary-IG signature is **ISG-dominated** (the
  strongest DE genes are the autocrine IFN response, not the lower-expressed
  IRF3-direct genes Ifnb1/Cxcl10), so S_polyIC ≈ S_IFNb and IFNb (direct IFNAR)
  out-engages polyIC (autocrine) → sign inverts. LPS → IFNb works at all times
  (LPS carries a broader non-ISG program).

GPU training (3 seeds × 3 time-point Stage1+2 + 7 binary models × 3) all completed.
The only failure was a trivial slurm-var bug (`$RES` not propagating into the
bridge job → empty `--targets`), fixed and re-run.

---

## Honest caveats

- **cross_asym gives direction, not cascade existence.** Negative (no-cascade)
  Sheu pairs also have large |cross_asym| (0.11–0.24). Fine by design: Path A
  (latent geometry) identifies *which* pairs are coupled; cross_asym only
  assigns direction to those. Magnitude is not a coupling gate.
- 88% / 86% are on *directional benchmarks of known cascades*, ~17 and 7 axes.
  Small n; the Sheu IFN-MUST class is effectively 1 clean win (LPS→IFNb) + 1
  mechanistic failure (polyIC→IFNb).

---

## Artifacts

Local:
- `reports/cascade_pairs/oes_crossasym_audited.md` — Oesinghaus 88%
- `reports/sheu_cascade/timepoint_comparison.md` — Sheu cross-time bottom line
- `reports/sheu_cascade/sheu_cascade_labels.yaml` + `sheu_axes_labeled.csv` — crystal-clear labels
- `reports/cascade_pairs/audit_log.md` + `cytokine_axes_audited.csv` — Oesinghaus audit
- `results/sheu_cascade/{1,3,5}hr/pathB/per_celltype.csv` — per-cell-type cross_asym

Code (committed, pushed):
- `scripts/run_pipeline_a_bridge_b.py` (+ `_sheu` variant) — cross_asym primary
- `scripts/retally_pipeline_against_audit.py` — `--metric cross_asym`
- `scripts/train_sheu2024_binary.py`, `finalize_sheu_labels.py`, `compare_timepoints.py`,
  `eval_sheu_axis_gate.py`, audit scripts
- `slurm/sheu_cascade/` — the dependency-chained DAG

---

## Open questions / next steps

1. **polyIC ISG-domination test**: re-derive S_polyIC excluding ISGs (or up-weight
   IRF3-direct genes); predict polyIC→IFNb sign flips to correct. Cheap, isolates
   the mechanism.
2. **VEGF signature**: the 2 Oesinghaus misses are both VEGF — inspect S_VEGF.
3. **Path A coupling + Path B direction end-to-end** on Sheu (the §21 gate ran but
   was secondary here).
4. **top_n sensitivity** of cross_asym (currently 50).
