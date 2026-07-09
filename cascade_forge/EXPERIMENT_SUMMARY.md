# cascade_forge — experiment summary (2026-07-09)

## What we built

A pip-installable package, **`cascade_forge`**, that turns a *user-authored* cascade
graph into a synthetic single-cell snapshot with **known ground-truth direction**, so
`cascadir` (thesis §4.5) can be tested against a truth we control. Until now `cascadir`
was only ever checked against literature-audited labels (17/7/6 pairs on real data);
there was no way to measure its direction accuracy against a known answer.

**Why build, not reuse:** a web scan found no Python-native, pip-installable, maintained
simulator that supports "many cell types + a *user-authored* directional cascade graph
(strength + pseudo-time delay) with a deliberately weak-but-detectable effect." dyngen /
scMultiSim are R-only; SERGIO is gene-TF-granular; scDesign2/3 & GRouNdGAN *fit a
reference* (ground truth is learned, not authored). So it is a small self-contained
simulator reusing the house planting math and real-data-calibrated constants.

## How it works

You author `cascades = {source: {downstream: (strength, pseudo_time_delta)}}` (delta
optional → 1.0; a bare scalar is a strength; downstream-only labels exist with no
outgoing edge). Then:

1. **Pseudo-time propagation** — for each applied label, seed only it (clamped on) and
   step through pseudo-time; each edge `X→Y (w, τ)` ramps `Y`'s activation toward
   `w·activation(X)` with time-constant `τ` (first-order kinetics). Multi-hop cascades
   emerge *in order*; chain strengths multiply at steady state.
2. **Planting** — a dominant cell-type marker signature (~1.2 over 0.30 baseline) plus a
   deliberately **weak** additive per-label program (`effect_size≈0.30`, ≪ the marker
   gap, but detectable at the pseudo-tube/bag level where cascadir's MIL + Integrated
   Gradients operate). An upstream label's cells carry *both* its own and the (autocrine)
   downstream program; the downstream label alone does not carry its upstream — the exact
   asymmetry `cross_asym = s(a,S_b) − s(b,S_a)` reads as direction.
3. **Output** — one `AnnData` per snapshot time (obs `condition`/`donor`/`cell_type`, a
   `"PBS"` control, ground truth in `adata.uns`), meeting the `cascadir` data contract.
   Flags: `responder_mode ∈ {all, receptor}`, `output ∈ {raw, lognorm}`.

## Experiment run (quickstart)

Authored a 6-label cascade (a 2-hop chain AlphaKine→BetaKine→GammaKine, a fan-in
OmegaKine→BetaKine, and an independent edge DeltaKine→EpsilonKine), forged at
`n_cell_types=4, n_cells_per_tube=300, n_donors=6, effect_size=0.30, output="raw"` →
**12,600 cells × 630 genes**, then fit `cascadir` and called `benchmark(direct_edges)`.

## Results

| metric | value |
|---|---|
| `cross_asym` direction accuracy | **100% (4/4 edges)** — all classified STRONG |
| symmetric `directional_score` control | 75% (3/4) — non-discriminative (small-n) |
| unit tests | 23 fast pass (~2.5 s) |
| end-to-end (trains cascadir) | 1 pass (~107 s) |

The 2-hop chain, the fan-in, and the independent edge were all called in the correct
direction. The key contrast holds: the **antisymmetric** `cross_asym` is right while the
**symmetric** control is not the signal (the gap widens with more pairs).

## Large experiment (2026-07-09)

Ran on the HUJI cluster (SLURM DAG: forge CPU → 7-way GPU benchmark array → aggregate),
**10 donors × 20 labels (+PBS) × 5000 cells/tube = 1.05M cells**, 1,410-gene panel, 8 cell
types. Authored graph: deep chain A→B→C→D, chain E→F→G, fan-out H→{I,J,K}, fan-in
{L,M}→N, feedback O↔P, and 4 isolated negative-control labels Q,R,S,T. Seven configs
(responder_mode all/receptor × snapshots {t3,t6} + an effect_size sweep). Plan +
per-config `RESULTS.md`: [`LARGE_EXPERIMENT_PLAN.md`](LARGE_EXPERIMENT_PLAN.md).

**Direction — headline (confound-corrected).** cross_asym recovers direction at **100%**
(90% at the weakest effect_size 0.15), while the symmetric `directional_score` control
sits at **~50% (chance)** — the same contrast the thesis reports on real data (88/86/83%
vs chance), now at 1M-cell scale.

| config | cross_asym acc | symmetric control |
|---|---:|---:|
| effect 0.15 (t6) | 90% | 50% ± 16% |
| effect 0.20 / 0.30 / 0.40 (t6) | 100% | 50% ± 16% |
| all vs receptor (eff 0.30, t3 & t6) | 100% | 50% ± 16% |

*Caveat + fix (important):* the first run scored the control at a **fake 100%** because
the authored graph put every upstream label alphabetically before its downstream, so a
trivial "first = upstream" rule wins. Fixed by **scrambling label names** so sort-order is
decoupled from direction (`forge --scramble_seed`, committed). cross_asym is
naming-invariant; the symmetric control, recomputed over random relabelings
(`scripts/analytical_scramble_correction.py`), collapses to 50% — the values above.

**Effect-size floor.** cross_asym 90% at effect 0.15, 100% at ≥0.20 → the
weaker-than-cell-type-but-detectable floor is ~0.15–0.20 in log space (the marker gap is
~0.9). Quantifies "weak but noticeable."

**Coupling (existence).** `signature_coupling(donor_level=True)` (10 donors) recovers the
truly-coupled pairs at ~100% recall at t6 (93% at t3 — deeper edges emerge later); overall
false-positive rate 8–11%. The isolated negatives Q,R,S,T are still spuriously flagged in
~19–24% of their pairs (13–17 of 70) — the known coupling over-call, even degree-corrected.

**Depth.** Direction accuracy is 98–100% across cascade depths 0–2 (the deep chain's weak
tail is recovered too).

**What it establishes.** cascade_forge generates ground-truth data at 1M-cell scale on
which cascadir's direction call is right and its symmetric control is at chance — a clean
synthetic validation of the method, plus a quantified detectability floor and an honest
readout of the coupling over-call on negatives.
