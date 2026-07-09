# Large experiment plan — 10 donors × 20 labels × 5000 cells/tube (~1.05M cells)

## Goal

Scale the quickstart (6 labels, 12.6K cells, 100% direction) to a **stress test**: a
richer authored cascade graph on **~1.05M cells** (10 donors × 21 conditions × 5000),
and measure where `cascadir`'s direction/coupling calls hold and where they break
(deep hops, weak/slow edges, false positives on isolated labels).

`10 × 20 × 5000 = 1,000,000` stimulated cells; adding the required `PBS` control
condition makes it **21 conditions → 1,050,000 cells**.

## Authored ground truth (20 labels)

A mix that exercises every regime cascadir must handle:

| structure | edges (strength, δ) | labels | tests |
|---|---|---|---|
| deep chain (depth 4) | A→B (0.75,2) · B→C (0.65,2) · C→D (0.55,2) | A,B,C,D | multi-hop depth + weak/deep floor |
| chain (depth 3) | E→F (0.70,1.5) · F→G (0.60,1.5) | E,F,G | shorter chain |
| fan-out | H→I (0.70,1) · H→J (0.60,2) · H→K (0.50,3) | H,I,J,K | one source, 3 targets, mixed delays |
| fan-in | L→N (0.65,1) · M→N (0.60,1) | L,M,N | two parents into one |
| feedback loop | O→P (0.60,1) · P→O (0.45,1) | O,P | bidirectional → **excluded** from signed accuracy |
| isolated negatives | — | Q,R,S,T | **false-positive control** (must stay uncoupled) |

20 labels, **12 direct edges** (10 signed-scorable + the O↔P feedback pair), 4
negative-control labels. Concretely:

```python
cascades = {
    "A": {"B": (0.75, 2.0)}, "B": {"C": (0.65, 2.0)}, "C": {"D": (0.55, 2.0)},  # deep chain
    "E": {"F": (0.70, 1.5)}, "F": {"G": (0.60, 1.5)},                            # chain
    "H": {"I": (0.70, 1.0), "J": (0.60, 2.0), "K": (0.50, 3.0)},                 # fan-out
    "L": {"N": (0.65, 1.0)}, "M": {"N": (0.60, 1.0)},                            # fan-in
    "O": {"P": (0.60, 1.0)}, "P": {"O": (0.45, 1.0)},                            # feedback
    # Q, R, S, T: not present -> isolated negative-control labels
}
# ensure Q,R,S,T exist by passing them explicitly, e.g. adding them as isolated labels
```

*(Isolated labels have no dict entry; the simulator must still emit them as conditions.
Add an `isolated_labels=("Q","R","S","T")` argument — see "Code change" below — since a
label that never appears in `cascades` is otherwise unknown to the graph.)*

## Simulator config

```python
sim = cf.CascadeSimulator(
    cascades,
    isolated_labels=("Q", "R", "S", "T"),   # NEW arg (small addition)
    n_cell_types=8,             # richer than quickstart's 4 (relays/consensus meaningful)
    n_cells_per_tube=5000,
    n_donors=10,
    effect_size=0.30,           # weak default; sweep 0.15/0.20/0.30/0.40 as a floor curve
    responder_mode="all",       # primary; also run a "receptor" variant for relays
    output="raw", sparse=True,  # NEW: sparse X (see below) to keep 1M cells in memory
    seed=0,
)
result = sim.simulate(snapshot_times=[3.0, 6.0])   # early vs mature; deep edges emerge later
```

Gene panel: `50 HK + 8×20 markers(160) + 20×50 programs(1000) + 200 bg = 1410 genes`.

## Resource estimate

- **Cells × genes:** 1.05M × 1410. Dense float32 ≈ **5.9 GB**; raw Poisson counts are
  mostly zeros on low-rate genes → **store `X` sparse (CSR)** to cut RAM/disk to ~1–2 GB.
- **Simulation (numpy, CPU):** vectorized over 210 tubes; minutes + ~6–12 GB RAM dense
  (much less sparse). A mem node is enough; no GPU needed to *forge*.
- **cascadir fit (GPU):** Stage-1 encoder trains on all 1.05M cells (cell-type
  classification) → **GPU strongly recommended**; 20 binary MIL models (250 epochs) run
  on subsampled pseudo-tubes (cheap); IG + cross_asym pool cells per (condition,celltype)
  (cheap). Est. **~1–3 h on one GPU**. Run on the HUJI cluster per CLAUDE.md §2.

## One small code change first

The package already handles `n_cell_types=8`, `n_labels=20`, `n_cells_per_tube=5000`,
`n_donors=10` as-is. Two additive tweaks make the large run clean:

1. **`isolated_labels=(...)`** on `CascadeSimulator` — labels with no edges that must
   still appear as conditions (the negative controls). (Graph/labels union + a validation
   that they don't collide with cascade labels or the control.)
2. **`sparse=True`** — return `X` as `scipy.sparse.csr_matrix` for `output="raw"` (Poisson
   counts) so 1M cells fit comfortably; AnnData/h5ad store it natively. Add a unit test
   that sparse and dense agree in expectation.

Both are backward-compatible and small (~1 test each).

## Run scaffold (add under repo `scripts/` + `slurm/`, per project convention)

- `scripts/forge_large_cascade.py` — build the dict above, run the simulator, `save()`
  the h5ad(s) + `ground_truth.json` to `results/cascade_forge_large/`. (CPU/mem node.)
- `scripts/benchmark_large_cascade.py` — load h5ad, `cd.CascadeDirection(...).fit(...)`,
  then: `benchmark(direct_edges − feedback)`, `signature_coupling(donor_level=True)`
  (existence, incl. the negative controls), write `RESULTS.md`. (GPU node.)
- `slurm/cascade_forge_large/{forge,benchmark}.slurm` + `submit.sh` (mirror
  `slurm/group_u/`).

## Metrics / what we'll report

1. **Direction accuracy** (`cross_asym`) on the 10 signed direct edges, per-edge table
   with `|cross_asym|` and STRONG/WEAK/AMBIGUOUS class.
2. **Accuracy vs depth** — does the deep chain's weakest edge (C→D, 0.55) and the slow
   fan-out branch (H→K, δ3) still recover, and only at the later snapshot (t=6 vs t=3)?
3. **Symmetric `directional_score` control** — should sit near chance over 10 pairs
   (the contrast that proves the antisymmetric statistic carries the signal).
4. **Coupling specificity / false positives** — `signature_coupling` over all
   C(20,2)=190 pairs: the 12 real edges should be coupled; the 4 isolated labels
   (Q,R,S,T, i.e. 4×19 − internal = their pairs) should **not** be flagged. Report the
   over-call rate (§28.2 degree-correction default on).
5. **Feedback pair** O↔P — expected small/ambiguous `cross_asym` (excluded from signed
   accuracy); sanity-check it isn't confidently mis-called.
6. **Sweeps** (optional, cheap re-forges): `effect_size ∈ {0.15,0.20,0.30,0.40}`
   (detectability floor) and `snapshot_times` (deep edges emerge later).

## Hypotheses / success criteria

- Strong, shallow edges (A→B, E→F, H→I, fan-in) → **direction ≈ 100%**.
- The weakest/deepest/slowest edges (C→D, H→K) → recovered at t=6, possibly missed at
  t=3 → demonstrates the pseudo-time axis is real, not cosmetic.
- Isolated negatives → **not** coupled (low false-positive rate) → specificity check.
- Symmetric control clearly below `cross_asym` (now with 10 pairs, not 4).

**GREEN** = signed direction accuracy ≥ 0.9 on shallow edges AND symmetric control clearly
lower AND isolated negatives largely uncoupled. Deep/slow-edge misses at the early
snapshot are informative, not failures.
