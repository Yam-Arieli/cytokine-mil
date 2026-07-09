# cascade_forge

**Author a directional cascade graph, forge a single-cell snapshot with known
ground-truth direction — to benchmark [`cascadir`](../cascadir).**

`cascadir` infers cascade *direction* from a single single-cell snapshot (no time
course). To know whether it is *right*, you need data whose cascades you authored.
`cascade_forge` is that generator: you hand it a cascade dict, it emits an `AnnData`
(or `.h5ad`) that satisfies the `cascadir` data contract, with the ground truth stored
in `adata.uns` and in a form you can feed straight to `cascadir`'s `benchmark(...)`.

There is no Python-native, pip-installable simulator that natively supports "many
discrete cell types + a *user-authored* directional cascade graph (strength +
pseudo-time delay) with a deliberately-weak-but-detectable effect" — dyngen/scMultiSim
are R-only, SERGIO is gene-TF-granular, and scDesign/GRouNdGAN *fit* a reference (the
ground truth is learned, not authored). So this is a small, self-contained simulator.

## Install

```bash
pip install -e cascade_forge            # the generator (numpy / pandas / anndata only)
pip install -e cascadir                 # only needed to *run* the benchmark on the output
```

## Ground truth: the cascade dict

```python
cascades = {
    # source_label: { downstream_label: (strength, pseudo_time_delta), ... }
    "IL12": {"IFNg": (0.7, 2.0)},          # IL12 -> IFNg: strength 0.7, delta 2.0
    "IFNg": {"IL12": (0.4,)},              # feedback loop; delta omitted -> 1.0
    "TNF":  {"IL6": (0.6, 1.0), "IL1b": 0.5},   # a scalar also means delta 1.0
    # IL6 / IL1b appear only downstream -> they exist as labels with no outgoing cascade
}
```

- **strength** — how strongly the source induces the downstream program (the autocrine
  relay magnitude). At steady state the downstream program appears at `strength ×`
  (source activation); along a chain the strengths multiply.
- **pseudo_time_delta** — the time constant for "one full iteration" of the source
  producing the downstream label; omitted → `1.0`.
- A label that appears only as a *downstream* target (never a key) exists as a label
  with no outgoing cascade.

## Forge a snapshot

```python
import cascade_forge as cf

sim = cf.CascadeSimulator(
    cascades,
    n_cell_types=4,          # number of (pseudo) cell types
    n_cells_per_tube=300,    # total cells per donor per label
    n_donors=6,              # number of donors (biological replicates)
    effect_size=0.30,        # weak program bump << cell-type marker gap (~1.2)
    responder_mode="all",    # "all" (every cell type responds) or "receptor" (relays)
    output="raw",            # "raw" Poisson counts or "lognorm"
    seed=0,
)

result = sim.simulate(snapshot_times=[1.0, 3.0, 6.0])   # a snapshot per pseudo-time
adata = result.adatas[3.0]        # one AnnData per requested pseudo-time
result.save("out/")               # out/snapshot_t{t}.h5ad + ground_truth.json
result.direct_edges               # [("IL12","IFNg"), ("TNF","IL6"), ...]
result.reachable_edges            # transitive closure (all true directional pairs)
```

Each `AnnData` has `obs` columns `condition` / `donor` / `cell_type`, a `"PBS"` control
condition, and ground truth in `adata.uns["cascade_forge"]`.

## Benchmark cascadir against it

```python
import cascadir as cd

est = cd.CascadeDirection(
    condition_col="condition", donor_col="donor", celltype_col="cell_type",
    control_label="PBS",
).fit(adata, assume="raw")           # use assume="lognorm" if output="lognorm"

bench = est.benchmark(result.direct_edges)   # accuracy of direction calls vs your truth
```

The symmetric `directional_score` control that `benchmark` also reports should sit near
chance — that contrast is the proof the antisymmetric `cross_asym` is doing the work.

## How the direction signal is planted

For an applied label `L`, a per-label **activation** is propagated over pseudo-time: `L`
is clamped on; each edge `X→Y` ramps `Y`'s activation toward `strength × activation(X)`
with time-constant `delta` (first-order kinetics, so multi-hop cascades emerge in
order). At the snapshot time, every label's program is added to the cells scaled by its
activation. So an **upstream** label's cells carry *both* its own program and the
(partial, autocrine) downstream program, while the downstream label's cells carry only
their own — exactly the asymmetry `cross_asym = s(a,S_b) − s(b,S_a)` reads as direction.
The per-cell effect is deliberately **weaker than the cell-type signature** but
detectable at the bag (pseudo-tube) level where cascadir's MIL + Integrated Gradients
operate.

## Modules

- `graph.py` — parse/normalize the cascade dict; labels, direct + reachable edges.
- `dynamics.py` — pseudo-time activation propagation.
- `expression.py` — gene layout, cell-type means, per-label programs, responder masks.
- `build.py` — `CascadeSimulator`, `SimulationResult`, AnnData assembly & save.
