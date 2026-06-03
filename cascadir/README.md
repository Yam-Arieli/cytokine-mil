# cascadir

**Cascade-direction inference from a single-cell snapshot.**

Given one snapshot of stimulus-labeled single-cell RNA-seq — several stimuli (cytokines,
ligands, …), a resting/unstimulated control, biological replicates, and cell-type labels —
`cascadir` estimates **who is upstream** in a signaling cascade for a pair of stimuli, using
the antisymmetric *cross-engagement asymmetry* statistic:

```
cross_asym(a, b) = s(a, S_b) − s(b, S_a)        # PBS-normalized
```

`s(x, S)` is the mean expression of gene set `S` in condition `x`'s cells; `S_a` / `S_b` are
**discovered** per-condition signatures (top genes by Integrated Gradients on a binary
stimulus-vs-control model). `cross_asym` is antisymmetric, so its **sign encodes direction**:
positive ⇒ `a` is upstream. The intuition: an upstream stimulus's cells carry *both* their
own program and the autocrine downstream program, while the downstream ligand's cells carry
mainly their own — so `s(upstream, S_down) > s(down, S_up)`.

This package is **standalone** (no cluster, no SLURM, no files written — everything runs in
memory on a device of your choice) and **strict by contract** (typed, runtime-validated,
precise errors). It is a clean extraction of the method validated on three datasets
(human PBMC 88%, mouse BMDM 86%, mouse lymph node 83%).

It is a **complete experiment runner**: from a brand-new dataset it does
**preprocess → Path A (which pairs are coupled) → bridge (signatures) → Path B (who is
upstream) → analysis (score against known labels)**.

> **Two complementary questions.** *Existence* — which pairs are coupled at all — and
> *direction* — who is upstream — are separate by design: a pair that is *not* coupled can
> still produce a large `|cross_asym|`, so cross_asym magnitude is **not** a coupling test.
> Use a coupling method to decide *whether*, then `cross_asym` to decide *which way*.

> **Coupling has TWO paths** (mirror-image trade-offs): **latent geometry**
> (`discover_axes` — coupling in the encoder embedding; works on a broad panel with many
> donors, no power on a targeted panel) and **signature space**
> (`signature_coupling` — coupling in the cytokine-*specific* genes; works on targeted
> panels / few donors and yields direction for free, but its gate over-calls on broad
> data unless `donor_level=True`). **Read [`MANUAL.md`](MANUAL.md) — it tells you which
> coupling path to use for your dataset, and is written for an agent driving a new
> dataset end-to-end.**

---

## Install

```bash
# from GitHub (this package lives in a subdirectory of the repo)
pip install "git+https://github.com/Yam-Arieli/cytokine-mil.git#subdirectory=cascadir"

# or from a local checkout
pip install -e cascadir/
```

Python ≥ 3.10. Dependencies: `numpy`, `pandas`, `scipy`, `torch`, `scanpy`, `anndata`, and
`scikit-misc` (used by the default `seurat_v3` HVG flavor — if your platform has no wheel for
it, drop it and pass `flavor="seurat"`; see below).

---

## Quickstart

```python
import scanpy as sc
import cascadir as cd

adata = sc.read_h5ad("my_experiment.h5ad")   # cells × genes

est = cd.CascadeDirection(
    condition_col="cytokine",   # obs column: which stimulus each cell got
    donor_col="donor",          # obs column: biological replicate
    celltype_col="cell_type",   # obs column: cell-type label
    control_label="PBS",        # the resting/unstimulated label
    device="cpu",               # or "cuda" / "mps" / None (auto)
).fit(adata, assume="raw")       # "raw" counts → auto-preprocess; see below

print(est.direction("IFNb", "IFNg"))   # one pair → DirectionCall
print(est.direction_table())           # all pairs, sorted by |cross_asym|
```

A runnable end-to-end demo on synthetic data (no real data needed):

```bash
python cascadir/examples/quickstart.py
```

---

## Is my data normalized or not? (read this)

`cascadir` needs **log-normalized, HVG-subset** expression. `preprocess` (called for you by
`.fit`) gets you there from either starting point and refuses to guess when it cannot tell:

| Your `adata.X` is… | What to do |
|---|---|
| **Raw UMI counts** (integer matrix) | `est.fit(adata, assume="raw")` (or `preprocess(adata, assume="raw")`). It stashes the counts, selects HVGs (`seurat_v3`, on counts), then `normalize_total(1e4)` + `log1p`. |
| **Already `normalize_total`+`log1p`** | `est.fit(adata, assume="lognorm")` and set `preprocess_config=cd.PreprocessConfig(flavor="seurat")` (seurat_v3 needs raw counts; `seurat` selects HVGs on the log values). If you *also* kept a `counts` layer, the default `seurat_v3` will use it automatically. |
| **Not sure** | Leave `assume="auto"`. If the matrix is unambiguous it just works; if it isn't (negative/z-scored values, or normalized-but-not-logged), you get a `NotPreprocessedError` that tells you exactly which `assume=` to pass. |

You can check the state yourself: `cd.is_raw_counts(adata)`, `cd.is_lognormalized(adata)`.
Re-running `preprocess` on already-log-normalized data is a no-op (it won't double-log).

---

## Input data contract

`.fit` runs `validate_anndata` first (strict). Your AnnData must have:

- `obs[condition_col]` — stimulus label per cell, **including `control_label`** (the PBS baseline).
- `obs[donor_col]` — biological replicate; **≥ 3 distinct donors** (pool `context × replicate`
  as pseudo-donors if you have fewer real replicates).
- `obs[celltype_col]` — cell-type label per cell (used for stratified sampling and as the
  per-cell-type key).
- `X` — finite, **non-negative** (raw counts or log-normalized; not z-scored).
- Enough cells per `(condition, cell_type)` (default ≥ 10) for the control and ≥ 1 stimulus.

Anything wrong is reported **all at once** with an actionable message
(`validate_anndata(..., strict=False)` returns a `ValidationReport` instead of raising).

---

## The pipeline (and how to run any step yourself)

`.fit` chains these public functions; you can call any of them directly and choose the
`device` each runs on:

```
validate_anndata        # strict suitability check
  → preprocess          # normalize/log1p + HVG (handles raw vs lognorm)
  → build_pseudotubes   # in-memory bags of cells, stratified by cell type
  → train_encoder       # Stage 1: cell-type-supervised cell encoder
  ├─ discover_axes      # coupling path 1 (latent geometry): which pairs are COUPLED (existence)
  └─ train_all_binary   # Stage 2: one stimulus-vs-control AB-MIL per condition
       → derive_signatures   # Integrated Gradients → top-N discovered S_X per condition
       → signature_coupling  # coupling path 2 (signature space): M[a,b]; coupling=M+Mᵀ, direction=M−Mᵀ
       → direction_call /    # cross_asym + null → STRONG/WEAK/AMBIGUOUS, who is upstream
         direction_table
  → score_directions    # analysis: accuracy vs known (upstream, downstream) labels
```

On the estimator: `est.discover_axes()` (Path A), `est.direction_table()` (Path B), and
`est.benchmark([(upstream, downstream), ...])` (analysis):

```python
est = cd.CascadeDirection(condition_col="cytokine", donor_col="donor",
                          celltype_col="cell_type", control_label="PBS").fit(adata)

axes  = est.discover_axes()                 # coupling path 1 (latent): coupled pairs + relay
cpl   = est.signature_coupling(donor_level=True)  # coupling path 2 (signatures): coupling + cross_asym
table = est.direction_table()               # who is upstream, per pair
bench = est.benchmark([("IFNb", "IFNg")])   # score directions vs known labels
print(axes.summary()); print(bench.summary())
```

A full runnable demo of all four stages: `python cascadir/examples/full_experiment.py`.

Every default is the value validated in the source study and lives in a config dataclass
(`PreprocessConfig`, `TubeConfig`, `TrainConfig`, `CrossAsymConfig`) you can override:

```python
est = cd.CascadeDirection(
    condition_col="cytokine", donor_col="donor", celltype_col="cell_type",
    train_config=cd.TrainConfig(binary_epochs=300),
    cross_asym_config=cd.CrossAsymConfig(top_n=50, n_null_perms=200),
    device="cuda",
)
```

A `DirectionCall` carries: `cross_asym_median`, `sign_consensus`, `classification`
(STRONG/WEAK/AMBIGUOUS), `direction` (`a_to_b`/`b_to_a`/`ambiguous`), `upstream`, `null_p`
(vs a random-gene-set null), and the full per-cell-type audit table.

---

## Honest caveats

- **Direction, not existence.** See the note at the top. Magnitude is not a coupling gate.
- **Needs separable signatures.** `cross_asym` requires `S_a ≠ S_b`. If two stimuli's
  discovered signatures collapse onto a shared program, the statistic loses its sign
  (a known, mechanistically-understood failure mode); identical signatures raise `SignatureError`.
- **Needs a stable control baseline.** The PBS/control is the per-cell-type reference; if a
  cell type has unusual constitutive expression on a signature, that cell type's call is noisy.
- **Small n / known cascades.** This is a methodology for assigning direction to *known* pairs,
  with modest per-experiment sample sizes; the per-pair null and sign-consensus are stronger
  evidence than any single median.
- **`null_p` is seeded per pair.** Each pair's random-gene-set null uses its own seeded RNG
  (reproducible in isolation), so `null_p` values are not numerically identical to the original
  research runs (which advanced one shared RNG across all pairs). The direction sign and the
  STRONG/WEAK/AMBIGUOUS classification do not depend on the null and are unaffected.
- **Latent coupling needs several donors.** `discover_axes` uses a donor-level Wilcoxon
  signed-rank test, so its `coupled` flag is meaningful only with enough donors (3 donors
  cannot reach p<0.05; the validated 121-axis run used ~10). With few donors,
  `AxisResult.underpowered` is `True` — rank by `axis_strength`, not `coupled`. And on a
  *targeted* gene panel it has **no power at all** (every q≈1) — use `signature_coupling` there.
- **`signature_coupling`'s cell-level gate is over-powered.** With thousands of cells, the
  random-gene-set null flags almost everything (it over-calls on broad data, hub-dominated).
  **The unit of independence is the donor** — pass `donor_level=True` for an honest (sign-test,
  conservative) gate. A donor-level *direction* null is on the roadmap, not yet shipped.
- **Which coupling path?** Broad panel + many donors → `discover_axes` (the standing result),
  with `signature_coupling(donor_level=True)` as a specificity cross-check. Targeted panel /
  few donors → `signature_coupling` is primary. See [`MANUAL.md`](MANUAL.md) §4. Direction
  (`cross_asym`) is computed the same way regardless and is the most-validated output.

---

## License

MIT. Part of the [cytokine-mil](https://github.com/Yam-Arieli/cytokine-mil) project.
