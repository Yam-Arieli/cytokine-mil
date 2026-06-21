# cascadir — MANUAL (read this first if you're an agent driving a new dataset)

`cascadir` infers **cytokine cascade structure from a single-cell snapshot**: which
cytokine pairs are *coupled* and, for coupled pairs, *which one is upstream* — with **no
time course**. This manual is written so a fresh session can run it on a new dataset
end-to-end and interpret the output honestly.

---

## 0. One-paragraph mental model

Train a per-cytokine **binary classifier** (cytokine vs control/PBS) on bags of cells
(pseudo-tubes), then read which genes it used with **Integrated Gradients** → each
cytokine's **specific signature `S_X`** (top-N genes). Build one **cross-engagement
matrix** in gene space:

```
M[a, b] = s(a, S_b) − s(PBS, S_b)     # a's cells' expression of b's signature, PBS-normalized
coupling(a, b)   = M[a, b] + M[b, a]  # SYMMETRIC      → are a, b coupled?  (existence)
cross_asym(a, b) = M[a, b] − M[b, a]  # ANTISYMMETRIC  → who is upstream?   (direction)
```

Biology of direction: an **upstream** stimulus's cells carry *both* programs (their own +
the autocrine downstream one); the downstream ligand's cells carry mainly their own ⇒
`s(upstream, S_down) > s(down, S_up)` ⇒ positive cross_asym for the upstream.

---

## 1. Data contract (what you must provide)

An **AnnData** (`adata`), cells × genes, with three `obs` columns and a control condition:

| requirement | detail |
|---|---|
| `condition_col` | the stimulus per cell (e.g. `"cytokine"`); one value is the **control** |
| `donor_col` | independent biological replicate (person / mouse / context×replicate). **The unit of independence.** |
| `celltype_col` | a cell-type label per cell (any clustering or expert label) |
| control label | the resting/unstimulated condition; **relabel it to a single string** (default `"PBS"`) and pass it as `control_label` |
| expression | raw counts **or** log-normalized; `fit(assume="raw"/"lognorm"/"auto")` handles both |
| size | aim for **≥ 3 donors** (latent coupling needs more — see §4); ≥ `min_cells` (10) per (condition, cell_type) |

The package is **dataset-agnostic**: anything satisfying the contract works. No cluster,
SLURM, or filesystem assumptions — pure in-memory.

---

## 2. Quick start (the whole method in one call)

```python
import cascadir as cd

est = cd.CascadeDirection(
    condition_col="cytokine", donor_col="donor", celltype_col="cell_type",
    control_label="PBS", device=None,         # device=None → auto cuda>mps>cpu
).fit(adata, assume="raw")                      # raw counts → auto preprocess

direction = est.direction_table()               # PATH B: who-is-upstream, all pairs
coupling  = est.signature_coupling(donor_level=True)   # signature-space coupling (existence)
axes      = est.discover_axes()                  # latent-geometry coupling (existence) — needs donors
```

`.fit()` runs: validate → preprocess (norm + HVGs) → pseudo-tubes → Stage-1 encoder →
per-cytokine binary models → IG signatures. Each stage is also a standalone function
(`cd.preprocess`, `cd.build_pseudotubes`, `cd.train_encoder`, `cd.train_all_binary`,
`cd.derive_signatures`, `cd.direction_table`, …) if you need to run one yourself.

---

## 3. The three questions and which function answers them

| question | method | returns |
|---|---|---|
| **Direction** — who is upstream? | `est.direction_table()` / `est.direction("A","B")` | `cross_asym_median`, `direction` (`a_to_b`/`b_to_a`/`ambiguous`), `classification` (STRONG/WEAK/AMBIGUOUS), `null_p` |
| **Coupling — signature space** | `est.signature_coupling(donor_level=True)` | `coupling` (=M+Mᵀ), `cross_asym` (=M−Mᵀ), `coupled` (bool), donor-level stats |
| **Coupling — latent space** | `est.discover_axes()` | `AxisResult`: `axis_strength`, `coupled` call, relay cell type, per-pair Wilcoxon |

`signature_coupling` and `direction_table` report the **same** `cross_asym` (both read it
off `M`); `signature_coupling` adds the symmetric coupling half.

---

## 3.5 Recurrent IG — signature trajectories over training (OPT-IN)

By default IG runs **once**, on the final binary model. You can instead capture IG **every
N epochs** of binary training, turning each static `S_X` into a *trajectory* of gene
rankings — and get the matching **per-epoch degree-corrected coupling panel**. This answers
"*when* does each gene get recruited into a signature, and does the coupling/direction call
stabilize over training?" (the analysis behind `hypotheses/recurrent_training_dynamics_IG.md`
and CLAUDE.md §31). It is fully opt-in; the default path is unchanged.

```python
# one call: capture every 10 epochs (or set train_config.checkpoint_ig_every_n_epochs)
est = cd.CascadeDirection(condition_col=..., donor_col=..., celltype_col=...,
                          control_label="PBS")
est.fit(adata, assume="auto", ig_checkpoint_every=10)

traj = est.signature_trajectory_table()   # tidy: condition, epoch, gene, ig, rank_ig
panels = est.coupling_trajectory()         # {epoch: DataFrame(coupling, cross_asym, ...)}
```

| method | returns |
|---|---|
| `est.fit(..., ig_checkpoint_every=N)` | captures into `est.signature_trajectories` (a `dict[str, SignatureTrajectory]`) during the **same** training pass |
| `est.signature_trajectory_table()` | long DataFrame `condition, epoch, gene, ig, rank_ig` (the full per-gene ranking unless `train_config.checkpoint_ig_top_n` is set) |
| `est.coupling_trajectory(degree_correct=True)` | `{epoch: DataFrame}` — `M+Mᵀ` (degree-corrected) and `M−Mᵀ` per pair, reusing `cross_engagement_matrix` + `_degree_center` unchanged |

Composable, lower-level entry points (run your own loop / your own models):
`cd.derive_signature_trajectory(tube_set, condition, encoder, checkpoint_every=10, ...)`
(one condition), `cd.signature_trajectory_collector(...)` (a `(trajectories, factory)` pair
to hand to `train_all_binary(checkpoint_every=..., on_checkpoint_factory=...)`), and
`cd.coupling_trajectory(signatures_by_epoch, cells_by_pair, gene_names)`.

**Cost:** one extra IG pass per checkpoint per condition (IG is cheap on GPU). The last
checkpoint at `epoch == binary_epochs` is captured after the final step, so it equals the
static `S_X` exactly (a built-in consistency anchor).

**Caveat — what the trajectory is.** With a **frozen** Stage-1 encoder the gene→feature map
is fixed across epochs, so recruitment *order* is the attention/classifier learning to
weight encoder features (read-out learning), **not** the representation drifting.
Cross-condition timing is comparable only because the encoder is shared. Treat
recruitment-order claims as descriptive unless they are seed-stable and donor-robust (the
honest-rigor bar in the hypotheses doc).

---

## 4. THE KEY DECISION — which coupling path? (the Oesinghaus-vs-Sheu lesson)

There are **two coupling paths with mirror-image failure modes.** Choose by dataset shape:

| | **Latent geometry** (`discover_axes`) | **Signature space** (`signature_coupling`) |
|---|---|---|
| space | encoder embedding (128-d), PBS-RC | cytokine-specific genes `S_X` |
| stats | donor-level Wilcoxon + FDR | gene-set null (cell-level) **or** donor sign-test (`donor_level=True`) |
| strength | rich representation; the standing published result | specific/interpretable; gives direction for free |
| weakness | **dominated by shared activation**; needs a broad panel + several donors | raw gate over-calls; **fixed by `degree_correct=True` (default)** + donor-level |
| **works on** | broad whole-transcriptome, many donors (human PBMC) | targeted panels / few donors (mouse BMDM) **and** broad data once degree-corrected |
| **fails on** | targeted panel / few donors → **q ≈ 1, no power** | (raw, no degree correction) broad data → ~80% "coupled", hub-dominated |

**Rule of thumb for a new dataset:**
- **Broad transcriptome (≥~4000 HVGs) AND ≥ ~8 donors** → `discover_axes` is one coupling
  call; also run `signature_coupling(donor_level=True)` — with `degree_correct=True` (default)
  and the donor-level gate this discriminates well (validated: see §8).
- **Targeted/curated gene panel OR few donors** → `signature_coupling` is the primary call
  (latent geometry will likely have no power). Keep `degree_correct=True`. **`donor_level`
  needs ~8+ well-covered donors** — on few-donor data (≈3–4) per-pair donor coverage
  collapses and the donor gate becomes inapplicable, so leave `donor_level=False` and rely
  on the cell-level **degree-corrected** gate (validated on the Sheu targeted panel, §8).
- Either way, **direction** (`direction_table`) is computed the same and is the most
  validated output; gate it on whichever coupling call you trust for that dataset.

WHY: latent geometry measures cytokines' deviation in the encoder embedding, which is
dominated by the **shared post-activation program** every cytokine co-induces (PBS-RC only
removes the *resting* baseline, not shared activation). On a targeted immune panel *every*
gene is a shared-activation gene → no specific signal → no power. Signature space works in
each cytokine's *specific* genes. Its raw gate over-called because a broadly-engaged
("hub") signature looks coupled to everything; **the degree correction (`degree_correct`,
on by default) subtracts each condition's overall engagement strength, leaving
pair-SPECIFIC residual coupling — the validated fix in both regimes (§8).** Being
symmetric, it changes only coupling (existence), never `cross_asym` (direction).

---

## 5. Honest caveats (do NOT skip — these are real, current limitations)

1. **Over-power / donor-level / hubs.** The cell-level gene-set null (`coupling_null_p`,
   and the `direction` `null_p`) is **over-powered**: with thousands of cells almost
   everything is "significant." **The unit of independence is the donor (effective N ≈
   #donors, not #cells).** For coupling, use `signature_coupling(donor_level=True)` (sign
   test across donors) **when you have ~8+ well-covered donors** — on few-donor datasets
   the per-pair donor coverage collapses and the donor gate can't run (keep it off and lean
   on the degree correction). Treat cell-level p-values as exploratory. Separately, the raw
   gate is **hub-dominated** (a broadly-engaged signature looks coupled to everything) —
   this is handled by `degree_correct=True` (default), the validated fix (§8). *(A
   donor-level direction null is on the roadmap; not yet in the package.)*
2. **Direction ≠ existence.** A non-coupled pair can still have a large `|cross_asym|`.
   Never use `|cross_asym|` as a coupling gate — decide coupling with a coupling method,
   then read direction on coupled pairs.
3. **Signature specificity is the linchpin.** If a cytokine's `S_X` collapses onto a shared
   program (e.g. polyIC's signature ≈ the interferon program), its direction can flip. If
   `S_a == S_b` exactly, `direction_call` raises `SignatureError` (cross_asym is 0).
4. **Small n.** Validated on 17 / 7 / 6 *labeled* pairs per dataset — a methodology
   demonstration on *known* cascades, not (yet) blind discovery.

---

## 6. Recipe for a NEW dataset (do this in order)

1. **Build the AnnData** to the §1 contract. Relabel the resting condition to one string
   and set `control_label` to it. Confirm ≥3 donors and the three `obs` columns.
2. **Sanity-validate:** `cd.validate_anndata(adata, condition_col=..., donor_col=...,
   celltype_col=..., control_label=...)` → fix any error it raises before fitting.
3. **Fit:** `est = cd.CascadeDirection(...).fit(adata, assume="auto")`. (GPU strongly
   recommended for the binary models; `device=None` auto-selects.)
4. **Pick the coupling path by §4** (broad+many-donors → `discover_axes`; targeted/few →
   `signature_coupling(donor_level=True)`).
5. **Direction:** `est.direction_table()` → keep STRONG/WEAK calls; read `direction`.
6. **Gate direction on coupling:** report direction only for pairs your chosen coupling
   call flagged as `coupled` (this is the honest "coupling → direction" pipeline).
7. **If you have known cascades:** `est.benchmark([(upstream, downstream), ...])` →
   accuracy of the direction calls vs your labels (+ the symmetric `directional_score`
   control, which should be ~chance — that contrast is the proof the *antisymmetric*
   statistic is doing the work).
8. **Interpret with §5 in mind.** Expect the signature gate to over-call on broad data;
   trust `donor_level=True`. Expect latent geometry to have no power on targeted panels.

---

## 7. Config reference (override the dataclasses on the constructor)

- `PreprocessConfig(n_hvgs=4000, target_sum=1e4, flavor="seurat_v3", batch_key=None)`
- `TubeConfig(n_per_cell_type=30, min_cells=10, n_tubes=10, seed=0)`
- `TrainConfig(embed_dim=128, hidden_dims=(512,256), attention_hidden_dim=64,
  encoder_epochs=50, binary_epochs=250, encoder_lr=0.01, binary_lr=3e-5, momentum=0.9,
  encoder_frozen=True)`
- `CrossAsymConfig(top_n=50, n_ig_steps=20, min_cells=10, magnitude_threshold=0.01,
  strong_consensus=0.75, weak_consensus=0.50, n_null_perms=100, null_seed=42)`

Pass e.g. `CascadeDirection(..., train_config=cd.TrainConfig(binary_epochs=150))`. The
defaults match the validated runs; change `top_n` (signature size) and `binary_epochs`
first if you need to trade speed for fidelity.

---

## 8. Calibration — what "good" looks like (validated results)

- **Direction** (`cross_asym`, on known cascades): **88% (human PBMC, 24h) / 86% (mouse
  BMDM, 5h) / 83% (mouse lymph-node, 4h in vivo)**; the symmetric `directional_score`
  control sits at chance (47% / – / 33%) on the same data.
- **Latent coupling** (`discover_axes`): on broad human PBMC → 121 coupling axes, ~50%
  literature-supported vs ~1% chance. On a targeted mouse panel → q ≈ 1 (no power).
- **Signature coupling** (`signature_coupling`): recovered the 2/2 textbook IFN cascades
  on the targeted panel that latent geometry missed. **Degree correction (`degree_correct`,
  default on) is the validated over-call fix in both regimes:**
  - *Broad + many donors (Oesinghaus, donor-level):* over-call 77% → **31%** while recall on
    known cascades *rose* 8/17 → **11/17** (≈2.1× enrichment vs ~1× for the raw gate).
  - *Targeted + few donors (Sheu 3hr, cell-level):* keeps the 2/2 IFN cascades (PIC–IFNb the
    #1 coupled pair, LPS–IFNb top-5), suppresses **all 3** pre-registered negatives, over-call
    ~80% → **~40%**. (The one MUST it drops, LPS–TNF, is the both-hub NF-κB pair — expected.)
  - Without degree correction the cell-level gate flags ~80% of pairs and is hub-dominated.
  - Caveat: the donor sign-flip/sign-test gauges *consistency*, not effect size — rank by
    `coupling` (the degree-corrected value) and apply a magnitude sense-check at the tail.

If your new dataset is broad+many-donor and `discover_axes` returns near-zero coupled
pairs at q≤0.05, that's the known underpower (rank by `axis_strength` instead, and lean on
`signature_coupling`). If `signature_coupling` (cell-level) flags almost everything, that's
the known over-power (switch to `donor_level=True`).

---

## 9. Pointers

- Method spec & history: `../CLAUDE.md` §20 (latent coupling / Path A), §26 (cross_asym),
  §28 (signature coupling).
- Honest results writeups: `../reports/cascade_pairs/SIGNATURE_COUPLING_RESULTS.md`,
  `../reports/cascade_pairs/GROUP_U_RESULTS.md`, `../reports/method_deep_dive/` (M5–M9).
- Worked example end-to-end: `examples/full_experiment.py`; per-step: `examples/steps/`.
- Public API: `import cascadir as cd; dir(cd)` (or `cascadir/src/cascadir/__init__.py`).
