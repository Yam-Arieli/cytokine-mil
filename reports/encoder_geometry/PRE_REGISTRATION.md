# Pre-registration — does the Stage-1 encoder bottleneck the gene space?

**Locked 2026-08-26, before `slurm/encoder_geometry/submit_encoder_geometry.sh` is run**
(CLAUDE.md §25.1). Nothing below may be edited after the probe job is submitted; the
results go in `ENCODER_GEOMETRY.md` beside this file.

## Question

Every Oesinghaus-90 re-fit emits signatures far less cytokine-specific than the published
anchor (mean between-cytokine Jaccard 0.18–0.39 vs 0.065; chance 0.006). Four sweeps have
falsified every settings-level explanation — condition breadth, encoder width, tube count
k, Stage-1 epochs, `top_n`, over-training, cell-level memorisation, ranking conventions,
D2/D3 leakage, Stage-1 donor structure, Stage-1 volume, the two-run merge artifact, panel
composition, cell-type stratification (§38.3 / §38.5).

This asks a different kind of question: **what does the trained encoder do to the gene
space?** If it maps many genes onto a few shared directions, every binary head sits on the
same impoverished representation, and IG must return overlapping genes for every cytokine
regardless of what the head learned. IG attributions factor as

```
dy/dx_g = (dy/dh) · (dh/dx_g)
```

so a low-rank `{dh/dx_g}` confines **all** heads to one low-dimensional gene subspace.

Two facts constrain the claim, and are stated here so they cannot be quietly dropped later:

1. The compression is at the **input** — `input_proj` is `Linear(4000 → 512)`, a 7.8×
   squeeze; the final block `down2` is 512→512 and compresses nothing.
2. **All fits share the identical architecture.** A bottleneck alone therefore cannot
   explain the differences between them. The claim under test is that *training* collapsed
   the effective rank, by different amounts in different fits.

## Measurement

Three probes of the gene → embedding map, identical statistics on each:

| probe | definition | status |
|---|---|---|
| `onehot` | `Z[g] = E(e_g) − E(0)` | primary as specified by the user |
| `jacobian` | `Z[g] = ∂E(x)/∂x_g` at 16 real cells (8 control, 8 stimulated, seed 42, Donor1, tube_idx 0) | **the primary for B1/B2/B4** — the only probe with the chain to IG above |
| `w1` | `Z[g] = W1[:, g]` | free linear reference |

`E(0)` is subtracted from the one-hot map because `input_proj` computes `LN(W1[:,g] + b1)`;
without it every gene carries the same bias offset and the cosine matrix reads as collapsed
for a trivial reason. `‖E(0)‖ / median‖Z_oh‖` is reported so bias domination stays visible.

Statistics: participation ratio `PR = (Σσ²)²/Σσ⁴` and `PR/d`; effective rank; variance in
the top-1/5/10 singular directions; mean pairwise `|cos|`; per-gene norm and its Gini.

**Controls.** (a) A matched-dimension **untrained** encoder per fit. This is the reference,
*not* the embedding dimension: an untrained encoder is a product of random matrices whose
spectrum already concentrates, so even a full-rank map reads well below its rank. (b) The
statistic itself carries a positive control in `tests/test_encoder_geometry.py` — a planted
rank-3 input projection must read as PR ≈ 3, a planted rank-8 as PR ≈ 8, and the metric
must order them.

## Predictions

- **B1 (a bottleneck exists).** Fits on the `cytokine_mil` path (published run B, §31
  recurrent-IG ×3) show a higher `PR/d` on the `jacobian` probe than the `cascadir` fits,
  with **no overlap between the two groups**.
- **B2 (it predicts the outcome).** Within the eight sweep arms — one shared seeded-random
  24-cytokine panel, one architecture, every other hyperparameter pinned, meanJ spanning
  0.180–0.394 — **Spearman(PR/d, meanJ) ≤ −0.7**.
- **B3 (mechanism at gene resolution).** Within collapsed fits, `Spearman(signature
  frequency, gene response norm) > 0.2` and exceeds the healthy fits' value by > 0.15.
- **B4 (the honest null).** If `PR/d` spans < 1.5× across all fits and neither B1 nor B2
  holds, the encoder's gene-space geometry is **not** the bottleneck and the collapse lives
  downstream — in the attention/classifier head or in the IG path. Three of the last four
  sweeps returned flat; this branch is live and is named up front so a null is reportable
  rather than reframed.

**Verdict rule.** B1 **and** B2 ⇒ bottleneck hypothesis supported. Exactly one ⇒
suggestive, not established. Neither ⇒ the search moves downstream of the encoder.

Comparability is enforced by two registry fields and must not be worked around: `panel`
(meanJ is comparable only within a panel — `sweep24`, `published24`, `recurrent45` are
scored separately) and `embed_dim` (12 fits are 512-wide, the two §37 fits 1024-wide, so
cross-width comparisons go through `PR/d`).

## Honest limits

- Correlation across fits, not causation: even a clean B2 leaves open whether the geometry
  causes the collapse or both follow from something upstream.
- The one-hot probe is off-distribution by construction and LayerNorm makes the map
  non-additive. Where it and the Jacobian disagree, the Jacobian is the one with a
  mechanistic chain to IG, and the report says so rather than quoting whichever is more
  striking.
- `published_runA` is excluded: three candidate run directories exist and none is
  unambiguously the one merged into `binary_ig_all24`, so it has no exact
  encoder-to-outcome pairing.
- `s1sweep_pub_replica` violates §16 by design (D2/D3 in Stage 1). It is diagnostic-only,
  labelled as such in every output, and must never seed a production fit.
- Nothing here validates the published 88%; that anchor's own Stage-1 leakage was measured
  in §38.5 (and it *hurt*). The benchmark question is untouched.
- Direction ≠ existence ≠ causation (§26.4) carries over unchanged.
