# Spec — the controlled code-path comparison

**Status:** specification only. Nothing has been run. Panel choice is open (§7).
**Becomes** CLAUDE.md §41 once the open decision is locked, per the §1 workflow.

---

## 1. What is being tested

CLAUDE.md §38.5 observed that the only variable separating every Oesinghaus fit measured
so far is **which training code ran**:

| path | fits | mean between-cytokine Jaccard |
|---|---|---|
| `cytokine_mil` | published anchor (24 cyt), §31 recurrent-IG seeds 42 / 123 (45 cyt) | **0.065, 0.077, 0.073** |
| `cascadir` | §36, §37 (×2), four §38.3 arms, four §38.4 arms, **§40** | **0.178 – 0.394** |

Fifteen fits, zero overlap. §40 is the fifteenth: the dropout + 1024-width + top-200
intervention landed at **0.186**, i.e. exactly where the path predicts and nowhere near
where the intervention predicted.

**The hypothesis:** signature specificity depends on the training code path, holding data
and hyperparameters fixed.

**Why the existing evidence is not enough.** It is entirely observational. §38.5 says so
itself: both `cytokine_mil` fits build Stage 1 with `build_stage1_manifest` over a
condition *subset* and are scored on their own panels, so path is not orthogonal to
breadth and panel. Both confounders were separately falsified (§38.3 flat across
1/19/46/91 encoder conditions; run-B-only at 0.079 on weak responders), and the sharpest
pair is §31 at 45 conditions = 0.077 against `rand45` at 46 conditions = 0.237 — but no
run has ever put the *same cells and the same tubes* through both paths.

**Why it matters beyond §38.** CLAUDE.md §29 states `cascadir` "mirrors the research
code". If the paths genuinely diverge on signature specificity, that touches everything
fitted through `cascadir` — §30 (COVID), §31's coupling panel, §32 (vaccination), §36,
§37, §38, §40 — and the packaged method itself. It does **not** touch the published
Oesinghaus/Sheu/ID direction numbers, which were produced on the research path.

---

## 2. Phase 0 — no training required

Both steps are hours of CPU. **Either one can end the investigation**, so both run before
any GPU is booked.

### 0a. Tube-identity gate (hard gate)

Assert the two paths see **bit-identical** inputs. For each (donor, condition) group, hash
the concatenated float32 `X` as loaded by

* `PseudoTubeDataset(manifest, ..., gene_names=hvgs, preload=True)` — the `cytokine_mil` path;
* `full90_tube_io.load_tube_set(shard_dir, ...)` — the `cascadir` path (§36 shards).

Also compare cell counts, cell-type label vectors, and gene order.

`prepare_oesinghaus_full90.py:59-79` builds the shards by reading the *same* committed
`.h5ad` tubes, reindexed to the same HVG list and cast to float32, so these are expected
to match. **If they do not, the answer is the data and not the code**, every downstream
phase is moot, and that is the finding.

### 0b. IG transplant — on the anchor itself

The published run B's trained weights survive on the cluster at
`results/oesinghaus_binary_missing16/run_20260530_191127_pid213865/`: one shared
`encoder_shared_stage1.pt` (32-key `OrderedDict`) plus 16 `model_<cyt>.pt` (37-key, with
`encoder.*` / attention / classifier prefixes). The `encoder.*` keys match
`cascadir.models.InstanceEncoder` **exactly** — verified by diffing the two encoder
modules: identical layer names and shapes, the only structural difference being §40's
parameter-free `pre_embed_drop`.

So: load the anchor's weights into `cascadir.AbMil`, run `cascadir.derive_signature`; and
independently run `scripts/run_binary_ig_probe.py`'s IG over the *same* weights. Same
models, same cells, two IG implementations. Then the reverse: §40's 90 heads
(`results/oes90_dc/models/`) through `cytokine_mil`'s IG.

The adapter must **assert key-for-key** that the mapping is total before loading — a
silently partial `load_state_dict` would fabricate the result.

Readout: per-cytokine top-50 Jaccard between the two IG implementations, and the resulting
meanJ from each.

**If the two IG paths disagree on the same weights, the search is over** at zero GPU cost.
This is the highest value-per-cost step in the whole programme and runs first.

---

## 3. Phase 1 — the controlled A/B

Runs only if Phase 0 is clean (0a matches, 0b agrees).

**Held identical across arms:** condition set; tubes (§36 shards, k=10); Stage-1 cells,
cell-type labels and val split; gene order; D2/D3 excluded **everywhere** including Stage 1
(§16); `top_n=50`; and the published "wide" hyperparameters — `embed_dim=512`,
`hidden_dims=(512,512)`, `attention_hidden_dim=128`, Stage-1 20 epochs @ 0.005 mom 0.9,
Stage-2 250 epochs @ 3e-5 mom 0.9, encoder frozen.

**Arms:** `P` = pure `cytokine_mil`, `C` = pure `cascadir`. **Three seeds each (42/123/7).**

### The seed control is not optional

The claimed effect (0.065–0.077 vs 0.18–0.39) has **never** been compared against
within-path seed variance at fixed settings. The only within-path replicate anywhere is
§31's two seeds (0.077, 0.073), and those come from a different protocol. If within-arm
spread is comparable to the between-arm gap, the hypothesis dies here and nothing further
is worth running. The stored memory note that the coupling pipeline is seed-noisy makes
this a live risk, not a formality.

**Primary readout:** mean pairwise Jaccard at top-50 via `analyze_encsweep.diversity`,
used verbatim — the same function behind every number in §38.3/§38.4/§40, so results drop
straight into the existing ladder.
**Secondary:** top-5 pool size, worst shared top-5 gene, distinct-gene count, collapse_x.

---

## 4. Phase 2 — bisection

Runs only if Phase 1 reproduces the gap. Feasible because the encoder `state_dict` keys are
identical in both directions (§2, verified).

| arm | Stage 1 | Stage 2 | IG |
|---|---|---|---|
| `T1` | `cytokine_mil` | `cascadir` | `cascadir` |
| `T2` | `cascadir` | `cytokine_mil` | `cytokine_mil` |

| T1 | T2 | conclusion |
|---|---|---|
| ≈ P | ≈ C | the effect is in **Stage 1** (the encoder weights) |
| ≈ C | ≈ P | the effect is **downstream of Stage 1** |
| both mid | | distributed / an interaction — bisect within a stage |

Combined with 0b (which isolates IG on fixed weights), this partitions the chain
{Stage 1, Stage 2, IG} for two extra fits.

Note §39.5 already **falsified** encoder gene-space geometry as the mechanism, so a
"Stage 1" verdict would mean the encoder weights matter through some property that
geometry does not capture — a genuinely new question, not a return to §39.

---

## 5. Pre-registration (lock before running, per §25.1)

* **Gate A (0a):** the two paths' `X` hashes match for every (donor, condition) group.
  Failure ⇒ stop and report; the finding is data provenance, not code.
* **Gate B (0b):** per-cytokine top-50 Jaccard between the two IG implementations on
  identical weights ≥ **0.95**. Below ⇒ IG is at least part of the answer.
* **P1 (the gap reproduces):** `mean(C) − mean(P) ≥ 0.08` with non-overlapping 3-seed ranges.
* **P2 (seed control):** within-arm seed spread < 1/3 of the between-arm gap.
* **Falsification:** `|mean(C) − mean(P)| < 0.03` ⇒ the path hypothesis is **rejected**,
  §38.5's correlation was confounded after all, and the cause of the collapse is again
  unknown. This outcome is live and must be reported as prominently as a confirmation.

---

## 6. Cost

Phase 0: no GPU. Phase 1: 2 paths × 3 seeds × N conditions binary models, plus 6 Stage-1
encoders — at N=24 that is 144 binary models, comparable to a single §38 sweep arm.
Phase 2: 2 more fits.

---

## 7. OPEN DECISION — the condition set

This is a scientific choice with real consequences and is **not** made here.

| option | n | comparable to | cost | drawback |
|---|---:|---|---|---|
| published run B's cytokines | 16 | the anchor's only **single-encoder** reference (0.079) | lowest | small panel ⇒ 120 pairs; composition chosen by benchmark membership |
| published-24 | 24 | the whole §38 ladder + §40 (0.065) | low | the anchor for it is a **merge of two encoders** (§38.5) |
| §38 seeded-random sweep24 | 24 | all eight §38.3/§38.4 arms (0.180–0.257) | low | no `cytokine_mil` fit exists on it, so the anchor is not directly comparable |
| all 90 | 90 | §36/§37/§40 | ~4× | expensive; no `cytokine_mil` fit at this breadth |

---

## 8. Guards

Reuse §37/§38 machinery unchanged: encoder sha256 (§27.6), tube-shard sha256, a head
refusing to recombine with a different encoder, `assert_agnostic()` at the top of every
stage, and the static AST benchmark-reference check with its positive control. **New:**
the Phase-0a `X`-hash gate, as a first-class assertion rather than a report line.

---

## 9. Honest limits

* This localises **where** the difference lives, not **why**. A Stage-2 or IG verdict still
  needs a line-level bisect inside that stage.
* Even the winning path sits ~10× above chance overlap. This recovers a known-good
  operating point; it does not make signatures clean in absolute terms.
* Nothing here validates the published 88%. That anchor still has Stage-1 D2/D3 leakage
  (§38.5 measured the leakage as *hurting* specificity, but it remains a protocol
  violation).
* Direction ≠ existence ≠ causation (§26.4) carries over.
