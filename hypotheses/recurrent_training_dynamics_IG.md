# Recurrent IG over training dynamics: what gene-recruitment *order* tells us

**Status:** hypothesis / experiment proposal (not run). Conceptual extension of the
binary-IG signature probe (`scripts/run_binary_ig_probe.py`, §26.2) and the
signature-coupling reframe (§28).

**One-line framing.** Today we run Integrated Gradients **once**, on the *final*
binary model, and read off the top-50 genes (`S_X`). If we instead run IG **every 10
epochs** during each cytokine's binary training, we replace the static signature with
a **gene-recruitment trajectory** — *when* each gene enters (and leaves) a cytokine's
top-50. The hypothesis is that the **order** in which genes are recruited is itself a
biological signal: a within-model analog of the cell-level "primary anchor vs secondary
relay" story (§8.3), and a candidate *independent* corroboration of the `cross_asym`
direction call (§26).

---

## 0. Setup recap (what changes, and what stays fixed)

The binary probe attributes `logit[positive_class=0]` (cytokine-vs-PBS) back to input
genes via IG, baseline = per-gene PBS mean, 20-step midpoint Riemann
(`integrated_gradients` in `scripts/run_binary_ig_probe.py`). Each cytokine has its
**own** binary AB-MIL model; **all models share one frozen Stage-1 encoder**
(`encoder_frozen=True`). So during a cytokine's training, the *only* parts that move are
the **attention module + bag classifier** — i.e. the *read-out* over a static
gene→feature map.

Two consequences that the whole document hangs on:

1. **The encoder is not a time-varying driver.** It is identical across cytokines *and*
   across epochs. It shapes *which* genes are linearly readable (and therefore which can
   ever be recruited early), but it cannot produce a *time-coupled* "gene leaves A,
   enters B" pattern by itself. Any temporal structure we see is the read-out learning,
   not the representation drifting.
2. **Cross-cytokine comparisons are only meaningful because the encoder is shared.** Two
   cytokines' IG values live in the same gene→feature coordinate system, so "gene g is
   recruited at epoch 30 in model A and epoch 90 in model B" is a comparison of like with
   like (modulo the convergence-speed caveat in §6).

### The new object: the recruitment tensor

Running IG every 10 epochs yields, per seed:

```
A[g, c, t]  = IG attribution of gene g for cytokine c at epoch t      (continuous)
R[g, c, t]  = rank of g within cytokine c at epoch t  (0 = top)
M[g, c, t]  = 1[ R[g,c,t] < 50 ]                                       (top-50 membership)
```

This is exactly the "panel matrix every 10 epochs" the question asks for: at each epoch a
(gene × cytokine) importance matrix, stacked over `t`. Everything below is a read-out of
this tensor (aggregated to the seed level — only seed-stable trajectories are trusted,
§5.1 / dynamics §9.1).

Derived per-(gene, cytokine) quantities:

- **Recruitment epoch** `τ_in(g,c)` = first `t` after which `g` stays in the top-50 for
  ≥ 80% of remaining checkpoints (a *persistence* criterion, not first touch — to reject
  boundary flicker).
- **Exit epoch** `τ_out(g,c)` = last `t` after which `g` is absent for ≥ 80% of remaining
  checkpoints (∞ if it stays).
- **Membership stability** `stab(g,c)` = fraction of checkpoints with `M=1`.
- **Rank volatility** `vol(g,c)` = MAD of `R[g,c,·]` over the epochs `g` is in band.

Derived per-(gene, epoch) quantities (the cross-cytokine view):

- **Promiscuity** `P(g,t) = Σ_c M[g,c,t]` — in how many cytokines' top-50 does `g` sit at
  epoch `t`.
- **Cytokine-specificity entropy** `H_g(t)` = entropy over cytokines of `g`'s normalized
  attribution. Falling `H_g(t)` = the gene is being "assigned" to fewer cytokines over
  training (specificity sharpening).

---

## 1. Question 1 — genes recruited **early** vs genes that only arrive **late**

### 1.1 Gene categories (per cytokine)

Partition each cytokine's top-50 by trajectory shape:

| Category | Definition | Hypothesized biology |
|---|---|---|
| **Anchor** (early-stable) | `τ_in` in first third, stays to end | High-effect, canonical, **direct/primary-response** genes — the genes that *define* the signature |
| **Climber** (late-arriving) | `τ_in` only in last third | Subtle / lower-effect-size — either **secondary/cascade** genes or **over-fit noise** (must be disambiguated, §1.3) |
| **Flicker** (transient) | enters early, `τ_out` before end | Generic-activation grabs that get out-competed once specific genes are learned (links to §2) |
| **Boundary** (volatile) | hovers near rank ~50, high `vol` | Mostly IG/sampling noise; not a real signature member |

### 1.2 Core hypothesis: recruitment order ≈ response primacy

**H1.** Genes are recruited in roughly **decreasing effect-size / increasing subtlety**
order. This is the expected behaviour of a read-out trained with a smooth optimizer
(SGD, §7): the largest, most consistent PBS-vs-stimulus contrasts give the most
loss-reduction per step and are fit first; residual variance is fit later. So
`τ_in(g,c)` is a *continuous proxy for how primary gene g is to cytokine c's response* —
a finer signal than the binary "in/out of top-50."

Concretely we expect **Anchor** genes to be the textbook canonical markers (type-I IFN
→ ISGs `ISG15/IFIT2/IFIT3/RSAD2`; NF-κB stimuli → `TNF/IL1B/CXCL8/CCL3/CCL4`; etc. —
the marker panel already in `run_binary_ig_probe.py`), and **Climbers** to be off-panel,
lower-fold-change genes.

**H1 is the gene-level analog of §8.3** ("early spike → Primary Anchor; steady climb →
Secondary Relay") — but read on the *attribution* trajectory of a gene rather than the
*confidence* trajectory of a cell. It also predicts a refinement of `S_X`: weight genes
by `stab × (1 − vol)` so the signature is dominated by trajectory-stable members
(see §8).

### 1.3 The critical disambiguation: late = secondary biology, or late = overfit?

A Climber is interesting **only** if it is real. The same "late arrival" is produced by
the model starting to memorize donor/tube-specific noise. Discriminators (all required
before any Climber is called "secondary biology"):

- **Val-AUC contribution.** Recompute the binary model's *held-out* AUC with vs without
  the Climber genes. Real secondary genes improve (or hold) val AUC; over-fit genes
  improve only train AUC (the §16 train≫val signature).
- **Seed stability.** A real Climber has a similar `τ_in` across seeds 42/123/7; an
  over-fit gene's late entry is seed-idiosyncratic.
- **Donor robustness.** Under leave-one-donor-out / donor bootstrap (§16, effective N =
  donors), the Climber's late recruitment should survive. Donor-specific memorization
  collapses.
- **Literature/biology.** Does the Climber map to a known secondary/feedback module for
  that cytokine (e.g. autocrine TNFR targets, feedback ISGs)?

**Falsifiable prediction P-early/late:** within seed-stable, donor-robust genes, `τ_in`
correlates **positively** with subtlety — operationalized as `τ_in` correlating
*negatively* with |log2FC vs PBS| (big-effect genes come first). If `τ_in` is
uncorrelated with effect size, "recruitment order = primacy" is dead and the trajectory
adds nothing over the static top-50.

---

## 2. Question 2 — genes that **sweep across cytokines** (the cross-model pattern)

The scenario: gene **G** is in cytokine **A**'s top-50 *early*, **drops out of A**, and
**appears in cytokine B's top-50 later**. The question correctly flags the trap:
**A and B are different models with no shared weights downstream of the frozen encoder,
so this is not a handoff.** There is no mechanism by which A's model "passes" G to B's
model. So we must enumerate what *could* produce the pattern and what (if anything) it
means.

### 2.1 Three candidate explanations (ranked from null to interesting)

1. **Null — independent boundary walks (default).** Two independent rank trajectories
   near the top-50 boundary will, by chance, sometimes show one falling as the other
   rises. A *single* A-out/B-in event is almost certainly this. ⇒ Never interpret single
   events; only aggregate, seed-stable, FDR-controlled structure counts.

2. **Shared-activation race within each model (the mechanistic core, ties to §22/§28).**
   The immune panels carry a **shared post-activation program** — genes nearly every
   stimulus co-induces. Early in training, a model grabs whatever separates stimulus from
   PBS, and shared-activation genes are *strong, cheap* separators ⇒ recruited early
   almost everywhere. As the read-out matures, a cytokine with a **strong specific**
   program learns its own genes and **sheds** the shared ones (they get out-competed →
   Flicker). A cytokine with a **weak/slow specific** program keeps leaning on shared
   genes longer (they stay, or even climb). So G being *early-out in A, late-in B*
   encodes **A has a stronger/faster specific signature than B for that shared gene** —
   i.e. it is a read-out of *relative signature specificity*, not a handoff. This is the
   **temporal visualization of the shared-activation confound** that §28 fights
   statically.

3. **Differential primacy of the same gene (the genuinely useful case).** If G is a
   true *direct* target of A but only a *secondary/cascade* product of B, then by H1
   (§1.2) G is recruited **early in A** (high effect) and **late in B** (low/noisy
   effect). The same gene's recruitment epoch *ranks how primary it is to each cytokine.*
   This is where the cross-model pattern becomes scientifically interesting — and it
   makes a falsifiable, direction-relevant prediction (§2.2).

### 2.2 The pay-off: a gene-level, independent corroboration of `cross_asym` direction

`cross_asym` (§26) infers A→B from the **final** signatures: an upstream stimulus's cells
carry *both* programs (its own + the autocrine downstream one), so `s(A, S_B) > s(B, S_A)`.
The recruitment tensor gives an **orthogonal** prediction about the **same** direction,
built from *timing* rather than final magnitude:

> **P-direction (recruitment).** For a Path-A-coupled pair (A, B) with shared genes
> `G_AB = S_A ∩ S_B`, define the recruitment-time advantage `Δτ(g) = τ_in(g,B) − τ_in(g,A)`.
> If **A is upstream of B**, the shared genes should be recruited **earlier in the
> upstream member** (they are A's direct targets, only B's secondary ones) ⇒ **mean
> Δτ(g) > 0** over `G_AB`.

Because P-direction uses *when* genes are recruited and `cross_asym` uses *how much*
program leaks across at the end, **agreement is genuinely independent evidence** for the
direction (and would strengthen the §26/§27 claims); **disagreement is a red flag** to
investigate per pair. This is the cross-model "sweep" turned into a direction statistic —
without ever claiming a within-network handoff.

**Hard caveat:** `Δτ` must be compared on a **normalized clock** (§6) — A and B converge
at different speeds (different N, separability), so raw epochs are not comparable.

### 2.3 Early-warning for signature collapse (the polyIC→IFNb failure mode)

`cross_asym`'s known failure (§26.4) is **signature collapse**: when `S_a ≈ S_b` (e.g.
polyIC's signature is ISG-dominated, indistinguishable from IFNb's), the sign can flip.
The tensor *predicts collapse before it bites*: if `S_A` and `S_B` are dominated by the
**same genes recruited early in both** (high `P(g,t)` for `g ∈ S_A∩S_B`, low specificity
entropy separation), the pair is on the shared-program diagonal and its direction call
should be flagged low-confidence. This converts a post-hoc known failure into a
pre-registered confidence gate.

### 2.4 The promiscuity / sharpening view (whole-panel readout)

Aggregate, across the tensor:

- **Promiscuity decay.** `P(g,t)` for shared-activation genes should be **high early,
  decay over training** as cytokines with strong specific programs shed them. Plotting
  mean `P(g,t)` over `t` for "shared" vs "specific" gene sets is a one-figure test of the
  §28 shared-activation thesis *in the temporal domain*.
- **Specificity sharpening.** Mean `H_g(t)` (cytokine-entropy of a gene's attribution)
  should **fall** over training — the panel "biclusters" genes onto their most-specific
  cytokine. The *rate* and *final value* of sharpening per gene is a specificity score; a
  gene that never sharpens (stays promiscuous to the end) is a pure shared-activation
  marker and should be down-weighted in every `S_X`.

---

## 3. Why this is worth the compute (what we gain over the static probe)

1. **A primacy ranking inside each signature** (`τ_in`): direct vs secondary genes,
   currently invisible in the flat top-50.
2. **An independent direction signal** (P-direction, §2.2) that triangulates
   `cross_asym` from timing instead of magnitude.
3. **A temporal, pre-hoc detector of the shared-activation confound** (§2.3/§2.4) — the
   single biggest threat to both the coupling gate (§28) and direction (§26).
4. **A stability-weighted `S_X`** (§8) that should reduce the boundary-noise membership
   churn in the static top-50.

---

## 4. Threats to validity (front-loaded, per project audit discipline)

- **Frozen-encoder scope.** Trajectories are read-out learning over a *static* gene→feature
  map, **not** representation learning. "Recruitment order" is partly dictated by which
  gene directions the encoder made linearly readable — a property shared by all cytokines,
  so it does not bias *cross*-cytokine comparisons, but it does mean the order is
  *encoder-conditional*, not a pure property of the biology. A contrast run with an
  unfrozen encoder would test sensitivity (different question, noted as future work).
- **Convergence-speed confound (kills naive `Δτ`).** Different cytokines have different N
  and separability ⇒ different epochs-to-plateau. **Normalize the clock** before any
  cross-model timing comparison: rank-normalize `τ_in` within each model, or index epochs
  by the model's own val-AUC fraction-of-final, not raw epoch. Report results on the
  normalized clock only.
- **IG noise & rank flicker.** IG on a small, curved model is noisy near the boundary; a
  ±few-rank wobble around 50 is mostly sampling. Mitigations: probe a **wider band**
  (track top-100, define membership at top-50 with hysteresis), use the **persistence**
  criterion (§0), and average IG over enough tubes (same tube cap as the static probe).
- **Overfitting masquerading as "late biology"** (§1.3) — gated by val-AUC contribution +
  seed + donor robustness.
- **Donor-level independence (§16).** Effective N = donors (~10 Oes), not cells/tubes.
  Every trajectory claim must survive donor bootstrap; per-cell or per-tube "significance"
  is pseudo-replication (the §27.6 over-power lesson).
- **Multiple comparisons.** genes × cytokines × epochs is huge. Any "G sweeps A→B" claim
  needs BH-FDR over the candidate set, with the §2.1 null (independent boundary walks) as
  the explicit null model — ideally a **timing-permutation null** (shuffle epoch labels
  within each gene×cytokine, recompute the aggregate `Δτ` / promiscuity statistic).
- **Correlational, not mechanistic (restate).** Cross-model patterns are *coincidences of
  independent fits* unless an aggregate, seed-stable, FDR-controlled, donor-robust
  structure survives. Even then it is consistent-with, not proof-of, direction (the
  §24.6/§26.4 causation caveat carries over verbatim).

---

## 5. Experiment design (minimal, additive — no method change to the package)

1. **Checkpoint during binary training.** `scripts/train_oesinghaus_binary.py` currently
   saves only the final `model_<cyt>.pt`. Add an every-10-epochs checkpoint
   (`model_<cyt>_ep<NN>.pt`) — or, cheaper, re-run training with checkpointing for the
   ~17 labeled + Group-U cytokines used in §26/§27.
2. **Recurrent IG.** Loop `run_binary_ig_probe.py`'s `integrated_gradients` over the
   epoch checkpoints (same baseline = PBS-per-gene-mean, same 20 steps, same tube cap) →
   emit `binary_ig_traj.parquet` with columns `(cytokine, gene, epoch, ig, rank_ig,
   seed)`. **CPU-only**, ~(#checkpoints) × the current ~40–80 min/run.
3. **Build the tensor + derived metrics** (§0) in a new
   `cytokine_mil/analysis/ig_dynamics.py` (mirror the static-signature conventions in
   `signature_coupling.py`); aggregate over seeds, keep only seed-stable entries.
4. **Figures:** (a) per-cytokine rank-trajectory ribbon (Anchor/Climber/Flicker colored);
   (b) mean promiscuity `P(g,t)` for shared vs specific gene sets; (c) `Δτ` distribution
   for the labeled directional pairs vs their `cross_asym` sign.

---

## 6. Pre-registered, falsifiable predictions (go/no-go flavor, per §25.1)

| ID | Prediction | Kills the idea if… |
|---|---|---|
| **P-A (primacy)** | Among seed-stable, donor-robust genes, `τ_in` correlates **negatively** with \|log2FC vs PBS\| (Spearman, donor-level) | ρ ≈ 0 ⇒ recruitment order carries no primacy info beyond the static top-50 |
| **P-B (canonical anchors)** | Marker-panel genes (ISGs for IFN, NF-κB targets for TLR/IL-1) are **Anchors** (early `τ_in`) in their expected cytokines | markers arrive late/flicker ⇒ "early = canonical" is wrong |
| **P-C (sharpening)** | Mean cytokine-specificity entropy `H_g(t)` **falls** over training; shared-activation genes stay high `P(g,t)` while specific genes' `P` decays | no sharpening ⇒ §28 shared-activation temporal picture unsupported |
| **P-D (direction triangulation)** | For labeled directional pairs, mean `Δτ` sign **agrees** with `cross_asym` on a normalized clock, beating a timing-permutation null | agreement ≈ chance ⇒ no independent direction signal; timing adds nothing to §26 |
| **P-E (collapse warning)** | Pairs flagged "shared-program early" (high early `P` on `S_A∩S_B`) are exactly the `cross_asym` low-confidence / sign-flip pairs (e.g. polyIC–IFNb) | flagged set ≠ failure set ⇒ no early-warning value |

**GREEN** (worth folding into the method): P-A and P-D both pass with donor-level rigor +
permutation null. **AMBER:** only P-A/P-B/P-C pass (descriptive primacy + sharpening, no
new direction signal) → report as an interpretability layer on `S_X`, not a direction
method. **RED:** P-A fails → recurrent IG is redundant with the static probe; drop.

---

## 7. Open questions

- Does an **unfrozen-encoder** binary model give a *different* recruitment order (the
  representation itself moving), and is that order more or less biologically ordered than
  the frozen-encoder read-out order?
- Is `Δτ` a *stronger* direction signal than `cross_asym` on the polyIC–IFNb collapse pair
  (timing might separate where magnitude collapses), or does it collapse the same way?
- Can the recruitment tensor be used to **prune** `S_X` to a stable core that improves the
  §28 donor-level coupling gate's specificity (the §28.2 over-call problem)?
- Does recruitment order transfer across **datasets** (Oes ↔ Sheu ↔ ID) for shared
  cytokines, or is it dataset/panel-specific (the §22 panel-collapse caveat)?
