# `cascadir` — Method Ground Truth (as implemented)

**Purpose.** This file is the single source of truth for what the `cascadir` package
*actually computes*, written as a pseudo-algorithm with exact formulas. It is derived by
reading the code, not the prose. Where the code disagrees with `thesis/thesis.tex` or
`CLAUDE.md`, **the code wins** and the disagreement is recorded in
[§0 Contradictions](#0-contradictions-code-vs-docs). Model/encoder *architecture* is
deliberately omitted (per request); everything else is here.

All references are `path:line`. Package root of the implementation:
`cascadir/src/cascadir/`.

---

## Notation

| symbol | meaning |
|---|---|
| $x$ | one condition (stimulus / cell-state / grade). One condition is the **control**, label $\mathrm{PBS}$. |
| $\mathcal G$ | the set of (HVG-selected) genes; $c_g$ = expression of gene $g$ in cell $c$. |
| $S_x \subseteq \mathcal G$ | condition $x$'s **discovered signature**, $\lvert S_x\rvert = \mathrm{top\_n} = 50$. |
| $T$ | a cell type. $\mathcal C_x$, $\mathcal C_x^{T}$ = cells of condition $x$ (of type $T$). $\mathcal C_0$ = control cells. |
| $\sigma_S(c) = \frac{1}{\lvert S\rvert}\sum_{g\in S} c_g$ | a cell's mean expression over gene set $S$. |
| $\bar\sigma_S(\mathcal C) = \frac{1}{\lvert\mathcal C\rvert}\sum_{c\in\mathcal C}\sigma_S(c)$ | population mean signature score. |
| $d$ | a donor (biological replicate); $D$ = number of donors. |

---

## 0. Contradictions (code vs docs)

These are the places where reading `cascadir` gives a different answer than the thesis or
`CLAUDE.md`. **Trust the code.**

**C1 — `cross_asym` is aggregated two different ways, and the headline number does NOT use
the thesis formula.**
The thesis (`thesis.tex:551-552, 652`) defines
$M_{ab} = \operatorname{median}_T s_T(a, S_b)$ and
$\mathrm{cross\_asym}(a,b) = M_{ab} - M_{ba}$ — a **difference of medians**.
That definition is what [`signature_coupling`](cascadir/src/cascadir/signature_coupling.py#L215-L348)
and `cross_engagement_matrix` compute
([signature_coupling.py:88-93](cascadir/src/cascadir/signature_coupling.py#L88-L93)).
But the **primary, validated** outputs — `.direction()`, `.direction_table()`,
`.benchmark()` (the 88 % / 86 % / 83 % numbers) — go through
[`directional_asymmetry_test` + `aggregate_direction`](cascadir/src/cascadir/cross_asym.py#L38-L132),
which compute $\operatorname{median}_T\big(s_T(a,S_b) - s_T(b,S_a)\big)$ — a **median of the
per-cell-type difference**, taken **only over cell types where $a$, $b$, *and* control all
have $\ge$ `min_cells`**. `cross_engagement_matrix` instead medians each term over its own
cell-type set (each requires only that *one* condition + control are present), so
$M_{ab}$ and $M_{ba}$ can be medianed over **different** cell-type sets. The two agree
only when $a$ and $b$ share the same scorable cell types. The `signature_coupling`
docstring admits this ("matches `direction_table` when conditions share cell types",
[signature_coupling.py:14-16](cascadir/src/cascadir/signature_coupling.py#L14-L16)).
→ The thesis prose describes the `signature_coupling` aggregation; the reported direction
accuracy comes from the `cross_asym.py` aggregation.

**C2 — the coupling "coupled" gate has an extra condition and a different sign-test
denominator.**
Thesis (`thesis.tex:642-646`): coupled $\iff p \le \alpha$ with
$p = 2^{-D}\sum_{k=n_+}^{D}\binom{D}{k}$ over **all** $D$ donors.
Code ([signature_coupling.py:326-345](cascadir/src/cascadir/signature_coupling.py#L326-L345)):
`coupled = (donor_sign_p ≤ α) AND (donor_coupling_mean > 0)` — an **extra positivity
requirement** — and the sign test uses $n_d$ = the number of donors in which the pair has a
**finite** (present) residual, not $D$:
$\texttt{donor\_sign\_p} = 2^{-n_d}\sum_{k=n_+}^{n_d}\binom{n_d}{k}$, with $n_+ = \#\{d : R^{(d)}_{ab} > 0\}$ among those $n_d$.

**C3 — the thesis Methods describes only *one* coupling path; `cascadir` implements two.**
The thesis coupling section (double-centring + donor sign test) is the **signature-space**
path ([§8](#8-coupling-a---signature-space-the-thesis-path)). `cascadir` also ships an
independent **latent-geometry** coupling path,
[`discover_axes`](cascadir/src/cascadir/coupling.py#L524-L579) ([§9](#9-coupling-b---latent-geometry-path-a-discover_axes)),
using the encoder embedding + PBS-RC + per-donor Wilcoxon. It is `CLAUDE.md §20`'s "Path A"
and is *not* mentioned in the thesis Methods.
**Status — deprecated, not used.** Latent geometry was evaluated and dropped after poor
results. The single validated coupling method is the signature-space $M$-based path
(§8, the thesis path). `discover_axes` is kept for documentation/reference only and is not
part of the reported pipeline.

**C4 — direction is a *pooled, cell-level* statistic; the donor is the unit of independence
only for coupling.** `.direction_table()` pools all cells across all donors
([pipeline.py:251](cascadir/src/cascadir/pipeline.py#L251) →
`cells_by_pair()` with `donors=None`, [types.py:192-229](cascadir/src/cascadir/types.py#L192-L229)),
and the direction `null_p` is a cell-level random-gene-set null. The donor-as-unit
discipline (`CLAUDE.md §5/§16`, `thesis §16`) is enforced only in the *coupling* gate
(`donor_level=True`) and the progression bootstrap — **not** in the headline direction call.

**C5 — the IG baseline is a tube-weighted control mean, not the plain control-cell mean.**
The figure caption (`thesis.tex:488`) says "the control mean copied across the $N$ cells."
Code ([signatures.py:64-74](cascadir/src/cascadir/signatures.py#L64-L74)): the baseline is
$\frac{1}{|\text{ctrl tubes}|}\sum_{\text{ctrl tube } t}\big(\text{mean over cells of }t\big)$
— each control **tube** weighted equally, not each control **cell**. (Tubes are
size-stratified so this is close, not identical.)

**C6 — degree correction is skipped when there are < 3 conditions**
([signature_coupling.py:268](cascadir/src/cascadir/signature_coupling.py#L268)); the
residual would collapse to 0. Not wrong, just unstated in the thesis.

**C7 — `directional_score` (the symmetric §24 control) is still computed and returned**
next to `cross_asym` ([cross_asym.py:98, 284](cascadir/src/cascadir/cross_asym.py#L98)).
It is antisymmetric-free / symmetric under $a\leftrightarrow b$ for self-signatures, so its
sign does **not** encode direction; it exists only as the "expect ~chance" control in
`benchmark`. Consistent with docs, listed here so it is not mistaken for a second direction
signal.

---

## The pipeline at a glance

`CascadeDirection.fit()` ([pipeline.py:108-253](cascadir/src/cascadir/pipeline.py#L108-L253)):

```
validate → preprocess → build_pseudotubes → train_encoder (Stage 1)
        → train_all_binary (Stage 2, one binary model per stimulus)
        → derive_signatures (IG)   ⇒ {S_x}
then query:
  direction / direction_table       (§10, the primary output)
  signature_coupling                (§8, coupling in signature space)
  discover_axes                     (§9, coupling in latent space)
  benchmark                         (§11)
```

Config defaults are frozen dataclasses in
[config.py](cascadir/src/cascadir/config.py); the exact numbers are inlined at each step and
tabulated in [§12](#12-config-defaults).

---

## 1. Input / data contract

Source: [pipeline.py:71-94](cascadir/src/cascadir/pipeline.py#L71-L94),
[validate.py:1-17](cascadir/src/cascadir/validate.py#L1-L17),
[types.py:192-229](cascadir/src/cascadir/types.py#L192-L229).

An `AnnData`, cells × genes, with three `obs` columns named by the caller:
`condition_col`, `donor_col`, `celltype_col`, and one condition value designated
`control_label` (default `"PBS"`). $X$ is raw counts **or** log-normalized.

---

## 2. Preprocessing

Source: [preprocess.py:176-249](cascadir/src/cascadir/preprocess.py#L176-L249)
(`normalize_log1p` [113-121](cascadir/src/cascadir/preprocess.py#L113-L121),
`select_hvgs` [124-173](cascadir/src/cascadir/preprocess.py#L124-L173)).

Detect the state of $X$ (`auto`/`raw`/`lognorm`;
[preprocess.py:56-110](cascadir/src/cascadir/preprocess.py#L56-L110)), then:

- **Raw counts:** stash counts → select HVGs on the **raw counts** via `seurat_v3`
  (`n_hvgs = 4000`) → then `normalize_total(target_sum = 1e4)` → `log1p`.
- **Log-normalized:** skip normalization; select HVGs via `seurat` on the log values
  (or `seurat_v3` if a `counts` layer is present).
- **If the panel already has $\le$ `n_hvgs` genes, keep all genes** (HVG selection is a
  no-op — targeted panels)
  ([preprocess.py:142-149](cascadir/src/cascadir/preprocess.py#L142-L149)).

Finally subset $X$ to the selected HVGs. Output $X$ is log-normalized and HVG-subset. HVGs
are computed on **all cells provided to `fit`** (before any `conditions=` subsetting, which
happens after preprocess, [pipeline.py:156-169](cascadir/src/cascadir/pipeline.py#L156-L169)).

---

## 3. Pseudo-tubes

Source: [pseudotubes.py:56-139](cascadir/src/cascadir/pseudotubes.py#L56-L139)
(`_sample_one_tube` [23-45](cascadir/src/cascadir/pseudotubes.py#L23-L45)).
Defaults: `n_per_cell_type = 30`, `min_cells = 10`, `n_tubes = 10`, `seed = 0`.

For every `(condition, donor)` pair with $\ge$ `min_cells` cells, build `n_tubes` tubes.
Each tube is a **stratified sample**: for each cell type $T$ present in that
`(condition, donor)` with $\ge$ `min_cells` cells, draw
$\min(\texttt{n\_per\_cell\_type}, |\mathcal C^T|)$ cells **without replacement**. Keep the
tube iff it has $\ge$ `min_cells` cells total. **Tube sizes are intentionally variable**
(not equalized). A tube stores its per-cell cell-type labels.

---

## 4. Stage-1 encoder training (cell-type supervision)

Source: [train.py:76-161](cascadir/src/cascadir/train.py#L76-L161).
Defaults: `embed_dim = 128`, `hidden_dims = (512, 256)`, `encoder_epochs = 50`,
`encoder_lr = 0.01`, `momentum = 0.9`, `batch_size = 256`, `seed = 42`.

Train the `InstanceEncoder` (architecture omitted) to classify **cell type** from a single
cell, on the **real preprocessed cells** (not on pseudo-tube draws). SGD + momentum,
cross-entropy, gradient-norm clip at 1.0. The trained backbone is the fixed cell
representation for Stage 2; the cell-type head is discarded downstream.

---

## 5. Stage-2 per-condition binary MIL training

Source: `train_all_binary` [train.py:411-467](cascadir/src/cascadir/train.py#L411-L467),
`train_binary_mil` [289-408](cascadir/src/cascadir/train.py#L289-L408),
mega-batch [241-281](cascadir/src/cascadir/train.py#L241-L281).
Defaults: `binary_epochs = 250`, `binary_lr = 3e-5`, `momentum = 0.9`,
`encoder_frozen = True`, `attention_hidden_dim = 64`, `seed = 42`.

For **each** stimulus $x$ (all non-control conditions), train **one binary AB-MIL** to
classify $x$-tubes vs control-tubes, with a **shared, frozen** Stage-1 encoder.
Label encoding: **positive (stimulus) → class 0**, control → class 1
([types.py:27-56](cascadir/src/cascadir/types.py#L27-L56)) — so that IG later attributes
the "this is the stimulus" logit (index 0).

**Optimization = SGD over mega-batches** ([train.py:253-281](cascadir/src/cascadir/train.py#L253-L281)):
- One epoch = a set of mega-batches. Per class, shuffle its tube indices; the minority
  class is up-sampled (with replacement) so every class has the same number of steps
  $= \max_c(\#\text{tubes of class }c)$
  ([train.py:241-250](cascadir/src/cascadir/train.py#L241-L250)).
- One mega-batch = one tube from **each** of the 2 classes. For each class tube compute
  $\text{loss}/n$ ($n = 2$), `.backward()` to **accumulate** grads over the classes, then a
  **single** `optimizer.step()`.

*(Perf note, results-identical: with a frozen encoder each tube is encoded once and cached;
training runs the attention+classifier head on cached embeddings —
[train.py:169-204, 358-390](cascadir/src/cascadir/train.py#L169-L204). Bit-identical to the
un-cached path; a pure speed-up.)*

Output: `{x : binary AB-MIL model}` (one per stimulus).

---

## 6. Signature extraction via Integrated Gradients

Source: `derive_signature` [signatures.py:77-139](cascadir/src/cascadir/signatures.py#L77-L139),
`integrated_gradients` [26-61](cascadir/src/cascadir/signatures.py#L26-L61),
`_control_baseline` [64-74](cascadir/src/cascadir/signatures.py#L64-L74).
Defaults: `top_n = 50`, `n_ig_steps = 20`.

**Baseline** (a length-$|\mathcal G|$ vector, broadcast across the $N$ cells of a tube):
$$
\mathbf b \;=\; \frac{1}{|\mathcal T_0|}\sum_{t \in \mathcal T_0}\Big(\tfrac{1}{|t|}\textstyle\sum_{c \in t} c\Big),
$$
i.e. the mean over control tubes $\mathcal T_0$ of each tube's per-gene cell-mean
(tube-weighted control mean — see **C5**).

**Integrated Gradients** of the stimulus logit $F = \text{logit}_{[0]}$ for a tube
$\mathbf x \in \mathbb R^{N\times G}$, midpoint rule, $m = 20$ steps
$\alpha_k = \frac{k - 0.5}{m},\ k=1..m$:
$$
\mathrm{IG}(\mathbf x) \;=\; (\mathbf x - \mathbf b)\;\odot\;\frac{1}{m}\sum_{k=1}^{m}\nabla_{\mathbf x} F\big(\mathbf b + \alpha_k(\mathbf x - \mathbf b)\big) \;\in\;\mathbb R^{N\times G}.
$$

**Per-gene attribution** = average IG over cells, then over the condition's tubes:
$$
A_g \;=\; \frac{1}{|\mathcal T_x|}\sum_{t \in \mathcal T_x}\ \frac{1}{|t|}\sum_{c \in t}\mathrm{IG}(t)_{c,g}.
$$

**Signature** $S_x$ = the `top_n = 50` genes with the largest $A_g$, ranked descending
([signatures.py:135-139](cascadir/src/cascadir/signatures.py#L135-L139)). This is the
data-driven estimator of the thesis's $S_X = \arg\max_{|G|=k}\operatorname{acc}_X(G)$
(no curated pathways).

Run for every stimulus ⇒ $\{S_x\}$.

---

## 7. `cells_by_pair` — the cell grouping the statistics consume

Source: [types.py:192-229](cascadir/src/cascadir/types.py#L192-L229).

Pool **all** cells (across all tubes and, by default, all donors) into
$$
\texttt{cells\_by\_pair}[(x, T)] \;=\; \text{matrix of all cells of condition } x \text{ and type } T.
$$
The control condition is always included. This dictionary — **pooled over donors** (C4) —
is the input to both the direction statistic (§10) and the signature-space coupling matrix
(§8). Per-donor variants are built on demand by passing `donors=[d]`.

---

## 8. Coupling (A) — signature space (the thesis path)

Source: `signature_coupling` [signature_coupling.py:215-348](cascadir/src/cascadir/signature_coupling.py#L215-L348),
`cross_engagement_matrix` [51-93](cascadir/src/cascadir/signature_coupling.py#L51-L93),
`_degree_center` [119-138](cascadir/src/cascadir/signature_coupling.py#L119-L138).

### 8.1 Engagement and the cross-engagement matrix $M$

Within cell type $T$, condition $a$'s **engagement** of $b$'s signature is its mean score
minus the control baseline:
$$
s_T(a, S_b) \;=\; \bar\sigma_{S_b}\!\big(\mathcal C_a^{T}\big) \;-\; \bar\sigma_{S_b}\!\big(\mathcal C_0^{T}\big).
$$
The cross-engagement matrix medians over cell types **where $a$ and the control each have
$\ge$ `min_cells`** (note: $b$'s presence is *not* required here — see C1):
$$
M_{ab} \;=\; \operatorname*{median}_{T\,:\,|\mathcal C_a^T|,|\mathcal C_0^T|\ge \texttt{min\_cells}}\ s_T(a, S_b).
$$
Diagonal excluded. ($M_{ab}$ is exactly `sA_PB_norm` of the directional test, generalized
to every ordered pair.)

### 8.2 Symmetric coupling + degree (hub) correction

$$
C_{ab} \;=\; M_{ab} + M_{ba}\qquad(\text{diagonal }=\text{NaN}).
$$
**Degree correction** (default on, applied only if $\ge 3$ conditions — C6): additive
double-centring of the symmetric $C$,
$$
R_{ab} \;=\; C_{ab} - d_a - d_b + g,\qquad d_a = \operatorname*{mean}_{b'\ne a} C_{ab'},\quad g = \operatorname*{mean}_{a}\, d_a,
$$
(NaNs excluded from all means, [signature_coupling.py:119-138](cascadir/src/cascadir/signature_coupling.py#L119-L138)).
This is symmetric, so it changes coupling (existence) **only**; `cross_asym` (direction) is
untouched. The reported `coupling` column is $R_{ab}$ (degree-corrected); `coupling_raw` is
$C_{ab}$.

### 8.3 Coupling significance — two gates

**Donor-level gate (recommended; used when `donor_level=True`).** Recompute $R^{(d)}_{ab}$
once per donor $d$ (same $M\to C\to R$ pipeline on that donor's cells). With
$n_d = \#\{d : R^{(d)}_{ab}\text{ finite}\}$ and $n_+ = \#\{d : R^{(d)}_{ab} > 0\}$, a
one-sided sign test:
$$
\texttt{donor\_sign\_p} \;=\; 2^{-n_d}\sum_{k=n_+}^{n_d}\binom{n_d}{k}.
$$
$$
\boxed{\;\texttt{coupled} \;=\; (\texttt{donor\_sign\_p}\le\alpha)\ \wedge\ (\overline{R^{(d)}_{ab}} > 0)\;}
$$
([signature_coupling.py:301-345](cascadir/src/cascadir/signature_coupling.py#L301-L345)).
See **C2** (thesis omits the $\overline{R}>0$ term and uses $D$, not $n_d$).

**Cell-level gate (fallback when no per-donor cells given).** A random-gene-set null on the
symmetric coupling: draw random gene sets of the median signature size from genes **disjoint
from every $S_x$**, build null coupling matrices (degree-corrected the same way when
applicable), and
$\texttt{coupling\_null\_p} = \operatorname{mean}_k\big(\text{null}_k \ge R_{ab}\big)$;
`coupled = coupling_null_p < α`. This gate is **over-powered** at cell scale (docstring
[signature_coupling.py:26-31](cascadir/src/cascadir/signature_coupling.py#L26-L31)) — treat
as exploratory.

Rows sorted by descending `coupling` ($R_{ab}$).

---

## 9. Coupling (B) — latent geometry (Path A, `discover_axes`) — DEPRECATED, NOT USED

> **Deprecated.** This path was evaluated, gave poor results, and is **no longer used**.
> Coupling is done **only** by the signature-space $M$-based method (§8, the thesis path) —
> the single validated coupling method. `discover_axes` is retained for documentation only.

Source: `discover_axes` [coupling.py:524-579](cascadir/src/cascadir/coupling.py#L524-L579),
`compute_directional_bias_per_donor` [123-206](cascadir/src/cascadir/coupling.py#L123-L206),
`test_directional_significance` [240-420](cascadir/src/cascadir/coupling.py#L240-L420),
`build_axis_table` [468-521](cascadir/src/cascadir/coupling.py#L468-L521).
**Not in the thesis Methods (C3).** Needs a trained encoder; the thesis path (§8) does not.

1. **Embeddings.** Run the encoder over every tube → cache
   $\{(H, \text{label}, \text{cell\_types}, \text{donor})\}$, $H_i$ = embedding of cell $i$.
2. **PBS-RC.** Per cell type $T$, $\mu_{\mathrm{PBS},T}$ = mean embedding over control cells
   of type $T$ (train donors only). Transform every cell: $\tilde H_i = H_i - \mu_{\mathrm{PBS},\tau(i)}$.
3. **Per-donor group means.** $\mu^{(d)}_{A,T}$ = mean $\tilde H$ over cells of condition
   $A$, type $T$, donor $d$. Global centroid $\mu_A$ = mean $\tilde H$ over all $A$-cells.
4. **Direction unit vector** ($\hat u$), by `direction_mode`:
   `global` → $\hat u_{A\to B} = \widehat{\mu_B - \mu_A}$; `cell_type` → $\hat u = \widehat{\mu_{B,T}}$.
5. **Per-donor forward projection**
   $b^{(d)}_{\text{fwd}}(A\!\to\!B, T) = (\mu^{(d)}_{A,T} - \mu_A)\cdot \hat u_{A\to B}$
   (reverse: same with $A\leftrightarrow B$).
6. **Significance.** Per $(A,B,T)$, a **one-sided Wilcoxon signed-rank** test
   ($H_1:\text{median} > 0$) across donors, **Bonferroni** × (number of cell types). Per
   ordered pair: $p_{\text{fwd}}(A,B) = \min_T$ Bonferroni-$p$; relay $T^\star = \arg\min_T$.
7. **Call** ([coupling.py:381-394](cascadir/src/cascadir/coupling.py#L381-L394)): with
   `fwd_sig` $= p_{\text{fwd}}(A,B)\le\alpha$, `rev_sig` $= p_{\text{fwd}}(B,A)\le\alpha$ →
   `A->B` / `B->A` / `shared` / `none`.
8. **Axis table** ([coupling.py:468-521](cascadir/src/cascadir/coupling.py#L468-L521)):
   collapse to unordered $\{a,b\}$; `coupled = call != 'none'`;
   `axis_strength = max` Wilcoxon $W$ (the ranking score); relay cell type reported.

**Power caveat** ([coupling.py:16-20, 35](cascadir/src/cascadir/coupling.py#L16-L20)):
the donor-level Wilcoxon can't reach small $p$ with few donors ($n{=}3 \Rightarrow$ best
$p = 1/8$); with `< 8` donors trust the `axis_strength` ranking, not `coupled`.

---

## 10. Direction — `cross_asym` (the primary output)

Source: `directional_asymmetry_test` [cross_asym.py:38-102](cascadir/src/cascadir/cross_asym.py#L38-L102),
`aggregate_direction` [110-132](cascadir/src/cascadir/cross_asym.py#L110-L132),
`classify_call` [135-145](cascadir/src/cascadir/cross_asym.py#L135-L145),
`direction_call` [228-328](cascadir/src/cascadir/cross_asym.py#L228-L328).
Defaults: `min_cells = 10`, `magnitude_threshold = 0.01`, `strong_consensus = 0.75`,
`weak_consensus = 0.50`, `n_null_perms = 100`, `null_seed = 42`.

**Canonicalization.** The pair is sorted alphabetically: $a \prec b$
([cross_asym.py:261](cascadir/src/cascadir/cross_asym.py#L261)). Sign convention: positive
median ⇒ $a$ upstream.

**Per cell type** $T$ (only where $a$, $b$, **and** control all have $\ge$ `min_cells` —
this common-cell-type requirement is the difference from §8, see **C1**):
$$
\mathrm{cross\_asym}_T(a,b) \;=\; \underbrace{\big[\bar\sigma_{S_b}(\mathcal C_a^T) - \bar\sigma_{S_b}(\mathcal C_0^T)\big]}_{s_T(a,S_b)} \;-\; \underbrace{\big[\bar\sigma_{S_a}(\mathcal C_b^T) - \bar\sigma_{S_a}(\mathcal C_0^T)\big]}_{s_T(b,S_a)}.
$$

**Aggregate over cell types** ([cross_asym.py:110-132](cascadir/src/cascadir/cross_asym.py#L110-L132)):
$$
\mathrm{cross\_asym}(a,b) \;=\; \operatorname*{median}_{T}\ \mathrm{cross\_asym}_T(a,b),
$$
and the **sign-consensus** (fraction of cell types agreeing with the median sign):
$$
\kappa_{ab} \;=\; \frac{1}{|\mathcal T|}\sum_{T\in\mathcal T}\mathbf 1\!\left[\operatorname{sign}\mathrm{cross\_asym}_T(a,b) = \operatorname{sign}\mathrm{cross\_asym}(a,b)\right]
$$
(for a zero median, $\kappa$ is the fraction of exact zeros).

**Classification** ([cross_asym.py:135-145](cascadir/src/cascadir/cross_asym.py#L135-L145)),
with $m = \mathrm{cross\_asym}(a,b)$:
$$
\text{call} = \begin{cases}
\text{AMBIGUOUS} & |m| < 0.01 \ \text{or } m,\kappa \text{ NaN}\\
\text{STRONG} & |m|\ge 0.01 \ \wedge\ \kappa \ge 0.75\\
\text{WEAK} & |m|\ge 0.01 \ \wedge\ 0.50 \le \kappa < 0.75\\
\text{AMBIGUOUS} & \text{otherwise}
\end{cases}
$$
**Direction** ([cross_asym.py:287-293](cascadir/src/cascadir/cross_asym.py#L287-L293)):
if STRONG/WEAK and $m\ne 0$: $m>0 \Rightarrow$ `a_to_b` ($a$ upstream);
$m<0 \Rightarrow$ `b_to_a` ($b$ upstream). Else `ambiguous`.

**Random-gene-set null** (cell-level, [cross_asym.py:153-207, 295-313](cascadir/src/cascadir/cross_asym.py#L153-L207)):
replace $S_a, S_b$ with random gene sets of the **same sizes**, drawn from genes
**disjoint from every observed $S_x$**; `n_null_perms = 100`. Two-sided empirical
$$
\texttt{null\_p} \;=\; \operatorname*{mean}_{k}\Big(\big|\mathrm{cross\_asym}^{(k)}_{\text{null}}\big| \;\ge\; |m|\Big).
$$
(If $S_a = S_b$ exactly, raise `SignatureError` — cross_asym would be 0,
[cross_asym.py:273-277](cascadir/src/cascadir/cross_asym.py#L273-L277).)

`direction_table` runs this for a list of pairs and sorts by $|m|$ descending
([cross_asym.py:331-372](cascadir/src/cascadir/cross_asym.py#L331-L372)). Also emitted per
pair: `directional_score_median` (the symmetric §24 control, C7):
$\text{asym}_{P_A} - \text{asym}_{P_B}$ where
$\text{asym}_{P_A} = s_T(a,S_a) - s_T(b,S_a)$, $\text{asym}_{P_B} = s_T(a,S_b) - s_T(b,S_b)$
([cross_asym.py:83-98](cascadir/src/cascadir/cross_asym.py#L83-L98)).

**Coupling ≠ existence.** A non-coupled pair can still have large $|m|$; `cross_asym` gives
direction only. Decide *whether* a pair is coupled with §8 or §9, then read direction on
coupled pairs.

---

## 11. Benchmark / scoring against known cascades

Source: `score_directions` [analysis.py:28-153](cascadir/src/cascadir/analysis.py#L28-L153).

Given labels as `(upstream, downstream)` pairs: canonicalize each to $(a,b)$ with $a\prec b$,
so `expected_sign = +1` if the labeled upstream equals $a$, else $-1$. Match to the
direction table by canonical pair.

- `cross_correct` = $[\operatorname{sign}(m) = \text{expected\_sign}]$.
- **`cross_accuracy`** = mean `cross_correct` over **non-AMBIGUOUS** found pairs (the
  headline denominator). `cross_accuracy_all` = over all found pairs.
- **`dirscore_accuracy`** = same for the symmetric `directional_score` control — expected
  ≈ chance; the contrast is the evidence that the *antisymmetric* statistic does the work.
- `n_null_pass` = non-AMBIGUOUS calls with `null_p < 0.05`.

---

## 12. Config defaults

Source: [config.py](cascadir/src/cascadir/config.py). Every default reproduces the validated
runs; changing them changes the science.

| group | field | default |
|---|---|---|
| Preprocess | `n_hvgs` / `target_sum` / `flavor` | `4000` / `1e4` / `seurat_v3` |
| Tube | `n_per_cell_type` / `min_cells` / `n_tubes` / `seed` | `30` / `10` / `10` / `0` |
| Train | `embed_dim` / `hidden_dims` / `attention_hidden_dim` | `128` / `(512,256)` / `64` |
| Train | `encoder_epochs` / `encoder_lr` | `50` / `0.01` |
| Train | `binary_epochs` / `binary_lr` / `momentum` | `250` / `3e-5` / `0.9` |
| Train | `encoder_frozen` / `cache_frozen_embeddings` | `True` / `True` (results-identical) |
| CrossAsym | `top_n` / `n_ig_steps` / `min_cells` | `50` / `20` / `10` |
| CrossAsym | `magnitude_threshold` / `strong_consensus` / `weak_consensus` | `0.01` / `0.75` / `0.50` |
| CrossAsym | `n_null_perms` / `null_seed` | `100` / `42` |

Global `CascadeDirection` seed for tube sampling + training: `42`
([pipeline.py:83](cascadir/src/cascadir/pipeline.py#L83)). Device auto-selects
cuda > mps > cpu ([train.py:41-49](cascadir/src/cascadir/train.py#L41-L49)).

---

## 13. Extensions (opt-in; not part of the core direction call)

### 13.1 Progression order recovery + donor-bootstrap
Source: [progression.py](cascadir/src/cascadir/progression.py) (`CLAUDE.md §30`). For a
condition axis that is a *progression* (severity grades, timepoints, cell states) with
**nested donors** (each donor has one condition), operate on a `(donor, cell_type)` score
cache and:
- `pooled_cross_asym` — the same $M_{ab}-M_{ba}$ direction statistic
  ([progression.py:111-162](cascadir/src/cascadir/progression.py#L111-L162)) with median
  over cell types; positive ⇒ $a$ upstream.
- `bootstrap_cross_asym` — resample donors **with replacement within each condition and
  the control** (donor = unit of independence), recompute pooled `cross_asym`, report
  per-pair CIs, sign-accuracy, and Kendall-$\tau$ vs the oracle order
  ([progression.py:165-251](cascadir/src/cascadir/progression.py#L165-L251)).
- `recover_order` — **Borda count** over pairwise `cross_asym` signs → a total order
  (most-upstream first), ties broken by summed signed magnitude then alphabetically
  ([progression.py:258-281](cascadir/src/cascadir/progression.py#L258-L281)).
- `kendall_tau` — concordance between recovered and true orders
  ([progression.py:316-335](cascadir/src/cascadir/progression.py#L316-L335)).

### 13.2 Recurrent IG — signature trajectories over training
Source: [dynamics.py](cascadir/src/cascadir/dynamics.py) (`CLAUDE.md §31`). Opt-in via
`fit(ig_checkpoint_every=N)` / `TrainConfig.checkpoint_ig_every_n_epochs`. Captures the IG
gene ranking every $N$ epochs of Stage-2 (same single training pass, no retraining), turning
each static $S_x$ into a per-epoch trajectory; the last checkpoint at
`epoch == binary_epochs` equals the static $S_x$ exactly. `coupling_trajectory` re-runs the
§8 cross-engagement + degree-correction math per epoch (only the signatures change). Default
path (IG once, on the final model) is unchanged when this is off.
