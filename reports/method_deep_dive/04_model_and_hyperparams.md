# M4 — The AB-MIL model + hyperparameters

Real code: `cytokine_mil/models/{instance_encoder, attention, bag_classifier, cytokine_abmil}.py`

## 1. The setup: why MIL?
A pseudo-tube = a **bag** of cells; the label (stimulus) lives at the **bag** level; we have
**no per-cell labels**. That is exactly **Multiple-Instance Learning**. We use
**Attention-Based MIL (AB-MIL, Ilse et al. 2018)**: encode each cell, let a learned
**attention** decide which cells matter, pool to one bag vector, classify it. Bonus: the
attention weights are **interpretable** (which cells drove the call) → feeds the dynamics /
relay analysis (M6).

## 2. Forward pass (one tube `X ∈ R^{N×G}`) — `cytokine_abmil.py:68-72`
```
H   = encoder(X)               # (N, d)   embed each of N cells
e_i = wᵀ tanh(V h_i)           # (N,)     one attention score per cell
a   = softmax(e) over cells    # (N,)     a_i ≥ 0,  Σ_i a_i = 1
z   = Σ_i a_i · h_i            # (d,)     attention-weighted MEAN of embeddings
ŷ   = W_c · z + b_c            # (K,)     class logits
```
Shape flow: `(N,G) → (N,d) → (N,) → (d,) → (K,)`. **One tube at a time**; variable `N` is
handled naturally (softmax + sum run over the cell axis). Returns `(ŷ, a, H)` so analysis
needs no second pass.

## 3. Components

### 3a. InstanceEncoder — a residual MLP, `G → d` (`instance_encoder.py`)
`input_proj` (Linear G→h0, LayerNorm, GELU) → `res1` (ResBlock @ h0) → `down1` (h0→h1) →
`res2` (ResBlock @ h1) → `down2` (h1→d). Defaults `hidden_dims=(512,256)`, `embed_dim=128`.
- **ResBlock**: `x + fc2(GELU(fc1(LayerNorm(x))))` (same dim).
- **DownBlock**: `skip(x) + fc2(GELU(fc1(LayerNorm(x))))`, with `skip = Linear(in→out, bias=False)` (changes dim).
- **Why residual + (pre-)LayerNorm:** the skip `x + F(x)` is a **gradient highway**
  (`∂y/∂x = I + F′`, can't vanish) and lets each block learn a small **correction** (identity
  is the default → pairs with the zero-init below; verified: zeroing `fc2` makes
  `ResBlock(x) == x` exactly). **LayerNorm is per-sample, across features** (each cell
  normalized by its *own* values) → independent of batch size and of other cells, so it works
  with **variable-size, one-tube-at-a-time bags** where BatchNorm would break. It's
  **pre-norm**: the residual stream `x` passes through raw; each block reads an LN'd *copy* of
  it, computes `F(x)`, and adds it back.
- **Identity init** (`_init_weights`, lines 102–107): Kaiming on weights, then **zero-init
  the last linear `fc2` of every block** → each block starts as an identity (ResBlock) /
  pure-skip (DownBlock) map → clean signal flow at start, initial logits ≈ uniform, so the
  **initial cross-entropy ≈ ln(K)** (a known-good starting point, no variance blow-up across skips).
- **No dropout** anywhere.
- Pre-trained in Stage 1 with a `cell_type_head`, which is then **discarded** — only the
  backbone enters the MIL model.

### 3b. AttentionModule — `attention.py` (Ilse et al. 2018, **non-gated**)
`V = Linear(d → attention_hidden_dim)`, `w = Linear(attention_hidden_dim → 1, bias=False)`.
`a = softmax( w · tanh(V·H), over cells )`.
- **Two small dims — don't conflate them.** `attention_hidden_dim` (e.g. 64) is *internal
  scratch* for the scorer (`V·h` → score) and **never reaches `z`**. `z` has size
  **`embed_dim`** because it is a weighted average of the `embed_dim` cell embeddings. Shape
  flow: `H(N,d) → V(N,A) → score(N,1) → a(N,) → z(d,)`. `attention_hidden_dim` only controls
  the *capacity of the relevance scorer*; it has no effect on `z` or anything downstream.
- **This is the plain tanh attention, NOT the gated variant** (there is no sigmoid gate `U`).
- **No dropout** — attention weights must be stable across epochs for dynamics tracking.
- **Learned pooling beats mean/max:** mean dilutes (most cells uninformative), max is brittle;
  attention learns per-cell relevance yet stays a smooth weighted average.
- **`z = Σ a_i h_i` is a convex combination** (`a` sums to 1) → a weighted *average*, on the
  same scale for any `N`. This is why variable-size bags (M3) are safe; a plain sum would grow
  with `N`. (Attention can also concentrate: boosting one cell's score → `a` ≈ one-hot → `z` ≈ that cell.)
- **`w` has `bias=False` on purpose:** softmax is shift-invariant, so a scalar added to every
  score cancels (verified) — a bias would be a no-op.
- **Interpretability (the science payoff):** `a_i` = where the model 'looked'; grouped by cell
  type it surfaces responder/relay cell types and feeds attention-entropy + instance-confidence
  trajectories (M6). This — not regularization — is why dropout is banned.
- Init Xavier (`V`, `w`) → near-uniform `a` at start (≈ mean pooling), sharpening during training.

### 3c. BagClassifier — `bag_classifier.py`
A single `Linear(d → K)` on the pooled bag vector `z`. Xavier init, zero bias. Linear on
purpose: representation power lives in the encoder; the head stays simple + interpretable.

## 4. The "knobs" (sets up Part 2)
Depth is **fixed** (input_proj + 2 ResBlocks + 2 DownBlocks). The tunable **widths** are:

| knob | meaning | set by |
|---|---|---|
| `input_dim` = G | # genes (features) | data: 4000 Oes / 500 Sheu |
| `hidden_dims=(h0,h1)` | encoder hidden widths | **choice** |
| `embed_dim` = d | bag-vector / embedding size | **choice** |
| `attention_hidden_dim` | attention internal width | **choice** |
| `n_classes` = K | # stimuli + PBS | task |

"Wider" = larger `h0/h1/d/attention_hidden_dim`; "thinner" = smaller. **Part 2 is the
reasoning for setting these** — the actual answer to *"wider or thinner given …"*.

## 5. Hyperparameters & training (M4 Part 2)

### 5.1 The 3 real binary variants (these are the Bridge models, M5)
| variant | data | #genes (quality) | embed_dim | hidden_dims | attn_hidden | Stage 1 | Stage 2 |
|---|---|---|---|---|---|---|---|
| **Sheu-binary** | mouse BMDM | 500 (curated, high SNR) | 128 | (256,128) | 64 | 20 @ 5e-3 | 200 @ **1e-4** |
| **Oes-narrow** (original 8) | human PBMC | 4000 (HVG, noisy) | 32 | (128,64) | 16 | 20 @ 5e-3 | 250 @ **3e-5** |
| **Oes-wide** (all 24, current) | human PBMC | 4000 (HVG, noisy) | 512 | (512,512) | 128 | 20 @ 5e-3 | 250 @ **3e-5** |

All: SGD, momentum 0.9, seed 42, encoder **frozen** in Stage 2. Oesinghaus moved narrow→wide
to standardise all 24 models on the higher-capacity config (`train_oesinghaus_binary_missing16.py`
docstring). The cross_asym 88% uses the **wide** Oes models.

### 5.2 Wider or thinner? — the rules (what drives width)
- **More input genes → wider.** Oes has 8× Sheu's features (4000 vs 500) → wider encoder.
- **Noisier data → wider (capacity) + lower LR (stability).** Oes HVGs are whole-transcriptome,
  noisy → wide 512 + LR 3e-5. Sheu's curated panel is high-SNR → narrower 128 suffices and a
  **higher** LR 1e-4 is safe (cleaner gradients — literally the code comment on `STAGE2_LR`).
- **Harder task / more classes → wider** (Oes multiclass was 91-way).
- **Smaller dataset → thinner** to limit overfitting (no dropout here → capacity itself is the
  regularizer). Sheu has few pseudo-donors.
- **`embed_dim` = the bottleneck.** Small (32) = aggressive compression (regularizes, can
  underfit); large (512) = expressive (more overfit risk). Oes went 32→512 because 32
  over-compressed 4000 noisy genes.
- **Tension:** noisy data wants *capacity* (wider) yet risks *overfit* (thinner). Resolution:
  go wider for capacity, control overfit with **LR, epochs, encoder-freezing, dataset design**.

### 5.3 Learning rate, optimizer, schedule
- **SGD + momentum 0.9, NOT Adam.** Adam's adaptive per-parameter LR causes non-monotonic jumps
  that **obscure the training dynamics** the project measures. SGD → smooth, interpretable
  curves (CLAUDE.md §7).
- **Stage 1 LR (5e-3) ≫ Stage 2 LR (1e-4 / 3e-5).** Stage 1 trains the encoder from scratch on
  an easy, data-rich task (cell-type classification) → big steps, few epochs (20). Stage 2
  trains attention+classifier on a **frozen** encoder for the harder, data-poor MIL task →
  small LR, many epochs (200–250).
- **Sheu S2 LR > Oes S2 LR** (1e-4 vs 3e-5): higher SNR per gene → cleaner gradients → larger
  safe steps. Optional LR warmup (~5 ep) tames erratic early steps.

### 5.4 The mega-batch trick (CLAUDE.md §7)
One **mega-batch** = one tube from **every class**, gradients **accumulated** across them, then
**one** optimizer step. Why: (a) every step sees all classes → balances class imbalance,
prevents erratic per-tube jumps; (b) it's also the *batching mechanism* — tubes are variable-`N`
and processed one at a time, so accumulation is how a "batch" is formed.

### 5.5 The 3 training stages (why staged)
1. **Stage 1 — pretrain encoder** on cell-type supervision (temporary `cell_type_head`);
   ~20 ep @ 5e-3. Goal: a good single-cell representation.
2. **Stage 2 — MIL** with the **frozen** encoder: train attention + classifier on tube labels;
   200–250 ep @ low LR.
3. **Stage 3 (optional) — unfreeze** & fine-tune end-to-end (not used in the binary trainers).

Why stage + freeze: (a) decouple *representation learning* (easy, millions of cells) from
*pooling/classification learning* (hard, few bags); (b) freezing stops the small-bag MIL
gradient from overfitting/corrupting the encoder; (c) a **stable frozen encoder is required
downstream** — dynamics tracking and especially the **Bridge's Integrated Gradients (M5)**
attribute through a *fixed* encoder, so the signatures are stable and comparable.

**Two separate encoders (don't confuse with Path A).** Each pipeline trains its *own* encoder
by this recipe. The binary **Bridge** pretrains **one** Stage-1 encoder and **shares + freezes
it across all per-stimulus binary heads** (`train_sheu2024_binary.py:192-207`, passed as
`shared_encoder=`) — so every `S_X` is attributed through the *same* fixed encoder and the
signatures share one comparable frame (what cross_asym needs in M7). It does **not** reuse
Path A's multiclass encoder; they are distinct trained objects (often different widths).
