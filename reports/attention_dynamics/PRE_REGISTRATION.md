# Pre-registration — Attention training-dynamics: cell-type-resolved cascade (Oesinghaus)

**Locked before the analysis job runs on real data (§25.1 discipline).** Method/spec:
CLAUDE.md §33. This file fixes the operationalizations and GREEN/AMBER/RED gates so the
verdict is not chosen after seeing the result. Direction-not-existence and not-causation
caveats (§26.4) carry over.

## Experiment

- **Data/scope:** Oesinghaus 24h PBMC, **multiclass** 91-class AB-MIL (the multiclass model
  is required: attention over all stimuli encodes *specificity*, and cross-stimulus
  comparison needs all stimuli in one model). One checkpointed Stage-2 run per seed.
- **Seeds:** 42, 123, 7 (seed-stability is the robustness axis; trajectories are point
  estimates until multi-seed agreement is shown — memory: dynamics pipeline is seed-noisy).
- **Checkpoints:** the locked grid is "≥ every 10 epochs of Stage-2"; the **full run saves a
  checkpoint every epoch** (`--checkpoint_epochs 1..250`, encoder frozen) — finer than the
  locked grid. The P1–P4 gates and `rise_frac=0.5` are **grid-agnostic** (τ is an epoch value,
  not a checkpoint index), so a finer grid only sharpens resolution; gates are unchanged.
  Frozen encoder is deliberate: the gene→embedding map is fixed, so recruitment order reflects
  the readout learning to weight cells, not representation drift, and per-cell attention at any
  epoch is reconstructable from the saved attention params.
- **Attention extraction:** `scripts/extract_attention_trajectory.py` → per-cell-type
  attention `A_X(T,t)` (donor-mean + per-donor) + within-type Gini concentration.

## Locked operationalizations

- **Recruitment epoch τ:** first checkpoint where a cell type reaches `RISE_FRAC=0.5` of its
  own final-epoch attention (relative, because attention magnitudes are ~1/N).
- **Recruitment bands:** first third / last third of the checkpoint range
  (`first_third=1/3`, `last_third=2/3`).
- **Primary** = τ in first third AND final attention ≥ the per-cytokine median across cell
  types. **Secondary** = τ in last third AND a `p_correct` **second-rise** ≥ 0.02 over the
  tail starting at τ's training-fraction. **Minor** = otherwise.
- **Relay-recruitment lag** for an ordered pair (A, B): relay cell type `T_B` = B's
  attention-primary cell type (highest final attention; data-driven, no receptor prior);
  `lag_d = τ(A, T_B; d) − τ(B, T_B; d)` per donor; aggregate = mean lag with a
  **donor-bootstrap** 95% CI (`N_BOOT=1000`, `seed=0`); **call** = sign of mean lag, declared
  only if the CI excludes 0 (else `ambiguous`). `lag>0 ⇒ A upstream (A→B)`.
- **Known cascades (benchmark):** IL-12→IFN-γ, IL-2→IFN-γ, IL-15→IFN-γ (Fig 4f/i: A→IFN-γ
  produced by NK, monocytes secondary). **Negative control:** IL-6 / TNF-α (coupled only via
  shared activation, §28 negative — no relay expected).
- **Directness proxy (P3):** per-cytokine learnability AUC = `trapz(donor-mean
  p_correct_trajectory)` (larger = more direct/easier).

## Pre-registered predictions and gates

| ID | prediction | GREEN | AMBER | RED |
|---|---|---|---|---|
| **P1** (primary recovery) | attention-primary cell type = known direct responder, recruited early | frac_match (top-3) ≥ 0.8 **and** frac_match_and_early ≥ 0.6 | frac_match ≥ 0.6 | else |
| **P2** (relay-lag direction) | known cascades give `lag>0` (CI excludes 0); control gives no lag | ≥ ⌈0.66·n⌉ cascades call `A→B` **and** ≥1 control `ambiguous` | ≥1 cascade `A→B` | else |
| **P3** (primacy/subtlety) | direct cytokines recruit their primary cell type earlier | Spearman(τ_primary, directness) < −0.1 (p<0.05) | rho < 0 | rho ≥ 0 |
| **P4** (negative control) | shared-activation-only pair shows `|lag|≈0` / CI includes 0 | both controls `ambiguous` | one `ambiguous` | both call a direction |

**Overall:** GREEN iff **P1 and P2** are both GREEN (attention dynamics add a prior-free,
cell-type-resolved, IG-independent layer worth folding into the method). AMBER if P1/P2/P3 is
partial. RED if P1 fails (attention-primary doesn't even recover known direct responders →
the readout is not grounded).

## Faithfulness guard (not a prediction)

P1's static analog already ran (`scripts/check_attention_cell_types.py`); the trajectory P1
must not *contradict* it (same top cell types at the final checkpoint). A contradiction flags
an extraction/alignment bug and is called out before any trajectory claim.
