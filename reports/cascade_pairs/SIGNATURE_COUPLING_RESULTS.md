# Signature-space coupling — results (CLAUDE.md §28)

Run 2026-06-03. A reframe of Path A: measure cytokine **coupling in the cytokine-specific
gene dimensions** (the discovered binary-IG signatures `S_X`), instead of in the encoder
latent space. Driver `scripts/run_signature_coupling.py`; module
`cytokine_mil/analysis/signature_coupling.py`.

## Why (the diagnosis)

Latent-geometry Path A (§20) measures coupling in the 128-d encoder embedding with PBS-RC.
PBS-RC removes the *resting* baseline but **not the shared post-activation program** — the
immune genes nearly every cytokine co-induces — so apparent coupling is dominated by
"cytokines all activate," not by specific biology. Evidence on Oesinghaus:
- **Spearman(coupling, Path A `axis_strength`) = +0.11** over 1128 pairs → the two notions
  rank coupling almost independently.
- `IL-6 — TNF-α` (pre-registered, mid Path A strength) has **negative** signature-space
  coupling: IL-6→STAT3 and TNF→NF-κB are specifically *distinct*, coupled only via shared
  activation. Latent geometry sees the activation; signature space sees the real biology.
- On Sheu the latent-geometry gate had **no power at all** (q=1 everywhere) — the 500-gene
  panel is *all* shared-activation genes.

## Method

Build the cross-engagement matrix `M[a,b] = s(a, S_b) − s(PBS, S_b)` (a's cells expressing
b's signature, PBS-normalised, median over cell types), in raw gene-expression space — **no
encoder embedding**. Two readouts:
- **Coupling (symmetric):** `M[a,b] + M[b,a]`, gated by a gene-set null (vs random gene sets
  of equal size) — "the strong-enough-signal gate."
- **Direction (antisymmetric):** `M[a,b] − M[b,a]` = cross_asym (§26), read only on coupled
  pairs.

(Unit-tested to match `directional_asymmetry_test` exactly; 11 tests pass.)

## Sheu — the decisive "irrelevant features" test ✓ WIN

Does coupling in *specific* dimensions recover the textbook IFN cascades that
latent-geometry Path A **failed** (0/2, q=1)? Run single-frame at 3hr (the frame the gate
failed on) and 5hr (best cross_asym frame), each with frame-matched per-TP signatures.

| frame | coupled pairs | MUST recovered | clean negatives uncoupled |
|---|---:|:-:|:-:|
| **3hr** | 18/21 | **2/2** (LPS–IFNb +0.63 p=0.000; polyIC–IFNb +1.22 p=0.000) | P3CSK–IFNb p=0.95, CpG–IFNb p=0.95 ✓ |
| **5hr** | 17/21 | **2/2** (IFNb–PIC +1.12 p=0.000; IFNb–LPS +0.71 p=0.000) | CpG–IFNb p=0.50, P3CSK–IFNb p=0.73 ✓ |

**Coupling recovery is robust across both frames (2/2, 2/2)** where latent geometry got
0/2. All NF-κB→TNF pairs also couple. **This confirms the diagnosis: Sheu's Path A failure
was measuring shared activation, not a missing signal.**

*Direction nuance:* LPS→IFNb is correct at both frames; **polyIC's direction flips to
IFNb→PIC at 5hr** — the known §26 polyIC ISG-collapse (S_polyIC ≈ S_IFNb). So coupling
(existence) is solid; direction inherits the pre-existing polyIC caveat.

## Oesinghaus — ran, but the gate is too loose ⚠

48 cytokines, 1128 pairs. The *signal* is present but the *gate* does not yet discriminate:
- **894/1128 (79%) pass the coupling null** — a gate admitting 79% of all pairs isn't
  discriminating.
- **Hub-dominated top:** IL-15 appears in 11 of the top 20 coupled pairs, CD40L in 5 — a
  degree/magnitude artifact (a broadly-engaged signature looks coupled to everything).
- **But the right biology is at the very top:** #1 `IL-15—IL-2` (the r=0.92 γc pair,
  PRE_REGISTERED), then IFN-β/IL-2, IFN-γ/IL-2, IFN-β/IFN-γ, IFN-ω/IL-15 (KNOWN pairs).
- Spearman 0.11 vs Path A → genuinely different (and arguably better-ordered) coupling.

Root cause is the same as everywhere this session: the gene-set null (cell-level, vs random
genes) only tests "are `S_X` more activation-responsive than random genes" — which most are,
because IG signatures still carry some shared-activation genes. **The gate needs a
donor-level recompute + a degree/hub correction** (z-score `M` per cytokine, or subtract
row+column effects) before it discriminates coupling on the broad pair space.

## Immune Dictionary — not run

Signature-space coupling has **not** been run on ID. (ID's latent-geometry Path A also never
produced output.) So ID currently has *no* coupling result — only the §26 direction (83%) on
hand-picked pre-registered pairs.

## Two coupling paths — complementary, mirror-image failure modes

| | Path 1 — encoder latent space (§20) | Path 2 — gene signatures (§28) |
|---|---|---|
| where | 128-d encoder embedding | raw expression over `S_X` |
| pro | rich representation; donor-level FDR; standing 121-axis result | specific dimensions; interpretable; robust where latent space is weak; gives direction for free |
| con | confounded by **shared activation**; needs rich data | rides on `S_X` specificity; gate **over-permissive** on broad spaces |
| works on | Oes ✓ | Sheu ✓ (rescued the cascades Path 1 missed) |
| fails on | Sheu ✗ (q=1) | Oes gate too loose ⚠ |

They fail in opposite ways — Path 1 too *confounded*, Path 2 too *permissive* — and each
works where the other struggles. **Likely endgame: a synthesis — Path 2's signature
specificity + Path 1's donor-level statistical discipline.**

## Next steps

1. **Donor-level + degree-corrected coupling gate** (the headline fix): recompute coupling
   significance at donor level and remove the hub/degree bias; re-evaluate the Oes axis set
   vs Path A's 121 + literature.
2. **Run signature-space coupling on ID** (it's never been tried; would complete the grid).
3. **Synthesis** of the two paths.

## Bottom line

Your reframe **works**: coupling in cytokine-specific gene dimensions recovers the cascades
(Sheu 2/2 at both frames) that the encoder-latent-space Path A missed, and on Oesinghaus it
surfaces the right biology at the top and ranks coupling very differently from latent
geometry. Its current limitation is the **coupling gate** (over-permissive / hub-biased on
broad pair spaces), which needs the same **donor-level** correction the §27 direction-FDR
needs. The signal is real; the gate is the work that remains.
