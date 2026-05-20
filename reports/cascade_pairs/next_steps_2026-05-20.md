# Next Steps After Literature Review

**Date:** 2026-05-20
**Trigger:** Literature review found directional inference is at chance (49% correct on 39 documented pairs). Axis discovery, however, works (50% literature-supported, vs ~1% chance baseline).

## The strategic situation in one paragraph

The cytokine-MIL pipeline is a working **cytokine pair coupling discoverer** — given 91 cytokines and PBMC single-cell data, it can find which cytokines belong to the same signaling axis with ~50% precision against literature. But it is **not** a working cascade *direction* inferrer in its current form: the geo refined readout is symmetric by construction, the ablation signal is direction-agnostic in a 24-h snapshot, and the resulting direction call is statistically a coin flip. The architectural fix (two-layer attention with separated direct-responder / cascade-relay heads) is implemented in code but has never been trained or wired into the analysis pipeline. We have three real options, with sharply different cost/reward profiles.

## Three concrete paths

### Path A — Reframe and finalize (LOW effort, immediate write-up)

**Estimated effort:** 2–3 days (mostly writing)
**Risk:** Low
**Outcome:** A finished, publishable cytokine-axis-discovery story

**Steps:**
1. **Rename the result.** Stop calling outputs "cascades"; call them "cytokine coupling axes." Update CLAUDE.md §0 hypothesis and §20.1 reporting.
2. **Replace "cascade_call" with "axis_call".** In `report_cascade_pairs.py`, the call goes from `{A→B, B→A}` to just `coupled / not_coupled`. Pool the forward and reverse relay scores; an axis is called if either direction passes the seed-stability bar.
3. **Replace cascade_pairs CSV with cytokine_axes CSV.** Columns: `(A, B)` unordered, `axis_strength` (max of fwd/rev pooled), `relay_T_candidates` (top-3 from T-distribution), `n_seeds_supporting`, `literature_status` (from this lit review: directional / coregulated / partial / novel).
4. **Update `audit_2026-05-20.md`** to note this reframing.
5. **Write the headline result paragraph** for the paper: "Out of N candidate pairs, 46 (38%) were found in published immunology axes, of which 17 were textbook directional. 54 are novel candidates for wet-lab validation." Numbers are real and defensible.
6. **Pick 3 top novel axis hits for wet-lab.** Recommended (high score + biologically plausible + relay cell type matches existing receptor expression):
   - **IFN-β ↔ IFN-ω** (pooled relay 0.074, both directions strong, both type-I IFN — likely a real direct co-induction axis worth confirming)
   - **APRIL ↔ Decorin** (0.020, no published link, could be a novel TGF-β-modulating connection)
   - **IL-27 ↔ IL-36α** (0.019, 4 seeds tied, both regulate Th1/Th17 — plausible novel cross-talk)

**What you give up:** The directional cascade ordering story. You're publishing what works, not what you originally hoped for.

### Path B — Mine existing dynamics for temporal direction (MEDIUM effort, no new training)

**Estimated effort:** 5–7 days
**Risk:** Medium — depends on whether confusion-trajectory asymmetry is detectable in our existing dynamics logs
**Outcome:** Possible recovery of directional signal from time-resolved training dynamics, without retraining

The core intuition: during training, **a real cascade A → B should be learned in a specific order**. Cells responding directly to A should be confidently classified as A early; cells responding via cascade to B-signature should be classified as B *later*, and they're initially misclassified as A. Confusion dynamics C(A, B, t) should show A → B confusion peaking before B → A confusion. CLAUDE.md §19 already specifies the pairwise (K, K, T) confusion tensor and an asymmetry score; this just needs to be wired through.

**Steps:**
1. **Verify confusion dynamics are logged.** Check `dynamics["confusion_entropy_trajectory"]` is being written by `_log_dynamics` in `training/train_mil.py` — it should be, per CLAUDE.md §8.4. If yes, also enable the full pairwise (K, K, T) tensor logging from §19.
2. **Implement `compute_asymmetry_score(C)`** in `cytokine_mil/analysis/confusion_dynamics.py`. Per CLAUDE.md §19: `Asym(A,B) = ∫_late C(A,B,t) − C(B,A,t) dt`. By construction antisymmetric: `Asym(A,B) = −Asym(B,A)`. The sign tells you the direction.
3. **Re-analyze the 8 existing seed checkpoints** on the cluster — confusion trajectories are already saved (or can be regenerated cheaply from the existing models). No retraining needed.
4. **Cross-validate against the 39 literature-documented pairs.** If `sign(Asym(A,B))` matches literature direction in >70% of the 39, you have a real directional signal. If still ~50%, the asymmetry is not in the data.
5. **Build a new pair reporter** that combines the (direction-agnostic) ablation/geo axis call with the (direction-bearing) confusion asymmetry sign. Update CLAUDE.md §20.1 with the new readout.

**What you give up:** Time. 5–7 days of cluster + analysis turnaround. If the confusion asymmetry is also at chance (which is possible), you've burned the week.

**Why this might work:** Confusion trajectories are a *temporal* signal. The geo/ablation readouts collapse training history to a single endpoint embedding. Temporal information could break the symmetry that endpoint-based methods can't.

### Path C — Train + adapt the v2 architecture (HIGH effort, architectural fix)

**Estimated effort:** 10–14 days
**Risk:** Medium-high (depends on whether SA/CA separation actually produces interpretable direct vs cascade signal)
**Outcome:** A model whose architecture *is* asymmetric, where SA attention identifies direct A-responders and CA attention identifies B-relay cells. Direction would come from "T attended by SA but not CA" → direct A-responder vs "T attended by CA but not SA" → cascade relay.

**Steps:**
1. **Train CytokineABMIL_V2 on full Oesinghaus.**
   - SLURM script exists: `scripts/run_oesinghaus_full_v2.slurm`
   - Current config: 3 seeds, 300 epochs, lr 0.003. Bump to **8 seeds** (matching the seed set we use everywhere: 42, 123, 7, 456, 789, 314, 271, 618) for stability filtering.
   - Estimated cluster time: ~24h per seed × 8 = ~8 days wall-clock with full GPU saturation (a5000)
2. **Verify v2 dynamics tracking.** CLAUDE.md §8.3 says `C_SA_i(t)` and `C_CA_i(t)` should both be logged. `experiment_v2_two_layer_attention.ipynb` exists — read it to confirm dynamics are stored per attention head.
3. **Extend `cytokine_mil/analysis/latent_geometry.py`** to compute per-attention-head centroid trajectories. Specifically: per-cell-type, per-cytokine `µ_SA_{A,T}` and `µ_CA_{A,T}` separately.
4. **Adapt the geo refined readout (CLAUDE.md §20.1).** The directional signal becomes: "T is SA-attended in A's bag but CA-attended in B's bag" → A → B cascade through T. This is the architectural asymmetry the single-layer version lacked.
5. **Re-run ablation_union chain on v2 outputs.** SLURM chain: `slurm/ablation_union/00_build_union → 01_run → 02_report`. Will need v2-specific variants of `analyze_cell_type_ablation.py` to use `a_SA` and `a_CA`.
6. **Re-run lit review against new outputs.** Repeat this current literature search on the v2 calls; goal is directional accuracy > 70% (vs 49% in v1).

**What you give up:** ~2 weeks of focused work plus cluster turnaround. If SA/CA don't separate cleanly (e.g., both converge on the same cells), the architectural fix doesn't fix the problem and you still don't have directionality.

**Why this is the right long-term play:** The current model literally cannot represent direction. Any directional signal we extract from it is either noise (geo/ablation) or temporal accident (confusion). v2 is designed to encode direction in its architecture; either it works (we get directional cascades) or it doesn't (and we definitively show direction needs time-resolved data, not just architecture).

## My recommendation

**Do Path A immediately, queue Path B, defer Path C unless A's paper review demands directional inference.**

Reasoning:
- Path A produces a finished result *this week*. The axis discovery is already strong; the lit review is already done; the writeup is mostly assembling existing artifacts.
- Path B is cheap relative to Path C (5–7 days vs 10–14) and uses data already on disk. If confusion dynamics give a clean directional signal, that's a much smaller paper revision than retraining.
- Path C is the *correct* fix but is expensive in calendar time and resources. Wait until you actually need direction — which might be never, if the axis-discovery result lands as-is.

## Decision table

| Question | Path A | Path B | Path C |
|---|---|---|---|
| Calendar time | 2–3 days | 5–7 days | 10–14 days |
| Cluster GPU-hours | 0 | 0–20 | ~200 |
| Code changes | small | moderate | large |
| Risk of dead-end | very low | moderate | moderate-high |
| Publishable result if it works | ✓ (axis discovery) | ✓ (directional + axis) | ✓ (directional + axis) |
| What if it fails | n/a, finished | fall back to Path A | fall back to Path A or B |

## Out of scope (for now)

- Retraining the InstanceEncoder with different HVGs or cell-type subsets — this changes the substrate, not the readout.
- Wet-lab validation of the 54 NOVEL axis pairs — external dependency, parallel track.
- Confusion dynamics on the 24h-kinetics-feasible subset only (CLAUDE.md §9.5) — useful for the writeup but not for direction.
- A two-stage classifier (predict B first, then A | B) — would address direction by design but doubles the training cost and was never speced in CLAUDE.md.

---

## Open question for you

Which path? Or hybrid (e.g., start A immediately while submitting Path C's training job in the background)?
