# Group-U direction FDR — full Path A → Path B (Oesinghaus)

Run 2026-06-03. Full pipeline over **all 121 Path A coupled axes** (after training
binary IG signatures for the cytokines missing from `binary_ig_all24`), with the §27.2
direction-permutation null. Train-donors-only is the pre-registered primary; pooled is the
§26 regression anchor. DAG jobs: 30726479 (train ×8 chunks) → 30726480 (ig_merge) →
30726481 (pipeline) → 30726489 (fdr).

---

## ⚠ VALIDITY — READ FIRST

**This run is NOT a valid "discovery-capable" result.** The pre-registration's checks
(`GROUP_U_PREREGISTRATION.md`) caught two problems; do not cite the π₀ / "discovery-capable"
headline as evidence:

1. **P3 is an over-power artifact, not biology.** The direction null permutes **cell**
   labels, and with thousands of cells per (cytokine, cell-type) the permutation
   distribution is razor-tight, so **16 of 17 labeled pairs (and ~all Group-U pairs)
   pass at `dir_p_emp = 0.0000`**. The resulting Storey π₀ = 0.038 ("96% of Group-U axes
   carry reliable direction") is the null being trivially beatable at large n — the
   classic pseudo-replication trap. **The unit of independence is the donor (≈10), not
   the cell** (CLAUDE.md §16). The null must be recomputed at the **donor level**.
2. **P4 regression FAILED.** Labeled cross_asym sign accuracy is **6/11 non-AMBIGUOUS
   (≈10/16 by raw sign)** vs §26's **15/17 (88%)** — and the pooled anchor reproduces the
   same drop, so it is not the donor split. The likely cause: this run's binary
   signatures were trained with a **separate encoder per 8-way chunk** (3 cytokines
   each), unlike §26's single shared encoder, so some signatures (VEGF-heavy pairs
   especially) shifted. **Until this is explained, every number from this run is
   suspect.**

**Net:** the §27 machinery runs end-to-end and cleanly, but this is not a defensible
Group-U discovery claim. P1 (power on knowns) passes; the two failures above are both the
same recurring lesson — **at single-cell scale, cell-level nulls/gates are over-powered;
coupling and direction must be judged donor-level.**

---

## What ran

- Evaluable axes (direction computed): **121**
- LABELED (audited `counts_in_benchmark`): **17**
- GROUP U (coupled by Path A, no directional prior): **104**

## Verdict vs pre-registration

| check | result | reading |
|---|---|---|
| **P1 (power)** | 11/11 labeled non-AMBIGUOUS pass (raw p<0.05 and BH-q≤0.10) → **PASS** | knowns are detectable — but see the over-power caveat: ~everything passes |
| **P3 (headline)** | train-only π₀ = 0.038 (est. 100/104 "reliable"); pooled π₀ = 0.077 (96/104) | **artifact** — over-powered cell-level null, NOT a discovery result |
| **P4 (regression)** | labeled 6/11 non-AMBIGUOUS vs §26 15/17 → **FAIL** | re-run does not reproduce §26 → signatures differ (chunked encoders) |
| **P2 (specificity)** | descriptive | meaningless while the null over-calls |

Confident-hypothesis bar (locked): `dir BH-q(GroupU) ≤ 0.10 AND cross_consensus ≥ 0.7 AND
|cross_median| ≥ P25(labeled-positive)=0.0068` → 63 axes clear it — but inflated by the
over-power, so not reported as hypotheses here.

## Data (kept for the record, not as a claim)

Top Group-U axes by the (inflated) confident bar — biology at the top is plausible
(IL-15, IL-12, IL-27 hubs; IFN-ω→IFN-β; IL-27→IL-6 is KNOWN_DIRECTIONAL), but the null
that admitted them is over-powered:

| direction_call | literature_status | cross_median | consensus |
|---|---|---:|---:|
| IL-15 → APRIL | PARTIAL | −0.095 | 0.94 |
| IFN-ω → IFN-β | KNOWN_COREGULATED | −0.074 | 0.88 |
| IL-15 → IL-9 | NOVEL | +0.054 | 1.00 |
| IL-12 → APRIL | NOVEL | −0.054 | 1.00 |
| IL-27 → IL-6 | KNOWN_DIRECTIONAL | +0.048 | 0.94 |
| IFN-β → IFN-γ | KNOWN_COREGULATED | +0.047 | 1.00 |

Labeled-set sign accuracy (the P4 failure): 10/16 correct by raw sign, 6/11 among
non-AMBIGUOUS — the regression vs §26's 15/17. Failures cluster on VEGF pairs
(IL-9/VEGF, IL-6/VEGF, IL-15/VEGF) + IFN-γ/IFN-ω, IL-36-α/IL-9.

## Fixes (both pending)

1. **Donor-level direction null** (decides whether *any* real Group-U signal survives once
   over-power is removed): aggregate cross_asym to donor first, permute/sign-test across
   the ≈10 train donors, then BH/π₀. Cheap — no retraining; re-uses the per-cell-type `M`.
2. **Reproduce the §26 signatures**: diff this run's chunked-encoder signatures vs the
   `binary_ig_all24` ones for the labeled cytokines; if the per-chunk encoders are the
   cause, retrain those cytokines with a single shared encoder (like missing16) and
   re-merge, then re-check P4.

## Bottom line

The end-to-end two-stage pipeline (Path A coupled set → Path B direction over all 121
axes) is *built and runs*, and the pre-registration did its job by flagging that this
instance is not a valid discovery claim. The Group-U question — "are the unlabeled coupled
pairs' directions real?" — remains **open**, pending the donor-level null. Path A's
published 121-axis coupling result and the §26 88% labeled-direction result are unaffected
by this run.
