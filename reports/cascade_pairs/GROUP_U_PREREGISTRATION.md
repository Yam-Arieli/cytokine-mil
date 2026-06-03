# Pre-registration — Full Path A→B + Group-U direction FDR (Oesinghaus)

**Locked before any audit/FDR script runs, per CLAUDE.md §25.1 / §27.4.**
Commit this file to `main` BEFORE submitting the DAG. Running
`run_group_u_fdr.py` before this commit is a protocol violation.

Date locked: 2026-06-03. Dataset: Oesinghaus 24h PBMC (Path A works here; this is
Oesinghaus-only — Sheu's Path A gate FAILED, ID's emitted nothing, so the two-stage
claim is not made for them and their §26 direction-only results are unchanged).

---

## 1. Question

Two holes in the §26 `cross_asym` result that this run closes:

1. **De-circularise the two-stage pipeline.** §26 evaluated direction on a curated /
   audit-derived shortlist (17 labeled pairs; 53 pairs directed total). Path A's coupling
   call was never actually used as the *input gate* to Path B end-to-end. Here we direct
   **all 121 Path A coupled axes** (after training binary IG signatures for the ~24
   missing cytokines), so the pipeline is "Path A finds 121 coupled pairs → Path B
   directs all of them."

2. **Characterise Group U.** The pairs Path A couples but for which we have **no
   directional prior** (~104 of 121) currently get a confident `cross_asym` sign that is
   never scored. We quantify how many carry a *statistically reliable* direction via a
   direction-specific permutation null + FDR.

## 2. Method (frozen)

- **Signatures `S_X`:** top-50 genes by binary-MIL Integrated Gradients (PBS-mean
  baseline, 20-step), per cytokine, wide config (embed=512, hidden=(512,512), attn=128,
  Stage1 20@0.005, Stage2 250@3e-5). Missing cytokines = `cytokine_axes.csv` cytokines −
  `binary_ig.parquet` cytokines (computed at runtime; expected ≈24).
- **Direction call:** `cross_asym(a,b) = s(a,S_b) − s(b,S_a)` (PBS-normalised), median
  across cell types, sign = direction (`+` ⇒ axis_a upstream). Identical to §26.
- **Direction-permutation null** (`cytokine_mil/analysis/direction_null.py`): hold
  `S_a,S_b` fixed; within each cell type permute a/b cell-condition labels (preserve
  counts), recompute `cross_asym`, aggregate median across cell types.
  `n_perm = 1000`, `direction_null_seed = 123`.
  Recentred two-sided empirical p:
  `p_emp = mean_k(|null_k − null_centre| ≥ |observed − null_centre|)`,
  `null_centre = mean(null_k)` (the `S_a`/`S_b` magnitude offset is a nuisance; the label
  effect is `observed − null_centre`).
- **Donor scope:** train donors only (`--exclude_donors Donor2 Donor3`), consistent with
  the binary models' training split, matching the §26 evaluation.
- **FDR:** Benjamini–Hochberg on Group-U `dir_p_emp` (manual; no scipy). Storey π₀ at
  λ=0.5: `π0 = #(p > 0.5) / (0.5·m)`, clamped to ≤ 1. Report n significant at q∈{0.05, 0.10}.

## 3. Locked thresholds

| parameter | value |
|---|---|
| `n_perm` (direction null) | **1000** |
| `direction_null_seed` | **123** |
| FDR levels reported | **q = 0.05 and q = 0.10** |
| confident-hypothesis def | `dir BH-q ≤ 0.10` **AND** `cross_consensus ≥ 0.7` **AND** `|cross_median| ≥ P25(labeled-positive |cross_median|)` |
| top-K for literature back-validation | **10** |
| labeled set | `counts_in_benchmark == True` in `cytokine_axes_audited.csv` |
| Group U | evaluable Path A axes NOT in the labeled set |

The P25 |cross_median| threshold is computed from the labeled positives **in this run**
(not hard-coded), so it self-calibrates to the run; it is locked as a *rule*, not a number.

## 4. Pre-registered predictions

- **P1 — power.** ≥ **80%** of labeled, non-AMBIGUOUS positives pass the direction null
  (`dir_p_emp` significant at BH-q ≤ 0.10). If the null lacks power even on knowns, the
  Group-U FDR is uninterpretable.
- **P2 — specificity.** The known near-zero / miss pairs (e.g. the §26 VEGF misses) and
  any truly symmetric pairs do **not** pass the null. (Note: overlap/coregulation pairs
  with a real magnitude asymmetry **may** pass — that is expected; the null tests
  "reliable directional asymmetry", not "cascade". Existence is Path A's job.)
- **P3 — headline (discovery-capable vs confirmation-only).** Group-U **π₀ < 0.9** —
  i.e., **> 10%** of the ~104 unknown coupled pairs carry a reliable directional
  asymmetry above the permutation null. This is the decisive prediction:
  - π₀ < 0.9 (Group U shifted above null) ⇒ the method is **discovery-capable**; we
    report "K novel directional hypotheses at FDR q" + the ranked list.
  - π₀ ≈ 1 (Group U indistinguishable from null) ⇒ the method is **confirmation-only**
    (recovers knowns, no evidence of novel-direction signal). We report this honestly.
- **P4 — regression.** Adding the ~24 cytokines does not change the labeled-subset
  `cross_asym` accuracy materially (within ±1 call of the §26 15/17 on the audited
  benchmark; the labeled pairs and their signatures are unchanged except for the shared
  encoder re-train, which is a known minor source of variation).

## 5. What this run does NOT claim

- Not causation (interventional data needed; §26.4).
- Not coupling existence — `|cross_asym|` and `dir_p_emp` do NOT decide whether a pair is
  coupled. Path A (latent geometry, FDR-controlled) is the existence gate; Group U is, by
  definition, already coupled by Path A.
- Not de-circularisation of the *coupling recall* claim — that needs an independent
  known-coupled list (out of scope here; logged as a follow-up).

## 6. Outputs

- `results/group_u/pipeline_full121/per_axis_summary.csv` — 121 axes, `cross_median`,
  `cross_consensus`, classification, `dir_p_emp`, gene-set-null `cross_p_emp_two_sided`,
  `dirscore_*` (symmetric control).
- `reports/cascade_pairs/GROUP_U_RESULTS.md` — P1–P4 verdicts, π₀, BH counts, ranked
  top-K Group-U hypotheses, regression check.
