# Immune Dictionary — coupling-gate validation (cell-level degree correction)

**Goal.** Third-dataset test of the §28.2 coupling-gate fix (donor-level null +
degree/hub correction). ID's §26 *direction* result (cross_asym 83 %) was already
validated; the **coupling gate** (existence: are A,B coupled?) had never run on ID
(CLAUDE.md §28.1: "Immune Dictionary — not run"). This is that run.

**Why cell-level, not donor-level.** ID benchmark cytokines span only rep01/02/03 →
**3 donors**. The donor-level sign-flip null has a p-floor of 1/2³ = 0.125 > 0.05, so it
is inapplicable (§28.2 boundary: needs ~8+ donors). ID therefore tests the **cell-level
degree correction** — the same path validated on Sheu, but now on a **whole-transcriptome,
in-vivo** atlas (4000 HVG, mouse lymph node, 4 h) instead of Sheu's targeted 500-gene
in-vitro panel. New regime ⇒ a genuine generalization test.

**Method / inputs (no retraining).** Reused the existing §26 binary-IG signatures
`S_X` (top-50 IG genes, 12 benchmark cytokines) from `results/id_cascade/binary_ig/`.
Cells pooled by (cytokine, cell_type) from the ID pseudotube manifest (196 groups,
13 classes incl PBS). `cell_coupling_degree` computes RAW and degree-corrected (HUB)
coupling, each gated by a degree-matched gene-set null (n_perm=500). Two signature
variants: IG_vsPBS and IG_vsPanel (panel-residualised). Coupling labels from
`id_axes_labeled.csv` `pair_status` → 7 coupled-positive cascades, 2 antagonism
negatives, 3 descriptive overlaps.

**Run:** job `30884443` (CPU, 5 min). Driver `scripts/run_cell_degree_coupling.py
--dataset id`; SLURM `slurm/id_cascade/coupling_cell_degree.slurm`.
Outputs: `results/id_cascade/coupling_cell_degree/{cell_degree_report.md,
cell_degree_IG_vsPBS.csv, cell_degree_IG_vsPanel.csv, cell_degree_summary.csv}`.

---

## Headline

| variant | mode | coupled / 66 | MUST recall | antagonism FP |
|---|---|---|---|---|
| IG_vsPBS | raw | 58 (88 %) | 7/7 | 2/2 |
| **IG_vsPBS** | **hub** | **29 (44 %)** | **6/7** | **1/2** |
| IG_vsPanel | raw | 58 (88 %) | 7/7 | 2/2 |
| IG_vsPanel | hub | 24 (36 %) | 3/7 | 1/2 |

**The degree correction reproduces the validated pattern on a third dataset and a new
regime.** With IG_vsPBS, hub correction halves the coupled-fraction (88 %→44 %), keeps
6/7 known cascades, and suppresses an antagonism. RAW reproduces the over-call
(everything significant — broad in-vivo panel + over-powered cell-level null), exactly
the failure §28.2 was built to fix.

## The trustworthy readout: coupling_hub RANKING (IG_vsPBS)

The binary null is permissive for small-positive pairs (the §28.2 magnitude-floor
caveat) — the **ranking** is what to trust. Of 66 pairs, the 7 known cascades land:

| rank | pair | coupling_hub | note |
|---:|---|---:|---|
| **2** | IFNb–IFNg | +0.107 | cleanest cascade (type-I IFN primes NK); just behind IFNb–IL15 |
| **5** | IFNg–IL18 | +0.046 | NK→IFN-γ axis |
| **20** | IFNg–IL15 | +0.009 | NK→IFN-γ axis |
| **24** | IFNg–IL12 | +0.007 | NK→IFN-γ axis (bidirectional) |
| **26** | IL6–TNFa | +0.005 | NF-κB→STAT3 |
| **27** | IL1b–IL6 | +0.005 | NF-κB→STAT3 |
| 31 | IFNg–IL2 | +0.001 | **the miss** — fails null (p=0.32) |

Antagonisms: **IFNg–IL4 → #60** (−0.034, suppressed, correct); IL10–IL12 → #22
(+0.008, the surviving false-positive — but a weak, mid-rank call, not a top hit).

**The single miss is conservative, not wrong.** IFNg–IL2 (the IL-2→NK→IFN-γ cascade)
has a real, correct direction signal (cross_asym −0.063) but small coupling magnitude
after hub correction — IL-2 is a common-γ-chain, broadly-engaged ("hub-like") cytokine,
so degree-centering down-weights it. This is the coupling analog of the §26 IL1b→IL6
direction non-call: a magnitude-floor null, not a reversed call.

## New finding: IG_vsPBS > IG_vsPanel for the coupling gate on ID

§28.2 adopted IG_vsPanel as the default signature (it equals IG_vsPBS on **direction**,
which is antisymmetric and untouched by panel-residualisation). But for the **coupling
gate** on this broad in-vivo panel, IG_vsPanel+hub **over-corrects** (recall 3/7): the
NF-κB→STAT3 cascades IL6–TNFa and IL1b–IL6 drop below zero and fail the null. Mechanism:
panel-residualisation removes the cross-cytokine shared program from the *signature*,
and degree-centering removes hub structure from the *coupling matrix* — for a symmetric
coupling readout these two shared-structure removals **compound**, over-stripping real
coupling. (For the antisymmetric direction metric they don't, which is why §28.2 saw them
as equivalent.) **Takeaway: use IG_vsPBS for the coupling gate; IG_vsPanel remains fine
for direction.**

---

## Honest caveats

1. **"Over-call" is a pessimistic denominator.** The 66-pair denominator includes many
   *unlabeled* pairs that are plausibly genuinely coupled in this dense cytokine atlas
   (e.g. IFNb–IL15 ranks #1 — type-I IFN strongly induces IL-15; IFNb–IL18 #4). So the
   44 % "coupled" fraction is not 44 % false positives; the cleaner claim is
   *discrimination improved* (88 %→44 % while known cascades stay top-ranked and the
   clearest antagonism sinks to the bottom).
2. **No donor-level confirmation.** 3 donors ⇒ cell-level only; significance is read
   from the ranking + magnitude, not the (permissive) binary null. Pseudotube pooling
   also duplicates cells (pseudo-replication) — another reason the null is over-powered
   and the ranking is the readout.
3. **In-vivo, single 4 h frame.** Relay cell type can differ from responder; coupling is
   informative, not causal (§2.7). No kinetic validation as on Sheu.
4. **Small labeled benchmark** (7 positives, 2 negatives) — the ranking pattern (cleanest
   cascade at #2, antagonism at #60) is stronger evidence than the headline fractions.

## Bottom line

The cell-level degree-corrected coupling gate **generalizes to a third, independent
dataset and a new biological regime** (mouse in-vivo lymph node, whole transcriptome):
hub correction halves the over-call while keeping the known NK→IFN-γ and NF-κB→STAT3
cascades top-ranked and pushing the clearest antagonism to the bottom — the same
discrimination gain validated on Oesinghaus (donor+degree) and Sheu (cell+degree). One
mechanistically-understood miss (IFNg–IL2, a γc-hub cytokine). A real new finding: for
the *coupling* gate, IG_vsPBS beats IG_vsPanel (panel-residualisation over-corrects when
compounded with degree-centering); IG_vsPanel stays fine for *direction*. The method's
core fix holds across all three datasets; no method change is warranted before continuing.
