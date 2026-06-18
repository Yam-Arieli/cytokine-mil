# COVID-progression RED result — adversarial verification verdict (§30)

Independent multi-agent verification of the RED result (run 30866717–30866720), over the
local artifacts (signatures, direction table, per-cell-type, apparatus, figures). Three
high-confidence checks + synthesis. **Bottom line up front: the method is correct, but the
signatures were confounded — this was NOT a clean test, so no conclusion about
disease-progression direction can be drawn from this run.**

## What the three checks found (all high confidence)

1. **Signatures are confounded (not grade-specific).** 6 genes appear in ALL 5 grade
   signatures — the proliferation/cell-cycle quintet `MKI67/TYMS/RRM2/NUSAP1/CENPF` (+`UBE2C`)
   plus CITE-seq tag `AB_SELP`. Top-50 signatures are 8–21 proliferation genes and 6–20
   `AB_*` antibody (ADT) tags; the canonical COVID ISG axis is nearly absent (ISG15 in only
   3/5). Critical's top genes are erythroid/hemoglobin (`HBA1/HBB/HBA2` = RBC ambient-RNA).
   → grade-vs-Healthy IG latched onto shared activation/proliferation + protein tags, not
   grade biology.

2. **cross_asym carries no severity-direction signal.** Magnitudes are near-zero
   (|median| 0.005–0.062); both engagement terms are large/positive/nearly equal
   (`sA_PB`≈0.078, `sB_PA`≈0.110) and cross_asym is their tiny residual. Decisively, the sign
   follows **alphabetical pair-ordering, not severity and not activation magnitude** (the
   Critical pairs prove it: the more-severe Critical does not dominate). Signed accuracy 5/10
   = chance; the SYMMETRIC `directional_score` control *beats* it (7/10). Not a sign bug.

3. **The machinery is validated; the COVID result is the `monotone_noseed` regime.** The
   synthetic apparatus PASSED: distinct-program ladder cross_accuracy=100% (all 6 signs
   correct, τ=+1.0) with the symmetric control at 50%; `monotone_noseed` FLIPS (τ=−1.0);
   `monotone_seeded` partially recovers (67%). COVID's `cross 44% < control 67%`, CI spanning
   chance, is the `monotone_noseed` fingerprint — a magnitude-intensity axis with no
   antisymmetric directional seed. The heatmap is uniformly positive (shared inflammation),
   and Mild/Moderate cells fully overlap in the signature scatter.

---

## Verification verdict (adversarial)

**(1) Verdict: the RED is genuine no-signal, but this was NOT a clean test of the
disease-progression-direction hypothesis.** Two things are both true and must be kept
separate. The *machinery* is correct — the synthetic apparatus PASSED: on the
distinct-program ladder `cross_asym` recovers the planted order at 100% (all 6 signs match
the oracle, Kendall τ=+1.0) while the symmetric `directional_score` control sits at 50%,
proving the sign convention is sound with no code/sign bug. So this is **not** a fixable
plumbing artifact. But the *input signatures* are confounded: the COVID result
(`cross_accuracy=0.44 < symmetric control 0.67`, Kendall τ=−0.2, donor-bootstrap CI
[0.20,0.70] spanning chance) sits squarely in the apparatus's `monotone_noseed` regime — a
monotone-intensity axis with no antisymmetric directional seed for `cross_asym` to read. We
therefore **cannot conclude anything about whether a snapshot encodes progression
direction**: the method neither succeeded nor was given a fair test.

**(2) Mechanism (the §30 magnitude-confound, realized).** Binary IG (grade-vs-Healthy)
latched onto the largest *generic* correlate of COVID rather than grade-specific biology, so
the five `S_grade` are near-identical: 6 genes appear in all 5 signatures — the
proliferation/cell-cycle quintet `MKI67/TYMS/RRM2/NUSAP1/CENPF` (+`UBE2C`) plus the CITE-seq
tag `AB_SELP` — and top-50 Jaccard rises with clinical adjacency (up to 0.39) purely through
this shared proliferation+AB core. The canonical COVID ISG axis is nearly absent (ISG15 in
only 3/5). Because `S_a ≈ S_b` on the shared program, both cross-engagement terms are large,
positive, and nearly equal (`sA_PB`≈0.078, `sB_PA`≈0.110), and `cross_asym` is their tiny
residual (|median| 0.005–0.062). That residual does not track severity (5/10 = chance) —
decisively, the Critical pairs show the alphabetically-*later* (= *less* severe) grade
engaging more, so the sign follows **alphabetical pair-ordering, not severity and not
activation magnitude**. The over-powered cell-level null (§27.6) then stamps these ~0.05
offsets as "significant" (`null_p=0.0`), and high per-cell-type sign consensus reflects a
*consistent small bias in the wrong coordinate*, not a strong directional signal. Two further
non-immune confounds ride along: Critical's top genes are erythroid/hemoglobin
(`HBA1/HBB/HBA2` = RBC ambient-RNA contamination) and 6–20 of each signature's top-50 are
CITE-seq `AB_` protein tags, so the "signature" is partly protein abundance, not a
transcriptional program.

**(3) Single most important flaw + concrete fix.** The flaw is the **signature-construction
step**: grade-vs-Healthy IG on a monotone-intensity axis produces signatures dominated by a
shared proliferation/cell-cycle program (plus ADT and RBC contamination) instead of
grade-distinguishing biology — violating the `S_a ≠ S_b` distinct-program precondition the
method requires. Concrete fix, in order of leverage:
- **Exclude non-transcriptional / contamination features before IG:** drop all `AB_*`
  CITE-seq ADT tags, and mask hemoglobin/erythroid genes
  (`HBA1/HBA2/HBB/HBM/AHSP/BLVRB/SLC25A37`).
- **Remove the shared proliferation axis:** regress out or exclude the cell-cycle module
  (`MKI67/TYMS/RRM2/NUSAP1/CENPF/UBE2C/TOP2A/STMN1/ASPM/PTTG1/TUBB/TUBA1B`), e.g. via Scanpy
  cell-cycle scoring, or composition-match cycling-cell fractions across grades.
- **Build grade-distinguishing signatures, not grade-vs-Healthy:** use adjacent
  grade-vs-grade contrasts (or a severity-residualized IG) so `S_grade` captures what changes
  *between* grades rather than the shared "this is activated COVID" magnitude.
- **Replace the over-powered null with the donor-level null (§16/§27.6)** so a ~0.05 offset is
  no longer auto-significant.

**(4) Bottom line.** Do not conclude anything about disease-progression direction from this
run — the apparatus says the method works but the signatures were confounded; re-run a
cleaned version (AB_/cell-cycle/RBC removed, grade-vs-grade signatures, donor-level null)
before any claim, positive or negative.
