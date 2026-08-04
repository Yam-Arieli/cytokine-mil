# Labeling protocol for the Oesinghaus directional benchmark

**What this file is.** The rule set behind `cytokine_axes_audited.csv` — the answer key the
88% (15/17) direction result is scored against. It documents how each pair's expected
upstream→downstream order was assigned, so the key can be checked independently of the person
who built it.

**Provenance caveat, stated up front.** The original rule document was written to
`~/.claude/plans/no-job-is-running-glittery-eagle.md` (referenced at `audit_decisions.yaml:4`)
and was never committed; it no longer exists on disk. This file is a **post-hoc
reconstruction** from the committed decision records — `audit_decisions.yaml`,
`audit_decisions_validated.yaml`, `audit_digest.csv` and `audit_log.md`. The label→rule
mapping is quoted from `audit_log.md:3`; the pair-status mapping was re-derived empirically
from the 53 committed rows and holds without exception. This is a faithful reconstruction of
a rule set the data shows was applied consistently. It is **not** a pre-registration and must
not be described as one.

---

## 1. Evidence source

`audit_digest.csv` — 53 rows (one per evaluable axis) × 20 columns. For each axis it holds,
**separately for each of the two directions**:

- `{dir}_summary` — the literature summary for that direction
- `{dir}_n_primary`, `{dir}_n_secondary` — primary and secondary citation counts
- `{dir}_citations_primary`, `{dir}_citations_secondary` — the citation lists
- `{dir}_n_lit_entries`, `{dir}_T_list` — supporting entries and implicated cell types

The 53 axes are the pairs for which both cytokines had a discovered binary-IG signature, drawn
from the 121 coupled axes in `cytokine_axes.csv`. That eligibility constraint is a coverage
limit, not a labeling decision, and is disclosed in the thesis where the benchmark is
introduced.

## 2. Per-direction labels

Each direction is labeled on its own, against its own evidence (`audit_log.md:3`):

| Rule | Label | Meaning |
|---|---|---|
| R1 | `POSITIVE_STRONG` | Direct, well-supported induction of the target by the source |
| R2 | `POSITIVE_WEAK` | Induction reported but thinly or indirectly supported |
| R3 | `INHIBITORY` | The source is documented to *suppress* the target |
| R4 | `UNKNOWN` | No evidence found for this direction |
| R5 | `POSITIVE_STRONG` | As R1, for a pre-registered pair |

Each label in `audit_decisions.yaml` carries the quote it was read from and the citation
counts backing it, so every call is traceable to its evidence.

## 3. Pair status, derived mechanically

The labeler never scores a *pair*. `pair_status` and `expected_sign` follow deterministically
from the two independent per-direction labels. Re-derived from all 53 committed rows:

| a→b label | b→a label | `pair_status` | n | graded? |
|---|---|---|---:|:-:|
| `POSITIVE_STRONG` | `UNKNOWN` | `DIRECTIONAL_a_to_b` (`expected_sign = +1`) | 10 | **yes** |
| `UNKNOWN` | `POSITIVE_STRONG` | `DIRECTIONAL_b_to_a` (`expected_sign = −1`) | 7 | **yes** |
| `POSITIVE_WEAK` | `UNKNOWN` | `WEAK_a_to_b` | 4 | no |
| `UNKNOWN` | `POSITIVE_WEAK` | `WEAK_b_to_a` | 3 | no |
| `POSITIVE_WEAK` | `POSITIVE_WEAK` | `LOW_CONFIDENCE` | 2 | no |
| `INHIBITORY` on either side | — | `PARTIAL_INHIBITORY` | 7 | no |
| `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | 20 | no |

**The graded criterion, in one line: a pair enters the benchmark only if one direction has
strong literature support and the other has none.** That yields the 17-pair benchmark
(`counts_in_benchmark = True`). No row in the 53 has `POSITIVE_STRONG` on *both* sides; had
one, it would be a documented feedback loop and excluded, which is how IL-12 ↔ IFN-γ is
handled in the Immune Dictionary benchmark.

## 4. Independence from the model output

The construction blocks the obvious circularity in three places:

1. Directions are labeled **separately**, each against its own literature evidence. The
   labeler is never presented with a pair to adjudicate.
2. `expected_sign` is **derived**, not chosen, by the table in §3.
3. Every call carries its quote and citation counts, so a reader can check the biology per
   pair rather than trust the procedure.

A reviewer pass over all 53 axes is recorded in `audit_decisions_validated.yaml`
(`reviewer_action: agree`, 53 of 53).

## 5. The seven revised labels

The audit replaced an earlier keyword-parsed key (`original_direction`, extracted
automatically from literature summaries). The two disagree on 7 of the 17 graded pairs. In
**all seven**, one direction is cited and the other explicitly has no evidence — so these
correct a parse error rather than pick a side in a live dispute.

| pair | old key | audited | cited direction (quote, citations) | opposite direction |
|---|---|---|---|---|
| IFN-γ / IFN-ω | IFN-γ up | **IFN-ω up** | "IFN-omega specifically enhances cytolytic CD8+ T cell IFN-gamma responses" (2 primary, 1 secondary) | UNKNOWN, 0 cites |
| IFN-γ / IL-2 | IFN-γ up | **IL-2 up** | "IL-2 is well-established to induce IFN-gamma production in CD8+ T cells via STAT5/T-bet signaling" (1 primary, 1 secondary) | UNKNOWN, 0 cites |
| IL-13 / VEGF | IL-13 up | **VEGF up** | "VEGF induces IL-13-dependent asthma-like phenotype in transgenic mice" (3 primary) | UNKNOWN, 0 cites |
| IL-15 / VEGF | VEGF up | **IL-15 up** | "IL-15 signaling via JAK-STAT5 drives VEGF-A expression in NK cells" (1 primary, 1 secondary) | "No published evidence VEGF induces IL-15" |
| IL-17A / IL-36-α | IL-17A up | **IL-36-α up** | "IL-36 (including IL-36alpha) activates Th17 cells and induces IL-17A expression" (2 primary, 1 secondary) | UNKNOWN, 0 cites |
| IL-36-α / IL-9 | IL-9 up | **IL-36-α up** | "IL-36 signaling (via IL-36R/MyD88/NF-kB) drives Th9 differentiation and IL-9 production in CD4+ T cells" (2 primary) | UNKNOWN, 0 cites |
| IL-9 / VEGF | VEGF up | **IL-9 up** | "IL-9 induces VEGF secretion from mast cells" (1 primary) | "No evidence VEGF induces IL-9 in monocytes" |

Quotes are verbatim from `audit_decisions.yaml`.

## 6. Robustness: the result does not rest on the revisions

Splitting the 17-pair benchmark by whether the audit changed the label
(`retally_pipeline_against_audit.py --metric cross_asym`):

| subset | accuracy |
|---|---|
| pairs the audit **did not** touch | **9 / 10 = 90%** (miss: IL-6/VEGF) |
| pairs the audit **revised** | 6 / 7 = 86% (miss: IL-13/VEGF) |
| full benchmark | 15 / 17 = 88% |

Accuracy is the same either side of the revision, so the headline is not produced by it. If
the audit had been fitted to the result, success would concentrate in the revised subset.

For completeness, the counterfactual: scoring the same pairs under the **original keyword
key** gives 9/16 = 56% (16 of the 17 carry an original directional tag; IL-13/TL1A was tagged
`coregulated_other`), against 14/16 under the audited key. That gap is the automated parse
being wrong on seven pairs where the literature is one-sided — see §5 — not the method's
output changing.

## 7. The excluded pairs

The 13 pairs demoted out of the graded set are the audit's second knob and deserve their own
disclosure. Seven of them (`WEAK_*`) carry a directional expectation and can be scored:
**4/7 = 57%**, near chance.

Two things to say about that honestly. First, the method signs `+` on all seven, so 4/7
reflects the mix of expected signs in that bucket rather than any discrimination. Second, the
reading we favour: `WEAK` marks pairs where the *literature* evidence is thin, so the answer
key is itself unreliable there, and measured accuracy must decay toward chance whether or not
the method is right. That is a reading, not a proof — the bucket cannot distinguish a noisy
key from a method that fails on weakly-coupled pairs.

## 8. Reproducing these numbers

```bash
python scripts/retally_pipeline_against_audit.py \
    --metric cross_asym \
    --pipeline_csv results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv \
    --out /tmp/retally_check.md
```

Expected: headline 15/17, weak bucket 4/7, original-tag 15/29, 7 tags flipped. The per-axis
table carries a `tag_changed` column; grouping by it reproduces the 9/10 and 6/7 split in §6.
The packaged `cascadir/examples/analysis/oesinghaus_per_axis.csv` carries the same per-axis
calls (`.claude/skills/cascadir-values/references/result_files.md`).

**Do not hand-derive `cross_asym` from `per_celltype.csv`** — go through the script above or
`cascadir`, per the `cascadir-values` skill.
