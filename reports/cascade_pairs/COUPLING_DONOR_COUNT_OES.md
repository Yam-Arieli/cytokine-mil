# Oesinghaus donor-count test — is the vsPBS/vsPanel COUPLING preference about donor COUNT or the dataset?

**Verdict (DATASET):** **NOT donor count — the ID divergence was dataset-specific.** The 3-donor Oes groups still favor IG_vsPanel (3/3). Retract 'few donors → vsPBS'; the ID result (vsPBS best) reflects ID's biology/panel, not donor count.

Held constant: dataset (Oes), method (cell-level degree coupling), signatures (binary_ig_all24, all donors), benchmark (cytokine_axes_audited.csv, 17 counts_in_benchmark positives). Varied: donors entering the coupling. Primary readout = coupling_hub RANKING of the benchmark pairs (binary null is permissive — trust the rank). Job 30885755 (array 0-3, cell-level; race-fix in oesinghaus_cell_loader temp-manifest naming, commit e6bb0c3).

## Per-run variant winner (by benchmark-pair mean coupling_hub rank; lower rank = ranked higher = better)

| run | vsPBS_mean_rank | vsPanel_mean_rank | vsPBS_recall | vsPanel_recall | vsPBS_overcall | vsPanel_overcall | winner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_all10 | 89.9412 | 84.8235 | 11 | 13 | 0.4674 | 0.4855 | vsPanel |
| low_g1 | 102.3529 | 88.8235 | 11 | 13 | 0.4565 | 0.4783 | vsPanel |
| low_g2 | 81.1765 | 77.4706 | 11 | 13 | 0.4601 | 0.4529 | vsPanel |
| low_g3 | 90.0588 | 83.5882 | 11 | 12 | 0.4384 | 0.4746 | vsPanel |

LOW groups favoring vsPBS: **0/3**; favoring vsPanel: **3/3**. HIGH (anchor): **vsPanel**.

## Full metrics (HUB mode)

| run | variant | n_pairs | n_pos | mean_rank_pos | median_rank_pos | pos_in_top20 | recall_hub | overcall_hub |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_all10 | IG_vsPBS | 276 | 17 | 89.9412 | 75.0000 | 4 | 11 | 0.4674 |
| high_all10 | IG_vsPanel | 276 | 17 | 84.8235 | 55.0000 | 4 | 13 | 0.4855 |
| low_g1 | IG_vsPBS | 276 | 17 | 102.3529 | 71.0000 | 3 | 11 | 0.4565 |
| low_g1 | IG_vsPanel | 276 | 17 | 88.8235 | 76.0000 | 4 | 13 | 0.4783 |
| low_g2 | IG_vsPBS | 276 | 17 | 81.1765 | 42.0000 | 5 | 11 | 0.4601 |
| low_g2 | IG_vsPanel | 276 | 17 | 77.4706 | 44.0000 | 5 | 13 | 0.4529 |
| low_g3 | IG_vsPBS | 276 | 17 | 90.0588 | 80.0000 | 4 | 11 | 0.4384 |
| low_g3 | IG_vsPanel | 276 | 17 | 83.5882 | 60.0000 | 4 | 12 | 0.4746 |

## Interpretation

Across every Oesinghaus condition — the 10-donor cell-level anchor and all three disjoint
3-donor groups — IG_vsPanel ranks the known cascades higher and recovers more of them than
IG_vsPBS, with comparable over-call. This **matches the §28.2 donor-level Oes result**
(vsPanel best) and is unchanged by dropping to the ID's 3-donor regime. **Donor count is
therefore not the lever.** The ID coupling run's vsPBS preference (recall 3/7 for vsPanel)
is a property of that dataset, not of having few donors.

**Honest scope.** This experiment held the Oes **signatures** fixed (all-donor binary_ig_all24)
and varied only the donors whose cells enter the coupling. It therefore rules out
**coupling-cell donor count** as the driver. It does **not** rule out a different confound:
ID's signatures were themselves trained on few donors (2 train + 1 val, wide config), so
"signature-estimation noise at low donor count" remains a candidate distinct from "dataset
biology". Disentangling those would require retraining Oes signatures on 3 donors and
repeating — the rejected branch of this plan. Leading hypotheses for ID's vsPanel
over-correction: ID is in-vivo (paracrine cross-cytokine signal inflates every cell's
cross-engagement → panel-residualisation strips more), and its 86-cytokine atlas defines a
very different "shared program" mean than Oes's PBMC panel.

## Practical upshot

Keep **IG_vsPanel as the coupling default** (§28.2) — it is robust on Oesinghaus across
donor counts and at the donor level. Treat ID's vsPBS preference as a **dataset-specific
exception**, not a donor-count rule. Do not gate the variant choice on donor count.

## Caveats
- Reused all-donor signatures → isolates the donor-count effect on the coupling PATH; does NOT test signature degradation at low donor count.
- Cell-level null is permissive → ranking is the readout, not recall_hub.
- 3 LOW groups = a consistency check (3 disjoint donor triples), not a CI.
- recall_hub denominator = 17 benchmark-positive pairs (matches the §28.2 donor-level Oes 11/17).
