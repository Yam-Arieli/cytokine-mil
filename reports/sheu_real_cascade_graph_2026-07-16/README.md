# Sheu real-data cascade graph — analog of cascade_forge Fig. 1 + Fig. 3

`sheu_cascade_graph.{png,pdf}` is the real-data counterpart of
`reports/cascade_forge_simulation_1M_cells_2026-07-09/report.pdf` Figures 1 and 3, built
from Sheu 2024 BMDM data instead of a synthetic simulation. No cluster access needed —
both source files are already committed/present locally:

- **Panel A (ground truth)** — the pre-registered TLR cascade wiring for Sheu's 7 stimuli,
  hardcoded from `reports/sheu_cascade/sheu_cascade_labels.yaml`: 7 directional benchmark
  edges (4 NF-κB→TNF, 2 IFN_MUST, 1 IFN_SHOULD) + 4 explicit no-cascade negative-control
  pairs. The remaining 10 of 21 pairs have no literature prior and aren't drawn.
- **Panel B (found graph)** — `cross_asym` calls from
  `results/sheu_cascade/5hr/pathB/per_axis_summary.csv` (the headline Sheu frame per
  CLAUDE.md §26.3), all 21 pairs, oriented by `cross_median` sign, width by
  `|cross_median|`. Colored by category: green = correct on a benchmark pair, red = wrong
  on a benchmark pair, orange dashed = negative-control pair (a sign was still computed —
  the known "direction, not existence" limitation, §26.4), gray dotted = no literature
  prior (uncalibrated).

**Result: 6/7 = 86% benchmark direction accuracy**, matching the CLAUDE.md headline
exactly. The one miss is IFNb↔PIC — the documented polyIC/IFNb signature-collapse failure
(S_polyIC is ISG-dominated ≈ S_IFNb, §26.3/§26.4).

- **Panel C (sorted arrow-bars)** — the Figure-3-bottom-panel analog: all 21 pairs as
  vertical arrows, height = width = `|cross_median|` (the same score and the same
  `width_fn` used for edge width in Panel B), sorted ascending left→right, same
  color/dash encoding as Panel B, tick label below each bar giving the oriented pair.

Regenerate with `python3 make_sheu_cascade_graph.py` from the repo root's Python env
(matplotlib + pandas; no yaml dependency).
