# Lab-meeting talk assets (2026-06)

Everything for the lab-group presentation. **One folder — attach all of it to Claude Design.**

## How to use
1. Open Claude Design (or Claude with the slide/artifact tool).
2. **Attach every file in this folder** (the 4 PNGs, the 2 `.xlsx` tables, and `SLIDES.md`).
3. **Paste the contents of `CLAUDE_DESIGN_PROMPT.md`** as your message.
4. The prompt tells Claude Design to use my slide text **verbatim** and place the figures/tables.

## Contents
| File | What it is | Goes on |
|---|---|---|
| `CLAUDE_DESIGN_PROMPT.md` | the prompt to paste | — |
| `SLIDES.md` | **exact slide texts** (verbatim; 13 talk + 2 backup) | all slides |
| `fig_direction_accuracy.png` | headline: 88/86/83% direction + symmetric control | Slide 8 |
| `fig_cascade_examples.png` | recovered textbook cascades (Immune Dictionary) | Slide 9 |
| `fig_sheu_coupling_win.png` | signature coupling recovers what latent geometry missed (Sheu) | Slide 10 |
| `fig_coupling_vs_pathA.png` | two coupling notions disagree (ρ=0.29) — optional backup | Slide 15 |
| `table_datasets.xlsx` / `.md` | the 3 datasets | Slide 4 |
| `table_results_grid.xlsx` / `.md` | method × dataset results grid | Slide 12 |
| `make_talk_assets.py` | regenerates every figure/table from the repo data | — |

## Notes / decisions
- **"Learning dynamics" is intentionally left out of the main arc** — the current method doesn't
  use it, and it would cost a slide explaining a retired concept to an unfamiliar audience. It's
  Slide 14 (backup) only.
- **Honest framing:** the 88/86/83% is *direction on known cascades* (validation), not blind
  discovery; Slide 11 is a candid limitations slide (over-powered cell-level nulls → donor-level
  is the fix). All numbers trace to real results in the repo.
- Conceptual diagrams (cascade cartoon, pseudo-tube, the IG flow, the M-decomposition) are
  **drawn by Claude Design** from specs in `SLIDES.md` — that's design, left to it on purpose.
- Figures regenerate with: `python reports/lab_meeting_2026-06/make_talk_assets.py`.
- Title slide name/lab/date are placeholders — confirm before presenting.
