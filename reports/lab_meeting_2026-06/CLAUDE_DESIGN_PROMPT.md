PASTE THIS INTO CLAUDE DESIGN (attach every file in this folder first).

---

Build a slide deck for a ~25-minute lab-group meeting. Audience: computational-biology
MSc/PhD students and my PI — they know single-cell RNA-seq and machine learning, but nothing
about this specific project. 16:9. Clean, modern academic style — you choose the colors,
fonts, layout, and visual hierarchy.

CRITICAL — use my slide text VERBATIM:
- The file SLIDES.md contains the exact title and body text for every slide.
- Use that text **word-for-word**. Do NOT rephrase, shorten, expand, summarize, or "improve"
  the wording. You may only style/lay it out (bullets, emphasis, spacing, columns).
- Keep my slide order and slide breaks exactly as in SLIDES.md (Slides 1–13 are the talk;
  Slides 14–15 are clearly-marked backup slides — put them after the end).

Figures and tables (attached as files) — place each on the slide where SLIDES.md names it:
- `fig_direction_accuracy.png` → Slide 8
- `fig_cascade_examples.png` → Slide 9
- `fig_sheu_coupling_win.png` → Slide 10
- `table_datasets.xlsx` (or `table_datasets.md`) → Slide 4 — render as a clean table
- `table_results_grid.xlsx` (or `table_results_grid.md`) → Slide 12 — render as a clean table
- `fig_coupling_vs_pathA.png` → optional, Slide 15 (backup)
These figures are final and contain the real data — use them as-is, do not regenerate or alter
the numbers.

Diagrams for YOU to draw (marked `[[ Claude Design: draw ... ]]` in SLIDES.md) — design these
fresh in your own clean style, matching the deck:
- Slide 2: a cascade cartoon — stimulus A → a cell secretes cytokine B → B acts on neighboring cells.
- Slide 5: a "pseudo-tube" — a bag of mixed cells mapping to one stimulus label (e.g. "IL-2").
- Slide 6: a left-to-right flow — binary classifier (cytokine vs PBS) → Integrated Gradients →
  a top-50 gene list labeled `S_X`.
- Slide 7: a small heatmap `M`, then an arrow splitting it into two heatmaps — `M + Mᵀ`
  labeled "coupling (symmetric)" and `M − Mᵀ` labeled "direction (antisymmetric)".

Tone: confident but honest — Slide 11 is a real "limitations" slide and should read as candid,
not as a weakness to hide. Make Slides 8 and 10 (the two result figures) the visual peaks.

Deliver an editable deck.
