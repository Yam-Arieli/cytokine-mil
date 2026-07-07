# Thesis

Paper-style LaTeX scaffold for the M.Sc. thesis. Write the content yourself —
each section currently holds only its title and a `% >>> write here` marker.

## Layout

```
thesis/
  thesis.tex       <- main file: preamble + title page + section titles
  references.bib   <- bibliography (seeded with the core datasets)
  figures/         <- put figures here (graphicspath points at it)
  README.md
```

## Build

With `latexmk` (recommended):

```bash
latexmk -pdf thesis.tex
latexmk -c            # clean aux files
```

Or manually (bibtex pass for the bibliography):

```bash
pdflatex thesis
bibtex   thesis
pdflatex thesis
pdflatex thesis
```

## Notes

- **Fonts:** uses `newtxtext` / `newtxmath` (Times-like). If your TeX install
  lacks them, comment those two lines in the preamble and uncomment
  `\usepackage{lmodern}`.
- **Engine:** compiles with `pdflatex`. If you later want custom `.otf`/`.ttf`
  fonts, switch to `xelatex`/`lualatex` + `fontspec` (drop `fontenc`/`newtx`).
- **Citations:** add entries to `references.bib`, cite with `\citep{key}` /
  `\citet{key}`. The default VS Code on-save recipe is **`pdflatex x2`** (no
  bibtex — it would error on an empty bibliography). Once you add your first
  `\cite{...}`, build once with the **`pdflatex > bibtex > pdflatex x2`** recipe
  (Cmd+Shift+P → "LaTeX Workshop: Build with recipe") so the references resolve.
- **Optional packages:** `pgfplots` (TikZ plots, as in `reports/method_summary`)
  and `cleveref` (`\cref`) are pre-wired but commented out in the preamble
  because the minimal TeX Live `basic` scheme lacks them. Enable with
  `tlmgr install pgfplots cleveref` then uncomment the relevant lines.
- Fill in the `\todo{...}` placeholders on the title page.
