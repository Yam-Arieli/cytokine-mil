#!/usr/bin/env python
"""Emit a TikZ figure of what cascadir RECOVERED on one run (all, eff0.30, t=6).

Every pair flagged coupled by signature_coupling is drawn as a directed edge oriented by
cross_asym, on the same node layout as the authored graph (Figure 1), colored by whether
it is a true cascade or a false positive. Reads a coupling.csv, writes found_graph.tex and
prints a summary. One run/seed only (not overlaid).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "coupling_t6.csv"
OUT = Path(__file__).resolve().parent / "found_graph.tex"       # Figure 3: recovered graph
OUT_BARS = Path(__file__).resolve().parent / "found_bars.tex"   # Figure 4: sorted arrow-bars

# Node positions — identical to Figure 1.
POS = {
    "A": (0, 5), "B": (2, 5), "C": (4, 5), "D": (6, 5),
    "E": (0, 3.6), "F": (2, 3.6), "G": (4, 3.6),
    "H": (0, 1.6), "I": (2.2, 2.4), "J": (2.2, 1.6), "K": (2.2, 0.8),
    "L": (4.4, 2.2), "M": (4.4, 1.0), "N": (6.4, 1.6),
    "O": (9, 4.9), "P": (9, 3.5),
    "Q": (8.4, 1.9), "R": (9.4, 1.9), "S": (8.4, 0.9), "T": (9.4, 0.9),
}
ISOLATED = {"Q", "R", "S", "T"}

# Authored ground truth (directed upstream -> downstream).
DIRECT = [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F"), ("F", "G"),
          ("H", "I"), ("H", "J"), ("H", "K"), ("L", "N"), ("M", "N"),
          ("O", "P"), ("P", "O")]
DIRECT_UND = {frozenset(e) for e in DIRECT}
FEEDBACK = {frozenset(("O", "P"))}


def _reachable():
    adj = {}
    for u, v in DIRECT:
        adj.setdefault(u, []).append(v)
    reach = set()
    for s in POS:
        seen, stack = set(), list(adj.get(s, []))
        while stack:
            n = stack.pop()
            if n in seen or n == s:
                continue
            seen.add(n)
            stack += adj.get(n, [])
        for t in seen:
            reach.add(frozenset((s, t)))
    return reach


REACH_UND = _reachable()


def main():
    df = pd.read_csv(CSV)
    coupled = df[df["coupled"].astype(str).str.lower().isin(["true", "1"])].copy()

    edges = []          # (src, dst, category)
    counts = {"direct": 0, "transitive": 0, "false": 0}
    for _, r in coupled.iterrows():
        a, b = str(r["condition_a"]), str(r["condition_b"])
        ca = float(r["cross_asym"])
        src, dst = (a, b) if ca >= 0 else (b, a)      # orient by cross_asym
        key = frozenset((a, b))
        if key in DIRECT_UND:
            cat = "direct"
        elif key in REACH_UND:
            cat = "transitive"
        else:
            cat = "false"
        counts[cat] += 1
        edges.append((src, dst, cat, float(r["coupling"])))

    coupled_keys = {frozenset((str(r["condition_a"]), str(r["condition_b"])))
                    for _, r in coupled.iterrows()}
    missed = [e for e in DIRECT if frozenset(e) not in coupled_keys
              and frozenset(e) not in FEEDBACK]  # feedback counted once
    # dedupe feedback for missed check
    missed = [e for e in DIRECT if frozenset(e) not in coupled_keys]

    # Edge WIDTH encodes coupling strength (wide range so it is obvious); color = true
    # (green) vs false (orange); dashed = transitive. Widths span the data's min..max.
    cvals = [c for _, _, _, c in edges]
    cmin, cmax = (min(cvals), max(cvals)) if cvals else (0.0, 1.0)
    WMIN, WMAX = 0.2, 5.0   # pt

    def width(c):
        t = (c - cmin) / (cmax - cmin) if cmax > cmin else 0.5
        return round(WMIN + max(0.0, min(1.0, t)) * (WMAX - WMIN), 2)

    color = {"direct": "fgood", "transitive": "fgood", "false": "fbad"}

    def estyle(cat, c):
        dash = ",densely dashed" if cat == "transitive" else ""
        op = ",opacity=0.85" if cat == "false" else ""
        return f"draw={color[cat]},line width={width(c)}pt{dash}{op}"

    # ---- LEFT panel: the recovered graph (unchanged; width encodes coupling) ----
    g = []
    g.append("\\begin{tikzpicture}[baseline=(current bounding box.center),>={Stealth[length=2mm]},")
    g.append("  cnode/.style={circle,draw,thick,minimum size=8mm,fill=methodblue!14,font=\\small},")
    g.append("  inode/.style={circle,draw,thick,dashed,minimum size=8mm,fill=black!6,font=\\small}]")
    for n, (x, y) in POS.items():
        st = "inode" if n in ISOLATED else "cnode"
        g.append(f"  \\node[{st}] ({n}) at ({x},{y}) {{{n}}};")
    order = {"false": 0, "transitive": 1, "direct": 2}   # draw false under, direct on top
    for src, dst, cat, c in sorted(edges, key=lambda e: order[e[2]]):
        conn = " to[bend left=12]" if cat != "direct" else " to"
        g.append(f"  \\draw[->,{estyle(cat,c)}] ({src}){conn} ({dst});")
    y0 = -0.5
    g.append(f"  \\node[font=\\scriptsize,text=black!70,anchor=east] at (1.0,{y0}) "
             f"{{edge width $=$ coupling:}};")
    for i, cv in enumerate([0.02, 0.15, 0.50]):
        xa = 1.4 + i * 2.5
        g.append(f"  \\draw[->,draw=black!60,line width={width(cv)}pt] "
                 f"({xa},{y0}) -- ({xa+1.2},{y0});")
        g.append(f"  \\node[font=\\scriptsize,text=black!60] at ({xa+0.6},{y0-0.34}) {{{cv:.2f}}};")
    g.append("\\end{tikzpicture}")

    # ---- RIGHT panel: the same edges as up-arrows, sorted ascending by coupling ----
    # Height AND width both encode coupling; color/dash carry category. Groups become
    # visible: hairline-short orange (isolated-negative noise floor) -> green dashed
    # (transitive) -> tall fat green (direct cascades). The lone tall-ish orange is M-L.
    HMAX, PITCH = 4.0, 0.16
    srt = sorted(edges, key=lambda e: e[3])              # ascending by coupling
    Xmax = (len(srt) - 1) * PITCH

    def ypos(c):
        return round(c / cmax * HMAX, 3) if cmax else 0.0

    b = []
    b.append("\\begin{tikzpicture}[baseline=(current bounding box.center),>={Stealth[length=1.4mm]}]")
    b.append(f"  \\draw[black!45] (-0.12,0) -- ({Xmax + 0.25:.2f},0);")
    b.append(f"  \\draw[black!45] (-0.12,0) -- (-0.12,{HMAX + 0.25:.2f});")
    for cc in (0.0, 0.25, 0.5):
        b.append(f"  \\draw[black!45] (-0.24,{ypos(cc):.2f}) -- (-0.12,{ypos(cc):.2f}) "
                 f"node[left,font=\\scriptsize,text=black!60,inner sep=1.5pt] {{{cc:.2f}}};")
    b.append(f"  \\node[rotate=90,font=\\small,text=black!70] at (-1.0,{HMAX / 2:.2f}) {{coupling}};")
    b.append(f"  \\node[font=\\scriptsize,text=black!60] at ({Xmax / 2:.2f},-0.55) "
             f"{{recovered edges, sorted}};")
    for i, (src, dst, cat, c) in enumerate(srt):
        x = round(i * PITCH, 3)
        b.append(f"  \\draw[->,{estyle(cat,c)}] ({x},0) -- ({x},{ypos(c)});")
    b.append("\\end{tikzpicture}")

    # Two separate figures now: the recovered graph (Fig 3) and the sorted arrow-bars
    # (Fig 4). Each is scaled to \linewidth on its own, so the bars are no longer squeezed
    # into a shared row (that shared-row scaling was what made them look dense).
    OUT.write_text("\\resizebox{\\linewidth}{!}{%\n" + "\n".join(g) + "\n}\n")
    OUT_BARS.write_text("\\resizebox{\\linewidth}{!}{%\n" + "\n".join(b) + "\n}\n")

    n_direct_truth = len({frozenset(e) for e in DIRECT})   # 11 (O<->P once)
    print(f"coupled pairs returned: {len(coupled)}")
    print(f"  direct cascades recovered : {counts['direct']} (of {n_direct_truth} authored direct)")
    print(f"  transitive (indirect real): {counts['transitive']}")
    print(f"  FALSE positives           : {counts['false']}")
    print(f"  missed direct edges       : {sorted(set(frozenset(e) for e in missed))}")
    print(f"[wrote] {OUT}")
    print(f"[wrote] {OUT_BARS}")


if __name__ == "__main__":
    main()
