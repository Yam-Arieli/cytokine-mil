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
OUT = Path(__file__).resolve().parent / "found_graph.tex"

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
    lines = []
    lines.append("\\begin{tikzpicture}[>={Stealth[length=2mm]},")
    lines.append("  cnode/.style={circle,draw,thick,minimum size=8mm,fill=methodblue!14,font=\\small},")
    lines.append("  inode/.style={circle,draw,thick,dashed,minimum size=8mm,fill=black!6,font=\\small}]")
    for n, (x, y) in POS.items():
        st = "inode" if n in ISOLATED else "cnode"
        lines.append(f"  \\node[{st}] ({n}) at ({x},{y}) {{{n}}};")
    # draw false first (under), then transitive, then direct (on top)
    order = {"false": 0, "transitive": 1, "direct": 2}
    for src, dst, cat, c in sorted(edges, key=lambda e: order[e[2]]):
        dash = ",densely dashed" if cat == "transitive" else ""
        op = ",opacity=0.85" if cat == "false" else ""
        conn = " to[bend left=12]" if cat != "direct" else " to"
        lines.append(f"  \\draw[->,draw={color[cat]},line width={width(c)}pt{dash}{op}]"
                     f" ({src}){conn} ({dst});")
    # width-scale legend (coupling -> width)
    y0 = -0.5
    lines.append(f"  \\node[font=\\scriptsize,text=black!70,anchor=east] at (1.0,{y0}) "
                 f"{{edge width $=$ coupling:}};")
    for i, cv in enumerate([0.02, 0.15, 0.50]):
        xa = 1.4 + i * 2.5
        lines.append(f"  \\draw[->,draw=black!60,line width={width(cv)}pt] "
                     f"({xa},{y0}) -- ({xa+1.2},{y0});")
        lines.append(f"  \\node[font=\\scriptsize,text=black!60] at ({xa+0.6},{y0-0.34}) "
                     f"{{{cv:.2f}}};")
    lines.append("\\end{tikzpicture}")
    OUT.write_text("\n".join(lines))

    n_direct_truth = len({frozenset(e) for e in DIRECT})   # 11 (O<->P once)
    print(f"coupled pairs returned: {len(coupled)}")
    print(f"  direct cascades recovered : {counts['direct']} (of {n_direct_truth} authored direct)")
    print(f"  transitive (indirect real): {counts['transitive']}")
    print(f"  FALSE positives           : {counts['false']}")
    print(f"  missed direct edges       : {sorted(set(frozenset(e) for e in missed))}")
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
