#!/usr/bin/env python
"""Emit the two thesis-figure specs for the recovered cascade_forge graph (Figure 13).

The lower panel of Figure 13 used to be a single bar chart of coupling only, hand-pasted
as inline TikZ. It becomes two panels side by side -- coupling (Figure-11 style, signed)
and |cross_asym| (Figure-10 style, magnitude) -- generated through the `thesis-figures`
skill instead of by hand.

Nothing here is transcribed: every value is read from the cascadir outputs, and every
edge's class comes from `expected_sign`, which is itself derived from the authored ground
truth. The two panels differ only in which expectation they encode:

  coupling  -- expected_sign = +1 if the pair is truly coupled in the authored graph
               (reachable, direct or transitive), -1 if it is not. Every recovered pair
               has positive coupling, so this renders truly-coupled pairs green and
               false positives orange.
  direction -- expected_sign set only for the 10 signed authored edges (the O<->P feedback
               pair has no signed direction and is excluded, as in benchmark_large_cascade),
               oriented upstream->downstream. Pairs with no authored direction stay grey.

Usage:
    python make_forge_specs.py --results_dir results/cascade_forge_large/all_eff0.30 \
                               --stem snapshot_t6 \
                               --out_dir ../../overleaf-thesis-cascades/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# Node positions -- identical to Figure 12 (the authored graph) and to the top panel of
# Figure 13, so the two figures read as the same layout.
POS = {
    "A": (0, 5), "B": (2, 5), "C": (4, 5), "D": (6, 5),
    "E": (0, 3.6), "F": (2, 3.6), "G": (4, 3.6),
    "H": (0, 1.6), "I": (2.2, 2.4), "J": (2.2, 1.6), "K": (2.2, 0.8),
    "L": (4.4, 2.2), "M": (4.4, 1.0), "N": (6.4, 1.6),
    "O": (9, 4.9), "P": (9, 3.5),
    "Q": (8.4, 1.9), "R": (9.4, 1.9), "S": (8.4, 0.9), "T": (9.4, 0.9),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", required=True,
                   help="directory holding <stem>.coupling.csv / .direction_all.csv / ground_truth.json")
    p.add_argument("--stem", default="snapshot_t6")
    p.add_argument("--out_dir", required=True, help="overleaf-thesis-cascades/figures")
    p.add_argument("--skip_direction", action="store_true",
                   help="emit only the coupling spec (for use before direction_all.csv exists)")
    return p.parse_args(argv)


def _read_coupled(path):
    """Return {frozenset(pair): coupling} for the pairs the coupling gate flagged."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        frozenset((r["condition_a"], r["condition_b"])): float(r["coupling"])
        for r in rows if r["coupled"].strip().lower() in ("true", "1")
    }


def _read_direction(path):
    """Return {frozenset(pair): (condition_a, cross_asym_median)} keyed by unordered pair.

    The sign of cross_asym_median is relative to `condition_a`, so both are kept -- the
    generator needs the value in the same (a, b) order the source data used.
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        frozenset((r["condition_a"], r["condition_b"])):
            (r["condition_a"], r["condition_b"], float(r["cross_asym_median"]))
        for r in rows
    }


def _scales(values, *, signed):
    """Scale constants from the actual value range, per the thesis-figures schema recipe."""
    max_abs = max(abs(v) for v in values)
    k = round(3.0 / max_abs, 2)
    hscale = 2.0 / max_abs
    # A signed panel extends both ways from zero and puts its labels below the negative
    # extent, so it needs less height per unit than an unsigned one.
    hscale = round(hscale / 1.6 if signed else hscale, 2)
    # ticks: 2-3 round numbers spanning [0, max_abs] -- pick the roundest step that
    # gives 2 or 3 intervals, so the axis stays as sparse as Figures 10/11's.
    mag = 10 ** math.floor(math.log10(max_abs))
    tick = min((m * mag for m in (0.2, 0.25, 0.5, 1.0, 2.0)
                if 2 <= max_abs / (m * mag) <= 3.999),
               key=lambda t: abs(max_abs / t - 3))
    ticks, t = [0.0], tick
    while t <= max_abs * 1.001:
        ticks.append(round(t, 10))
        t += tick
    return k, hscale, ticks


def _legend_refs(values):
    """Three representative |value|s for the graph legend: smallest, middle, largest."""
    v = sorted(abs(x) for x in values)
    return [round(v[0], 3), round(v[len(v) // 2], 3), round(v[-1], 3)]


def build_specs(results_dir: Path, stem: str, *, skip_direction: bool):
    gt = json.loads((results_dir / "ground_truth.json").read_text())
    reachable = {frozenset(e) for e in gt["reachable_edges"]}
    bidir = {frozenset(e) for e in gt["bidirectional_pairs"]}
    # Upstream node per signed authored edge (feedback pair has none).
    upstream = {frozenset((a, b)): a for a, b in gt["direct_edges"]
                if frozenset((a, b)) not in bidir}

    coupled = _read_coupled(results_dir / f"{stem}.coupling.csv")
    nodes = [{"id": n, "x": x, "y": y, "label": n} for n, (x, y) in POS.items()]

    # ---- coupling spec: expectation is "is this pair coupled at all?" ----
    cedges = []
    for pair, value in sorted(coupled.items(), key=lambda kv: sorted(kv[0])):
        a, b = sorted(pair)
        cedges.append({"a": a, "b": b, "value": value,
                       "expected_sign": 1 if pair in reachable else -1})
    ck, chs, cticks = _scales([e["value"] for e in cedges], signed=True)

    coupling_spec = {
        "figure_id": "forge-coupling",
        "label": "fig:forge-coupling-bars",
        "caption": "PLACEHOLDER -- this panel is used via split_panels, not as a standalone figure.",
        "value_label": "coupling (degree-corrected)",
        "_source": (
            f"{results_dir}/{stem}.coupling.csv, column `coupling` (degree-corrected, "
            "donor-level; est.signature_coupling(donor_level=True) in "
            "scripts/benchmark_large_cascade.py) -> edge value, read verbatim. "
            "expected_sign derived from ground_truth.json reachable_edges: +1 if the pair "
            "is truly coupled in the authored graph, -1 if it is not."
        ),
        "nodes": nodes,
        "edges": cedges,
        "classes": {"correct": {"color": "fgood"}, "wrong": {"color": "fbad"},
                    "other": {"color": "gray!45"}},
        "graph_width_scale": {"k": ck, "floor_pt": 0.3},
        "bar_width_scale": {"k": ck, "floor_pt": 0.2},
        "bar_height_scale": chs,
        "bar_label_font_size": 1.9,
        "bar_label_font_skip": 2.2,
        "bar_signed": True,
        "y_ticks": cticks,
        "legend": {"value_refs": _legend_refs([e["value"] for e in cedges]),
                   "class_order": ["correct", "wrong"],
                   "class_labels": {"correct": "authored coupling recovered",
                                    "wrong": "false positive"}},
        "benchmark_style": "undirected",
    }

    specs = {"forge-coupling": coupling_spec}
    if skip_direction:
        return specs

    direction = _read_direction(results_dir / f"{stem}.direction_all.csv")
    missing = set(coupled) - set(direction)
    if missing:
        raise SystemExit(
            f"direction_all.csv is missing {len(missing)} coupled pairs: "
            f"{[tuple(sorted(m)) for m in sorted(missing, key=lambda s: sorted(s))]}"
        )

    dedges = []
    for pair in sorted(coupled, key=lambda p: sorted(p)):
        a, b, value = direction[pair]
        e = {"a": a, "b": b, "value": value}
        if pair in upstream:
            # +1 means `a` is the authored upstream condition, -1 means `b` is.
            e["expected_sign"] = 1 if upstream[pair] == a else -1
        dedges.append(e)
    dk, dhs, dticks = _scales([e["value"] for e in dedges], signed=False)

    specs["forge-direction"] = {
        "figure_id": "forge-direction",
        "label": "fig:forge-direction-bars",
        "caption": "PLACEHOLDER -- this panel is used via split_panels, not as a standalone figure.",
        "value_label": "$|\\mathrm{cross\\_asym}|$",
        "_source": (
            f"{results_dir}/{stem}.direction_all.csv, column `cross_asym_median` "
            "(cascadir CascadeDirection.direction_table on the pairs the coupling gate "
            "flagged) -> edge value, read verbatim. expected_sign derived from "
            "ground_truth.json direct_edges, excluding the O<->P feedback pair, which has "
            "no signed authored direction."
        ),
        "nodes": nodes,
        "edges": dedges,
        "classes": {"correct": {"color": "fgood"}, "wrong": {"color": "fbad"},
                    "other": {"color": "gray!45"}},
        "graph_width_scale": {"k": dk, "floor_pt": 0.3},
        "bar_width_scale": {"k": dk, "floor_pt": 0.2},
        "bar_height_scale": dhs,
        "bar_label_font_size": 1.9,
        "bar_label_font_skip": 2.2,
        "bar_signed": False,
        "y_ticks": dticks,
        "legend": {"value_refs": _legend_refs([e["value"] for e in dedges]),
                   "class_order": ["correct", "wrong", "other"],
                   "class_labels": {"correct": "authored direction recovered",
                                    "wrong": "authored direction reversed",
                                    "other": "no authored direction"}},
        "benchmark_style": "directed",
        "arrow_direction_source": "expected",
        "direction_legend_note": (
            "authored edges point upstream$\\to$downstream (colour $=$ verdict); "
            "the rest point by sign($\\mathrm{cross\\_asym}$)"
        ),
    }
    return specs


def main(argv=None):
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for figure_id, spec in build_specs(results_dir, args.stem,
                                       skip_direction=args.skip_direction).items():
        path = out_dir / f"{figure_id}.spec.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        n_bench = sum(1 for e in spec["edges"] if "expected_sign" in e)
        print(f"wrote {path}  ({len(spec['edges'])} edges, {n_bench} with an expectation, "
              f"k={spec['graph_width_scale']['k']}, hscale={spec['bar_height_scale']}, "
              f"ticks={spec['y_ticks']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
