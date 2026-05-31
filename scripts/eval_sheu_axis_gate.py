"""
§21 axis-discovery gate for Path A on Sheu (per time point) — secondary check.

Reads whatever axis/cascade CSV the latent-geometry step produced for one time
point and reports whether the pre-registered MUST / SHOULD / MUST-NOT pairs are
recovered. This is the COUPLING check (Path A), separate from the directional
benchmark (Path B). Best-effort + defensive: if the geometry output format is
unexpected, it logs and exits 0 so the DAG is not blocked.

Pre-registered (§21, Sheu manifest names):
  MUST     : LPS—TNF, PIC—IFNb
  SHOULD   : LPS—IFNb, P3CSK—CpG, LPSlo—P3CSK
  MUST-NOT : P3CSK—IFNb, CpG—IFNb, TNF—IFNb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MUST = [("LPS", "TNF"), ("IFNb", "PIC")]
SHOULD = [("IFNb", "LPS"), ("CpG", "P3CSK"), ("LPSlo", "P3CSK")]
MUST_NOT = [("IFNb", "P3CSK"), ("CpG", "IFNb"), ("IFNb", "TNF")]


def _canon(a, b):
    return tuple(sorted([a, b]))


def _find_axis_csv(geo_dir: Path) -> Path | None:
    for name in ("cytokine_axes.csv", "axes.csv", "cascade_axes.csv"):
        cands = list(geo_dir.rglob(name))
        if cands:
            return cands[0]
    # any csv with axis_a / axis_b columns
    for c in geo_dir.rglob("*.csv"):
        try:
            cols = pd.read_csv(c, nrows=1).columns
            if "axis_a" in cols and "axis_b" in cols:
                return c
        except Exception:
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--geo_dir", required=True, help="Latent-geometry output dir for this time point.")
    p.add_argument("--out", required=True)
    p.add_argument("--time", default="")
    args = p.parse_args()

    geo_dir = Path(args.geo_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# Sheu §21 axis gate (Path A coupling) — {args.time}", ""]
    axis_csv = _find_axis_csv(geo_dir) if geo_dir.exists() else None
    if axis_csv is None:
        lines.append(f"NOTE: no axis CSV found under `{geo_dir}` — Path A output "
                     "format not as expected; skipping gate (Path B is the primary readout).")
        out.write_text("\n".join(lines) + "\n")
        print(f"Wrote {out} (no axis CSV found)")
        return

    df = pd.read_csv(axis_csv)
    found = set(_canon(a, b) for a, b in zip(df["axis_a"], df["axis_b"]))
    # strength column if present
    strength_col = next((c for c in ("axis_strength", "strength", "pooled_a_to_b") if c in df.columns), None)

    def _status(pairs, label):
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| pair | present | strength |")
        lines.append("|---|:-:|---|")
        n_present = 0
        for a, b in pairs:
            key = _canon(a, b)
            present = key in found
            n_present += int(present)
            strength = ""
            if present and strength_col:
                row = df[[_canon(x, y) == key for x, y in zip(df["axis_a"], df["axis_b"])]]
                if len(row):
                    strength = f"{float(row[strength_col].iloc[0]):.4f}"
            lines.append(f"| {a}—{b} | {'✓' if present else '✗'} | {strength} |")
        lines.append("")
        return n_present

    n_must = _status(MUST, "MUST recover")
    _status(SHOULD, "SHOULD recover")
    n_mustnot = _status(MUST_NOT, "MUST-NOT call (false positives)")

    lines.append("## Gate")
    lines.append("")
    lines.append(f"- MUST recovered: **{n_must}/{len(MUST)}**")
    lines.append(f"- MUST-NOT falsely called: **{n_mustnot}/{len(MUST_NOT)}**")
    lines.append(f"- axis CSV: `{axis_csv}`")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}: MUST {n_must}/{len(MUST)}, MUST-NOT {n_mustnot}/{len(MUST_NOT)}")


if __name__ == "__main__":
    main()
