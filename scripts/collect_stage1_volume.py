"""Stage-1 training volume per fit — the confound check for the geometry probe.

The one-hot gene-map rank varies 2.7x across fits (CLAUDE.md §39.5), and the fits happen to
sort by training code path. But the two code paths ALSO differ systematically in how much
Stage-1 data their encoder saw: `build_stage1_manifest` (the cytokine_mil path) takes one
tube per condition, while the cascadir sweeps trained on a fixed 36K-cell budget. Volume is
therefore a competing explanation for the same ordering, and it has to be measured rather
than assumed either way.

Reads only what each run already recorded: `encoder_meta.json` (`n_cells`) for cascadir
fits, `manifest_stage1_shared.json` (sum of `n_cells`) for cytokine_mil fits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _encoder_geometry_fits as R  # noqa: E402


def stage1_volume(fit: dict) -> dict:
    """(cells, tubes, source, final train loss) for one fit, or NaNs where unrecorded."""
    run = (REPO_ROOT / fit["encoder"]).parent
    out = {"fit": fit["key"], "code_path": fit["code_path"],
           "stage1_cells": None, "stage1_tubes": None, "stage1_loss": None, "source": "-"}

    meta_p = run / "encoder_meta.json"
    if meta_p.exists():                       # cascadir fits record it directly
        m = json.loads(meta_p.read_text())
        out.update(stage1_cells=m.get("n_cells"), stage1_loss=m.get("final_train_loss"),
                   source="encoder_meta.json")

    man_p = run / "manifest_stage1_shared.json"
    if man_p.exists():                        # cytokine_mil fits record the manifest
        entries = json.loads(man_p.read_text())
        out.update(stage1_cells=int(sum(int(e["n_cells"]) for e in entries)),
                   stage1_tubes=len(entries), source="manifest_stage1_shared.json")

    # the sweeps keep per-arm cell counts in the run-level meta instead
    if out["stage1_cells"] is None:
        for cand in (run.parent / "encsweep_meta.json",):
            if cand.exists():
                m = json.loads(cand.read_text())
                arm = m.get("arms", {}).get(run.name)
                if arm:
                    out.update(stage1_cells=arm.get("n_cells"), source=cand.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_csv", default=str(REPO_ROOT / "results" / "encoder_geometry"
                                            / "stage1_volume.csv"))
    args = ap.parse_args()

    rows = [stage1_volume(f) for f in R.FITS if (REPO_ROOT / f["encoder"]).exists()]
    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\n[write] {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
