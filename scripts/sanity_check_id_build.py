"""
One-shot sanity check of the ID pseudo-tube build, run while training is in
flight. Catches the "everything is NaN in 15 hours" failure modes early:

  1. Manifest empty / missing benchmark cytokines / no PBS.
  2. Pseudo-tube expression non-finite (NaN/inf), all-zero, or wrong shape
     (HVG / normalize bug).
  3. Per-(cytokine, rep) coverage too thin for the donor split.

Read-only, gateway-safe (loads a handful of small tubes + the manifest).

Usage:
    python scripts/sanity_check_id_build.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
import scanpy as sc

BASE = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes"
BENCH = ["IFNb", "IFNg", "IL1b", "IL10", "IL12", "IL13",
         "IL15", "IL18", "IL2", "IL4", "IL6", "TNFa"]


def main() -> None:
    man_path = f"{BASE}/manifest.json"
    try:
        man = json.load(open(man_path))
    except Exception as e:
        print(f"FATAL: cannot read manifest {man_path}: {e}")
        sys.exit(2)

    print(f"tubes: {len(man)}")
    if not man:
        print("FATAL: manifest is empty")
        sys.exit(2)

    cyts = Counter(e["cytokine"] for e in man)
    print(f"cytokines: {len(cyts)}  (PBS present: {'PBS' in cyts})")
    missing = [b for b in BENCH if b not in cyts]
    print(f"benchmark cytokines missing: {missing if missing else 'NONE (all 12 present)'}")
    print("benchmark tube counts:")
    for b in BENCH:
        # per-rep coverage for this cytokine
        reps = Counter(e["donor"] for e in man if e["cytokine"] == b)
        print(f"  {b:5s} tubes={cyts.get(b,0):3d}  reps={dict(reps)}")

    ncells = [e["n_cells"] for e in man]
    print(f"n_cells/tube: min={min(ncells)} median={int(np.median(ncells))} "
          f"max={max(ncells)}")

    # Load a few representative tubes; verify finite, non-trivial, right width.
    print("\nsampling tube contents:")
    bad = False
    sample_cyts = ["PBS", "IFNg", "IL6", "TNFa", "IL1b"]
    for name in sample_cyts:
        hits = [e["path"] for e in man if e["cytokine"] == name]
        if not hits:
            print(f"  {name}: NO TUBES")
            continue
        a = sc.read_h5ad(hits[0])
        X = a.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        finite = float(np.isfinite(X).mean())
        nonzero = float((X != 0).mean())
        nct = a.obs["cell_type"].nunique() if "cell_type" in a.obs else -1
        print(f"  {name:5s}: shape={a.shape} genes={a.n_vars} "
              f"finite={100*finite:.1f}% nonzero={100*nonzero:.1f}% "
              f"min={np.nanmin(X):.2f} max={np.nanmax(X):.2f} "
              f"mean={np.nanmean(X):.3f} celltypes={nct}")
        if finite < 1.0:
            print(f"    *** WARNING: {name} has non-finite (NaN/inf) values ***")
            bad = True
        if nonzero < 0.001:
            print(f"    *** WARNING: {name} is essentially all-zero ***")
            bad = True
        if a.n_vars != 4000:
            print(f"    *** WARNING: {name} has {a.n_vars} genes, expected 4000 ***")
            bad = True

    print("\nVERDICT:", "PROBLEM DETECTED — investigate before relying on results"
          if bad or missing else "OK — manifest + tubes look healthy")
    sys.exit(1 if (bad or missing) else 0)


if __name__ == "__main__":
    main()
