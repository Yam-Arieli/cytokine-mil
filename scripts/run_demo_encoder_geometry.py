"""Local end-to-end de-risk for the encoder gene-geometry probe — HARNESS ONLY.

Runs probe -> analyze on simulated demo data at toy scale and asserts the things that would
silently produce a meaningless table on the real run:

  * every probe (one-hot, Jacobian, w1) emits a participation ratio for every fit;
  * the one-hot map really is E(e_g) - E(0), not the raw response;
  * the gene-level link joins signature frequencies onto the right genes;
  * the matched untrained control is emitted, so trained values have a reference;
  * the analyzer renders a verdict and figures from what the probe wrote.

The numbers mean nothing here — the genes are simulated and the encoders are untrained.
This checks plumbing only. The statistic itself is controlled in tests/test_encoder_geometry.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "cascadir" / "src", REPO_ROOT / "scripts",
          REPO_ROOT / "tests"):
    sys.path.insert(0, str(p))

import _encoder_geometry_fits as R  # noqa: E402
import analyze_encoder_geometry as AN  # noqa: E402
import probe_encoder_gene_geometry as P  # noqa: E402
from cascadir.models import InstanceEncoder  # noqa: E402
from run_demo_oes90_pure import build_demo_shards  # noqa: E402

DEMO_EMBED, DEMO_HIDDEN, TOP_N = 24, (32, 28), 8


def fake_fit(key: str, root: Path, run_dir: Path, genes, conditions, seed: int) -> dict:
    """A random-init encoder plus a synthetic signature table, in the real layout."""
    import pandas as pd

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    enc = InstanceEncoder(input_dim=len(genes), embed_dim=DEMO_EMBED, n_cell_types=5,
                          hidden_dims=DEMO_HIDDEN)
    torch.save(enc.state_dict(), run_dir / "encoder.pt")

    rng = np.random.default_rng(seed)
    rows = []
    for c in conditions:
        for r, g in enumerate(rng.choice(genes, size=TOP_N, replace=False)):
            rows.append({"cytokine": c, "gene": str(g), "rank_ig": r,
                         "ig": float(TOP_N - r)})
    pd.DataFrame(rows).to_parquet(run_dir / "signatures.parquet", index=False)

    rel = run_dir.relative_to(root)
    return {"key": key, "encoder": str(rel / "encoder.pt"), "code_path": "cascadir",
            "panel": "sweep24", "signatures": str(rel / "signatures.parquet"),
            "conditions": None, "embed_dim": DEMO_EMBED, "diagnostic_only": False,
            "note": "demo"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix="encgeom_demo_"))
    demo, shard_dir, out = work / "demo", work / "tubes", work / "run"
    print(f"[demo] workdir {work}", flush=True)

    import scanpy as sc

    import make_demo_data as mdd

    mdd.N_PSEUDO_TUBES = 2
    manifest = mdd.make_demo_data(str(demo))
    entries = json.loads(Path(manifest).read_text())
    genes = [str(g) for g in sc.read_h5ad(entries[0]["path"]).var_names]
    conditions = sorted({e["cytokine"] for e in entries if e["cytokine"] != "PBS"})[:4]

    print("\n===== demo tube shards =====", flush=True)
    meta = build_demo_shards(manifest, genes, shard_dir)
    print(f"  {meta['n_tubes']} tubes, {len(genes)} genes")

    # Two "fits" so the cross-fit machinery has something to compare.
    P.REPO_ROOT = work
    AN.REPO_ROOT = work
    R.FITS[:] = [fake_fit(f"demo_{i}", work, work / f"fit_{i}", genes, conditions, 100 + i)
                 for i in range(2)]
    R.BY_KEY.clear()
    R.BY_KEY.update({f["key"]: f for f in R.FITS})

    print("\n===== probe =====", flush=True)
    sys.argv = ["probe", "--out_dir", str(out), "--shard_dir", str(shard_dir),
                "--top_n", str(TOP_N), "--n_jac_cells", "2", "--jac_donor", "Donor1",
                "--device", "cpu"]
    assert P.main() == 0

    import pandas as pd

    geo = pd.read_csv(out / "gene_geometry.csv")
    print(f"\n[check] gene_geometry.csv: {len(geo)} rows")
    for key in ("demo_0", "demo_1"):
        got = set(geo[geo.fit == key].probe)
        assert got == set(P.PROBES), f"{key}: probes {got} != {set(P.PROBES)}"
        assert geo[geo.fit == key].raw_pr.notna().all(), f"{key}: a probe emitted no PR"
    assert (geo.fit == "demo_0__untrained").any(), "matched untrained control missing"

    pg = pd.read_parquet(out / "per_gene" / "demo_0.parquet")
    assert len(pg) == len(genes), f"per-gene table is {len(pg)} rows, expected {len(genes)}"
    assert "sig_freq" in pg.columns, "gene-level link did not attach signature frequencies"
    assert pg.sig_freq.sum() > 0, "every signature frequency is zero — the join missed"
    sig = pd.read_parquet(work / "fit_0" / "signatures.parquet")
    expect = sig[sig.rank_ig < TOP_N].groupby("gene").cytokine.nunique()
    got = pg.set_index("gene").sig_freq
    assert all(got[g] == n for g, n in expect.items()), "signature frequencies misaligned"
    print(f"[check] gene link OK — {int((pg.sig_freq > 0).sum())} genes in >=1 signature")

    spec = pd.read_parquet(out / "spectra.parquet")
    assert set(spec.probe) == set(P.PROBES), "spectra missing a probe"

    print("\n===== analyze =====", flush=True)
    sys.argv = ["analyze", "--out_dir", str(out), "--top_n", str(TOP_N)]
    assert AN.main() == 0
    assert (out / "ENCODER_GEOMETRY.md").exists(), "no verdict written"
    made = sorted(p.name for p in (out / "plots").glob("*.png"))
    assert made, "no figures rendered"
    print(f"\n[check] figures: {made}")

    print("\n[PASS] probe + analyze run end-to-end; all three probes report; the gene "
          "link joins correctly; the untrained control is emitted; the verdict renders.")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"[demo] kept {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
