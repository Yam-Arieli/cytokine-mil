"""Local end-to-end de-risk for Phase 0a — HARNESS ONLY, not biology.

Builds `tests/make_demo_data.py`'s simulated tubes, materialises them as (donor, condition)
shards exactly as §36 did on the cluster, and runs `phase0_tube_identity.py` over both.

It asserts BOTH directions, because a gate that cannot fail proves nothing:

  * POSITIVE — when the two sides really do read the same tubes, the gate PASSES and
    reports zero mismatches;
  * NEGATIVE — after one shard is perturbed by a single float, the gate FAILS and names
    the affected tubes.

The demo's numbers mean nothing (the genes are simulated). This checks plumbing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "cascadir" / "src", REPO_ROOT / "scripts",
           REPO_ROOT / "tests"):
    sys.path.insert(0, str(_p))

GATE = REPO_ROOT / "scripts" / "phase0_tube_identity.py"


def _run_gate(manifest: str, hvg: Path, shard_dir: Path, out_dir: Path):
    r = subprocess.run(
        [sys.executable, str(GATE), "--manifest", manifest, "--hvg_path", str(hvg),
         "--shard_dir", str(shard_dir), "--out_dir", str(out_dir)],
        capture_output=True, text=True,
    )
    summary_path = out_dir / "phase0a_tube_identity.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return r, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="phase0_demo_"))
    demo, shard_dir = work / "demo", work / "tubes"
    print(f"[demo] workdir {work}", flush=True)

    import scanpy as sc

    import make_demo_data as mdd
    from run_demo_oes90_pure import build_demo_shards

    mdd.N_PSEUDO_TUBES = 2
    manifest = mdd.make_demo_data(str(demo))
    entries = json.loads(Path(manifest).read_text())
    gene_names = [str(g) for g in sc.read_h5ad(entries[0]["path"]).var_names]
    hvg = work / "hvg.json"
    hvg.write_text(json.dumps(gene_names))

    print("\n===== materialising demo shards =====", flush=True)
    build_demo_shards(manifest, gene_names, shard_dir)

    print("\n===== POSITIVE: identical data must PASS =====", flush=True)
    r, summary = _run_gate(manifest, hvg, shard_dir, work / "out_pos")
    print(r.stdout[-1800:])
    if r.returncode != 0 or not summary.get("passed"):
        print(r.stderr[-1500:])
        raise AssertionError("gate 0a FAILED on identical data — it is broken.")
    if summary["n_both"] == 0:
        raise AssertionError("gate 0a compared zero tubes — it would 'pass' vacuously.")
    print(f"[ok] {summary['n_both']} tubes compared, 0 mismatches")

    print("\n===== NEGATIVE: a one-float edit must FAIL =====", flush=True)
    victim = sorted(shard_dir.glob("*.npy"))[0]
    X = np.load(victim)
    X[0, 0] = np.float32(X[0, 0] + 1.0)
    np.save(victim, X)
    print(f"[demo] perturbed a single value in {victim.name}")

    r2, summary2 = _run_gate(manifest, hvg, shard_dir, work / "out_neg")
    print(r2.stdout[-1200:])
    if r2.returncode == 0 or summary2.get("passed"):
        raise AssertionError(
            "gate 0a PASSED on deliberately corrupted shards — it cannot detect a "
            "mismatch, so a real 'pass' would mean nothing."
        )
    if summary2["n_sha_mismatch"] < 1:
        raise AssertionError("gate 0a failed for the wrong reason — no sha mismatch reported.")
    print(f"[ok] gate refused: {summary2['n_sha_mismatch']} tube(s) flagged, "
          f"e.g. {summary2['sha_mismatch_examples'][:1]}")

    print("\n[PASS] gate 0a detects identity AND detects corruption.")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"[demo] kept {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
