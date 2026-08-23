#!/usr/bin/env python
"""Local end-to-end smoke test of the Oesinghaus full-90 DAG (harness only, NOT biology).

Runs stages 0-6 on the synthetic demo fixture (`tests/make_demo_data.py`: 10 cytokines +
PBS, 3 donors, 5 cell types, 200 genes) with tiny epoch counts, so a wiring bug surfaces
on a laptop in seconds instead of after hours of cluster time.

It also asserts the §27.6 guard actually fires: stage 2 must REFUSE to run against an
encoder whose sha256 does not match the one stage 1 wrote.

Nothing here validates the science — the demo data has no planted biology. It validates
that the artifacts flow, the shapes are right, and the provenance assertions bite.

Usage:  python scripts/run_demo_full90_pipeline.py [--workdir DIR] [--keep]
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import _full90_config as C  # noqa: E402

DEMO_EPOCHS_STAGE1 = 3
DEMO_EPOCHS_STAGE2 = 8
DEMO_CHUNKS = 3
DEMO_NULL_PERMS = 5


def run(script: str, *argv: str) -> None:
    banner = f"===== {script} {' '.join(argv)} ====="
    print(f"\n{banner}", flush=True)
    sys.argv = [script, *argv]
    try:
        runpy.run_path(str(REPO_ROOT / "scripts" / script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise SystemExit(f"{script} exited {exc.code}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir or tempfile.mkdtemp(prefix="full90_demo_"))
    demo, run_dir = work / "demo", work / "run"
    print(f"[demo] workdir {work}", flush=True)

    # tiny settings, applied once to the shared config every stage imports
    C.STAGE1_EPOCHS = DEMO_EPOCHS_STAGE1
    C.STAGE2_EPOCHS = DEMO_EPOCHS_STAGE2
    C.N_CHUNKS = DEMO_CHUNKS
    C.N_NULL_PERMS = DEMO_NULL_PERMS

    import json

    import scanpy as sc
    from make_demo_data import make_demo_data

    manifest = make_demo_data(str(demo))
    tube0 = json.loads(Path(manifest).read_text())[0]["path"]
    hvg = demo / "hvg_list.json"
    hvg.write_text(json.dumps([str(g) for g in sc.read_h5ad(tube0).var_names]))

    run("prepare_oesinghaus_full90.py", "--manifest_path", manifest,
        "--hvg_path", str(hvg), "--output_dir", str(run_dir), "--exclude_donors")
    run("train_oesinghaus_full90_encoder.py", "--output_dir", str(run_dir), "--device", "cpu")

    # the §27.6 guard must bite before any chunk trains on a mismatched encoder
    good = (run_dir / "encoder_sha256.txt").read_text()
    (run_dir / "encoder_sha256.txt").write_text("0" * 64 + "\n")
    try:
        run("train_oesinghaus_full90_chunk.py", "--output_dir", str(run_dir),
            "--chunk_id", "0", "--n_chunks", str(DEMO_CHUNKS), "--device", "cpu")
    except SystemExit as exc:
        print(f"[guard] encoder mismatch correctly refused: {exc}", flush=True)
    else:
        raise AssertionError("encoder sha256 guard did NOT fire — §27.6 protection is broken")
    (run_dir / "encoder_sha256.txt").write_text(good)

    for k in range(DEMO_CHUNKS):
        run("train_oesinghaus_full90_chunk.py", "--output_dir", str(run_dir),
            "--chunk_id", str(k), "--n_chunks", str(DEMO_CHUNKS), "--device", "cpu")
    run("merge_full90_signatures.py", "--output_dir", str(run_dir))
    run("run_oesinghaus_full90_coupling.py", "--output_dir", str(run_dir))
    run("run_oesinghaus_full90_direction.py", "--output_dir", str(run_dir))
    run("analyze_oesinghaus_full90.py", "--output_dir", str(run_dir),
        "--report_path", str(run_dir / "RESULTS.md"))

    import pandas as pd

    n_stim = len(json.loads((run_dir / "prepare_meta.json").read_text())["stimuli"])
    expected_pairs = n_stim * (n_stim - 1) // 2
    checks = {
        "coupling_donor_degree.csv": expected_pairs,
        "direction_table.csv": expected_pairs,
        "per_pair_summary.csv": expected_pairs,
    }
    print("\n===== checks =====", flush=True)
    for name, n in checks.items():
        df = pd.read_csv(run_dir / name)
        assert len(df) == n, f"{name}: {len(df)} rows, expected {n}"
        print(f"  {name}: {len(df)} rows OK", flush=True)
    for col in ("q_donor", "coupled_q05", "coupled_q10", "donor_sign_p"):
        assert col in pd.read_csv(run_dir / "coupling_donor_degree.csv").columns, col
    assert (run_dir / "RESULTS.md").exists()
    assert (run_dir / "benchmark_summary.txt").exists()
    print(f"  RESULTS.md: {(run_dir / 'RESULTS.md').stat().st_size} bytes OK", flush=True)
    print("\nALL DEMO CHECKS PASSED (harness only — no biology validated)", flush=True)

    if not args.keep and args.workdir is None:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"[demo] artifacts kept in {work}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
