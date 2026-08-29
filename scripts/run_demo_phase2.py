"""Local end-to-end de-risk for Phase 1+2 — HARNESS ONLY, not biology.

Runs stage A (both encoder paths) -> stage B (all four arms) -> stage C (analysis) on
`tests/make_demo_data.py`'s simulated data at toy scale, and asserts the things that would
silently corrupt the real run:

  * the two paths really do produce DIFFERENT encoders (if not, the transplant arms are
    the same experiment twice and the bisection is meaningless);
  * every arm produces a signature table over the full demo panel;
  * an arm REFUSES to run against a tampered encoder digest (the §27.6 guard, which is what
    stops `cm_cd` from quietly using cascadir's encoder and inverting the verdict);
  * the analyser runs and emits a verdict.

The demo's numbers mean nothing — the genes are simulated. This checks plumbing.
"""

from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "cascadir" / "src", REPO_ROOT / "scripts",
           REPO_ROOT / "tests"):
    sys.path.insert(0, str(_p))

import _phase2_config as C  # noqa: E402

DEMO_SEEDS = [42, 123]
DEMO_PANEL = ["IL-2", "IFN-alpha", "IL-6"]


def patch_config(work: Path) -> None:
    C.OUT_DIR = work / "phase2"
    C.EMBED_DIM = 32
    C.HIDDEN_DIMS = (32, 32)
    C.ATTENTION_HIDDEN_DIM = 8
    C.STAGE1_EPOCHS = 2
    C.STAGE1_BATCH = 64
    C.STAGE2_EPOCHS = 3
    C.TOP_N = 10
    C.DEEP_N = 20
    C.IG_STEPS = 3
    C.SEEDS = DEMO_SEEDS
    C.PANEL = DEMO_PANEL
    C.VAL_DONORS = ["Donor3"]


def run(script: str, *args: str) -> None:
    argv = [script, *args]
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(REPO_ROOT / "scripts" / script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise AssertionError(f"{script} exited {exc.code}") from exc
    finally:
        sys.argv = old


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="phase2_demo_"))
    demo, shard_dir = work / "demo", work / "tubes"
    print(f"[demo] workdir {work}", flush=True)
    patch_config(work)

    import anndata
    import numpy as np
    import scanpy as sc

    import make_demo_data as mdd
    from run_demo_oes90_pure import build_demo_shards

    mdd.N_PSEUDO_TUBES = 2
    manifest_path = mdd.make_demo_data(str(demo))
    entries = json.loads(Path(manifest_path).read_text())
    gene_names = [str(g) for g in sc.read_h5ad(entries[0]["path"]).var_names]
    hvg = work / "hvg.json"
    hvg.write_text(json.dumps(gene_names))

    # Shards exclude the val donor, mirroring the real run where §36's prepare already
    # dropped D2/D3 and split_manifest_by_donor drops them again — so BOTH Stage-2 paths
    # see the same donors.
    train_entries = [e for e in entries if e["donor"] not in C.VAL_DONORS]
    train_manifest = work / "manifest_train_only.json"
    train_manifest.write_text(json.dumps(train_entries))
    print("\n===== materialising demo shards (train donors only) =====", flush=True)
    build_demo_shards(str(train_manifest), gene_names, shard_dir)

    # Stage-1 cells: one flat AnnData, the same cells for both paths.
    print("\n===== building demo stage1 cells =====", flush=True)
    Xs, cts = [], []
    for e in train_entries:
        ad = sc.read_h5ad(e["path"])[:, gene_names]
        X = ad.X
        Xs.append(X.toarray() if hasattr(X, "toarray") else np.asarray(X))
        cts += [str(c) for c in ad.obs["cell_type"]]
    s1 = anndata.AnnData(X=np.ascontiguousarray(np.vstack(Xs), dtype=np.float32))
    s1.obs["cell_type"] = cts
    s1.var_names = gene_names
    s1_path = work / "stage1_cells.h5ad"
    s1.write_h5ad(s1_path)
    print(f"  {s1.n_obs} cells x {s1.n_vars} genes, {len(set(cts))} cell types")

    common = ["--out_dir", str(C.OUT_DIR), "--device", "cpu"]

    print("\n===== stage A: encoders =====", flush=True)
    for path in C.PATHS:
        for seed in C.SEEDS:
            run("phase2_train_encoder.py", "--path", path, "--seed", str(seed),
                "--stage1_cells", str(s1_path), *common)

    shas = {(p, s): (C.encoder_path_dir(p, s) / "encoder_sha256.txt").read_text().strip()
            for p in C.PATHS for s in C.SEEDS}
    print("\n[check] encoder digests:")
    for k, v in shas.items():
        print(f"   {k}: {v[:16]}...")
    for seed in C.SEEDS:
        assert shas[("cm", seed)] != shas[("cd", seed)], (
            f"seed {seed}: the two paths produced IDENTICAL encoders — the transplant "
            "arms would be the same experiment twice and the bisection would be vacuous."
        )
    assert len(set(shas.values())) == len(shas), "some encoders collided across seeds"

    print("\n===== stage B: four arms =====", flush=True)
    for enc_p, s2_p in C.ARMS:
        for seed in C.SEEDS:
            run("phase2_train_arm.py", "--encoder_path", enc_p, "--stage2_path", s2_p,
                "--seed", str(seed), "--shard_dir", str(shard_dir),
                "--manifest", manifest_path, "--hvg_path", str(hvg),
                "--panel", *C.PANEL, *common)

    import pandas as pd
    for enc_p, s2_p in C.ARMS:
        for seed in C.SEEDS:
            f = C.arm_dir(enc_p, s2_p, seed) / "signatures.parquet"
            assert f.exists(), f"missing {f}"
            df = pd.read_parquet(f)
            assert set(df.cytokine.unique()) == set(C.PANEL), (
                f"{f} covers {sorted(df.cytokine.unique())}, expected {C.PANEL}")

    print("\n===== encoder-digest guard =====", flush=True)
    victim = C.encoder_path_dir("cm", C.SEEDS[0]) / "encoder_sha256.txt"
    good = victim.read_text()
    victim.write_text("0" * 64)
    try:
        run("phase2_train_arm.py", "--encoder_path", "cm", "--stage2_path", "cd",
            "--seed", str(C.SEEDS[0]), "--shard_dir", str(shard_dir),
            "--manifest", manifest_path, "--hvg_path", str(hvg),
            "--panel", C.PANEL[0], *common)
    except AssertionError as exc:
        print(f"[guard] tampered digest correctly refused: {str(exc)[:120]}")
    else:
        raise AssertionError(
            "an arm ran happily against a tampered encoder digest — the §27.6 guard is "
            "broken and an arm could silently use the wrong path's encoder."
        )
    finally:
        victim.write_text(good)

    print("\n===== stage C: analysis =====", flush=True)
    run("analyze_phase2.py", "--out_dir", str(C.OUT_DIR), "--top_n", str(C.TOP_N))
    assert (C.OUT_DIR / "PHASE2_VERDICT.md").exists(), "no verdict written"
    assert (C.OUT_DIR / "phase2_arm_summary.csv").exists(), "no arm summary written"

    print("\n[PASS] Phase 1+2 runs end-to-end; paths give different encoders; all four "
          "arms produce signatures; the digest guard fires; a verdict is written.")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"[demo] kept {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
