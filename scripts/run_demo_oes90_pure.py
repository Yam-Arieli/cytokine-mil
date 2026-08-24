#!/usr/bin/env python
"""Local end-to-end smoke test of the Oesinghaus 90-cytokine PURE DAG (harness, NOT biology).

Runs every stage on the synthetic demo fixture (`tests/make_demo_data.py`: 10 cytokines +
PBS, 3 donors, 5 cell types, 200 genes, 4 tubes per (donor, condition)) with tiny epoch
counts, so a wiring bug surfaces on a laptop in seconds instead of after hours of cluster
time.

It also asserts the three provenance guards actually bite:
  * stage 3 refuses an encoder whose sha256 differs from stage 1's (CLAUDE.md §27.6);
  * stage 3 refuses an embedding cache whose digest differs from stage 2's;
  * merge refuses chunks that disagree on the encoder.

And it asserts the agnosticism guard: none of the pure-run scripts may pull in
`_full90_config`, which carries the audited-pair and published-panel paths.

Nothing here validates the science — the demo data has no planted biology. It validates
that the artifacts flow, the shapes are right, and the assertions fire.

Usage:  python scripts/run_demo_oes90_pure.py [--workdir DIR] [--keep]
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import _oes90_pure_config as C  # noqa: E402

DEMO_TUBES_PER_GROUP = 4
DEMO_MAIN = [0, 1]
DEMO_RESERVE = [2, 3]
DEMO_STAGE1_EPOCHS = 6
DEMO_STAGE2_EPOCHS = 6
DEMO_CHUNKS = 3
DEMO_NULL_PERMS = 5
DEMO_TOP_N = 20
DEMO_CELLS_PER_CYTOKINE = 120


def run(script: str, *argv: str) -> None:
    banner = f"===== {script} {' '.join(argv)} ====="
    print(f"\n{banner}", flush=True)
    sys.argv = [script, *argv]
    try:
        runpy.run_path(str(REPO_ROOT / "scripts" / script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise SystemExit(f"{script} exited {exc.code}")


def expect_failure(what: str, script: str, *argv: str) -> None:
    """Run a stage that MUST refuse; raise if it happily proceeds."""
    try:
        run(script, *argv)
    except (SystemExit, AssertionError) as exc:
        print(f"[guard] {what} correctly refused: {type(exc).__name__}: {exc}", flush=True)
    else:
        raise AssertionError(f"{what} guard did NOT fire — the protection is broken")


def build_demo_shards(manifest_path: str, gene_names: list, shard_dir: Path) -> dict:
    """Materialise the demo manifest as (donor, condition) shards, as §36 did on cluster."""
    from collections import defaultdict

    from cascadir.types import PseudoTube

    from cytokine_mil.analysis.full90_tube_io import ShardWriter

    prep = runpy.run_path(str(REPO_ROOT / "scripts" / "prepare_oes90_pure.py"))
    read_tube = prep["_read_tube"]

    manifest = json.loads(Path(manifest_path).read_text())
    groups = defaultdict(list)
    for e in manifest:
        groups[(e["donor"], e["cytokine"])].append(e)

    writer = ShardWriter(shard_dir)
    for (donor, condition), entries in sorted(groups.items()):
        tubes = []
        for e in sorted(entries, key=lambda x: x.get("tube_idx", 0)):
            X, cell_types, _ = read_tube(e["path"], gene_names)
            tubes.append(
                PseudoTube(
                    X=X,
                    condition=str(condition),
                    donor=str(donor),
                    cell_types=tuple(cell_types.tolist()),
                    cell_types_included=tuple(sorted(set(cell_types.tolist()))),
                    tube_idx=int(e.get("tube_idx", 0)),
                )
            )
        writer.add_group(donor, condition, tubes)
    return writer.finalize(gene_names, C.CONTROL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir or tempfile.mkdtemp(prefix="oes90_pure_demo_"))
    demo, run_dir, shard_dir = work / "demo", work / "run", work / "tubes"
    print(f"[demo] workdir {work}", flush=True)

    # tiny settings, applied once to the shared config every stage imports
    C.MAIN_TUBE_INDICES = DEMO_MAIN
    C.RESERVE_TUBE_INDICES = DEMO_RESERVE
    C.STAGE1_EPOCHS = DEMO_STAGE1_EPOCHS
    C.STAGE1_PATIENCE = 2
    C.STAGE1_EXTRA_EPOCHS = 1
    C.STAGE2_EPOCHS = DEMO_STAGE2_EPOCHS
    C.N_CHUNKS = DEMO_CHUNKS
    C.N_NULL_PERMS = DEMO_NULL_PERMS
    C.TOP_N = DEMO_TOP_N
    C.CELLS_PER_CYTOKINE = DEMO_CELLS_PER_CYTOKINE
    C.VAL_DONORS = []  # the demo has only 3 donors; holding two out would leave one

    import scanpy as sc

    import make_demo_data as mdd

    mdd.N_PSEUDO_TUBES = DEMO_TUBES_PER_GROUP
    manifest = mdd.make_demo_data(str(demo))
    tube0 = json.loads(Path(manifest).read_text())[0]["path"]
    gene_names = [str(g) for g in sc.read_h5ad(tube0).var_names]
    hvg = demo / "hvg_list.json"
    hvg.write_text(json.dumps(gene_names))

    print("\n===== building demo tube shards =====", flush=True)
    shard_meta = build_demo_shards(manifest, gene_names, shard_dir)
    print(f"  {shard_meta['n_tubes']} tubes, sha={shard_meta['shards_sha256'][:16]}...",
          flush=True)

    common = ["--out_dir", str(run_dir)]
    run("prepare_oes90_pure.py", "--shard_dir", str(shard_dir),
        "--manifest_path", manifest, "--hvg_path", str(hvg),
        "--exclude_donors", *common)
    run("train_oes90_pure_encoder.py", *common, "--device", "cpu")
    run("encode_oes90_pure_tubes.py", *common, "--device", "cpu")

    # --- guard 1: a mismatched encoder must stop stage 3 before it trains ---------
    good_enc = (run_dir / "encoder_sha256.txt").read_text()
    (run_dir / "encoder_sha256.txt").write_text("0" * 64 + "\n")
    expect_failure("encoder sha256 mismatch", "train_oes90_pure_chunk.py",
                   *common, "--chunk_id", "0", "--n_chunks", str(DEMO_CHUNKS),
                   "--device", "cpu")
    (run_dir / "encoder_sha256.txt").write_text(good_enc)

    # --- guard 2: a mismatched embedding cache must stop stage 3 too --------------
    good_emb = (run_dir / "embeddings_sha256.txt").read_text()
    (run_dir / "embeddings_sha256.txt").write_text("f" * 64 + "\n")
    expect_failure("embedding cache mismatch", "train_oes90_pure_chunk.py",
                   *common, "--chunk_id", "0", "--n_chunks", str(DEMO_CHUNKS),
                   "--device", "cpu")
    (run_dir / "embeddings_sha256.txt").write_text(good_emb)

    for k in range(DEMO_CHUNKS):
        run("train_oes90_pure_chunk.py", *common, "--chunk_id", str(k),
            "--n_chunks", str(DEMO_CHUNKS), "--device", "cpu")
    for k in range(DEMO_CHUNKS):
        run("ig_oes90_pure.py", *common, "--chunk_id", str(k),
            "--n_chunks", str(DEMO_CHUNKS), "--device", "cpu", "--top_n", str(DEMO_TOP_N))

    # --- guard 3: merge must refuse chunks that disagree on the encoder -----------
    meta_path = run_dir / "ig_chunk_0_meta.json"
    saved = meta_path.read_text()
    blob = json.loads(saved)
    blob["encoder_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(blob))
    expect_failure("split-encoder provenance", "merge_oes90_pure_signatures.py",
                   *common, "--n_chunks", str(DEMO_CHUNKS))
    meta_path.write_text(saved)

    run("merge_oes90_pure_signatures.py", *common, "--n_chunks", str(DEMO_CHUNKS))
    run("run_oes90_pure_coupling.py", *common)
    run("run_oes90_pure_direction.py", *common)

    # --- checks ------------------------------------------------------------------
    import pandas as pd

    prep_meta = json.loads((run_dir / "prepare_meta.json").read_text())
    n_stim = len(prep_meta["shards"]["conditions"]) - 1  # minus the control
    expected_pairs = n_stim * (n_stim - 1) // 2

    print("\n===== checks =====", flush=True)
    for name, n in {
        "coupling_donor_degree.csv": expected_pairs,
        "direction_table.csv": expected_pairs,
    }.items():
        df = pd.read_csv(run_dir / name)
        assert len(df) == n, f"{name}: {len(df)} rows, expected {n}"
        print(f"  {name}: {len(df)} rows OK", flush=True)

    coup = pd.read_csv(run_dir / "coupling_donor_degree.csv")
    for col in ("q_donor", "coupled_q05", "coupled_q10", "donor_sign_p"):
        assert col in coup.columns, f"coupling table missing {col}"
    print(f"  coupling columns OK ({len(coup.columns)} columns)", flush=True)

    for which in ("main", "reserve"):
        sig = pd.read_parquet(run_dir / f"signatures_{which}.parquet")
        assert sig.cytokine.nunique() == n_stim, f"{which}: {sig.cytokine.nunique()} conditions"
        assert (sig.groupby("cytokine").size() == DEMO_TOP_N).all()
        print(f"  signatures_{which}.parquet: {n_stim} x {DEMO_TOP_N} OK", flush=True)

    stab = pd.read_csv(run_dir / "signature_stability.csv")
    assert len(stab) == n_stim and stab.jaccard.between(0, 1).all()
    print(f"  signature_stability.csv: median Jaccard {stab.jaccard.median():.3f} OK", flush=True)

    eng = pd.read_parquet(run_dir / "engagement_per_celltype.parquet")
    for col in ("condition_a", "condition_b", "sA_PB_norm", "sB_PA_norm", "cross_asym"):
        assert col in eng.columns, f"engagement table missing {col}"
    print(f"  engagement_per_celltype.parquet: {len(eng)} rows OK", flush=True)

    n_heads = len(list((run_dir / "models").glob("*_head.pt")))
    n_hist = len(list((run_dir / "history").glob("*_train.csv")))
    assert n_heads == n_stim, f"{n_heads} saved heads, expected {n_stim}"
    assert n_hist == n_stim, f"{n_hist} training histories, expected {n_stim}"
    print(f"  saved models/histories: {n_heads}/{n_hist} OK", flush=True)

    hist = pd.read_csv(run_dir / "encoder_history.csv")
    assert {"train_loss", "train_acc", "val_loss", "val_acc"} <= set(hist.columns)
    assert hist.val_loss.notna().all(), "encoder history has no validation curve"
    print(f"  encoder_history.csv: {len(hist)} epochs, val curve present OK", flush=True)

    assert (run_dir / "encoder.pt").exists() and (run_dir / "encoder_last.pt").exists()
    assert (run_dir / "embeddings" / "embeddings_meta.json").exists()
    print("  encoder + encoded tubes present OK", flush=True)

    # --- agnosticism: no pure-run module may pull in the benchmark constants ------
    leaked = [m for m in sys.modules if m.endswith("_full90_config")]
    assert not leaked, f"benchmark config leaked into the pure run: {leaked}"
    C.assert_agnostic()
    print("  agnosticism guard OK (no audited-pair constants in scope)", flush=True)

    print("\nALL DEMO CHECKS PASSED (harness only — no biology validated)", flush=True)

    if not args.keep and args.workdir is None:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"[demo] artifacts kept in {work}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
