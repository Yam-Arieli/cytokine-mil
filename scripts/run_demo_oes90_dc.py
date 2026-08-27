#!/usr/bin/env python
"""Local end-to-end smoke test of the §40 dropout+curation DAG (harness, NOT biology).

Runs every stage on the synthetic demo fixture (`tests/make_demo_data.py`: 10 cytokines +
PBS, 3 donors, 5 cell types, 200 genes, 4 tubes per (donor, condition)) with tiny epoch
counts, so a wiring bug surfaces on a laptop in seconds instead of after hours of cluster
time. `build_demo_shards` and `run` are imported from §37's demo rather than re-written —
stage 0 is literally §37's script, so the fixtures must match.

On top of §37's guards it asserts the four things §40 adds:
  * the encoder records its dropout and is the FINAL-epoch checkpoint (restore_best=False),
    not the best-validation one — the whole point of §40's fixed Stage-1 schedule;
  * the curation cap is DERIVED, and matches an independent recomputation here;
  * curated signatures really are shorter, contain no over-cap gene, and the report
    accounts for every input condition;
  * BOTH arms (curated and raw) produce a coupling table with the donor-level gate and a
    direction table merged from shards.

Nothing here validates the science — the demo data has no planted biology. It validates
that the artifacts flow, the shapes are right, and the assertions fire.

Usage:  python scripts/run_demo_oes90_dc.py [--workdir DIR] [--keep]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "cascadir" / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _oes90_dc_config as C  # noqa: E402
import _oes90_pure_config as P  # noqa: E402  (stage 0 is §37's script; it reads its own config)
from run_demo_oes90_pure import build_demo_shards, expect_failure, run  # noqa: E402

DEMO_TUBES_PER_GROUP = 4
DEMO_MAIN = [0, 1]
DEMO_RESERVE = [2, 3]
DEMO_STAGE1_EPOCHS = 6
DEMO_STAGE2_EPOCHS = 6
DEMO_CHUNKS = 3
DEMO_DIR_SHARDS = 2
DEMO_NULL_PERMS = 5
DEMO_TOP_N = 20
DEMO_CELLS_PER_CYTOKINE = 120
# 10 demo conditions at top-20 out of 200 genes => E[count/gene] = 1.0, so a cap of 3
# would remove almost nothing and never exercise the curation path. Target a stringency
# that actually bites at demo scale; the cluster run derives its own from the real numbers.
DEMO_CURATION_TARGET = 0.25


def expect_refusal(what: str, script: str, *argv: str) -> None:
    """Like §37's `expect_failure`, but also accepts an OS-level refusal.

    §37's helper catches SystemExit/AssertionError, which covers its three digest guards.
    §40's shard merge refuses a missing shard with FileNotFoundError — the right exception
    for a missing file, and still a refusal.
    """
    try:
        run(script, *argv)
    except (SystemExit, AssertionError, FileNotFoundError, OSError) as exc:
        print(f"[guard] {what} correctly refused: {type(exc).__name__}: {exc}", flush=True)
    else:
        raise AssertionError(f"{what} guard did NOT fire — the protection is broken")


def patch_config_for_demo() -> None:
    """Tiny settings, applied once to the shared modules every stage imports.

    §40's config re-exports §37's plumbing but keeps its own tube/epoch constants, and
    stage 0 is §37's script reading §37's config — so both modules have to be patched or
    the two halves of the demo would disagree about the tube split.
    """
    for mod in (C, P):
        mod.MAIN_TUBE_INDICES = DEMO_MAIN
        mod.RESERVE_TUBE_INDICES = DEMO_RESERVE
        mod.CELLS_PER_CYTOKINE = DEMO_CELLS_PER_CYTOKINE
        mod.N_CHUNKS = DEMO_CHUNKS
        mod.N_NULL_PERMS = DEMO_NULL_PERMS
        mod.TOP_N = DEMO_TOP_N
        mod.VAL_DONORS = []  # the demo has 3 donors; holding two out would leave one
    C.STAGE1_EPOCHS = DEMO_STAGE1_EPOCHS
    C.STAGE2_EPOCHS = DEMO_STAGE2_EPOCHS
    C.N_DIRECTION_SHARDS = DEMO_DIR_SHARDS
    C.CURATION_TARGET_NULL_REMOVAL = DEMO_CURATION_TARGET


def assert_no_benchmark_references() -> None:
    """Static check: does any §40 stage REFERENCE a benchmark artefact?

    `assert_agnostic()` catches the runtime version (a stage importing `_full90_config`).
    This catches the static one, and walks the parsed tree rather than grepping, because a
    grep also matches prose — `_oes90_dc_config`'s own docstring names `AUDITED_CSV` while
    explaining that it is out of scope. It carries a positive control: if the check does
    not fire on a file that genuinely uses these constants, the check itself is broken.
    """
    import ast

    banned = {"_full90_config", "AUDITED_CSV", "PUBLISHED_COUPLING_CSV",
              "load_audited_labels", "cytokine_axes_audited.csv", "cytokine_axes.csv",
              "donor_coupling_hub_IG_vsPBS.csv", "binary_ig_all24",
              "literature_review_aggregate.json"}

    def scan(path: Path) -> list:
        tree = ast.parse(path.read_text())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)) and ast.get_docstring(node) is not None:
                docstrings.add(id(node.body[0].value))
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hits += [(node.lineno, a.name) for a in node.names if a.name in banned]
            elif isinstance(node, ast.ImportFrom) and (node.module or "") in banned:
                hits.append((node.lineno, node.module))
            elif isinstance(node, ast.Name) and node.id in banned:
                hits.append((node.lineno, node.id))
            elif isinstance(node, ast.Attribute) and node.attr in banned:
                hits.append((node.lineno, node.attr))
            elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and id(node) not in docstrings and node.value in banned):
                hits.append((node.lineno, node.value))
        return hits

    scripts = REPO_ROOT / "scripts"
    # This file is the checker, not a stage: it necessarily contains the banned strings
    # in `banned` above, so scanning it would always self-trip.
    stages = ["_oes90_dc_config", "_oes90_dc_estimator", "prepare_oes90_pure",
              "train_oes90_dc_encoder", "encode_oes90_dc_tubes", "train_oes90_dc_chunk",
              "ig_oes90_dc", "merge_oes90_dc_signatures", "run_oes90_dc_coupling",
              "run_oes90_dc_direction", "merge_oes90_dc_direction"]
    offenders = {n: h for n in stages if (h := scan(scripts / f"{n}.py"))}
    if offenders:
        raise AssertionError(f"§40 stages reference benchmark artefacts: {offenders}")
    if not scan(scripts / "_full90_config.py"):
        raise AssertionError(
            "the benchmark-reference check did not fire on _full90_config.py, which "
            "genuinely uses those constants — the check itself is broken."
        )
    print(f"[guard] {len(stages)} §40 stages reference no benchmark artefact "
          "(positive control fired)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir or tempfile.mkdtemp(prefix="oes90_dc_demo_"))
    demo, run_dir, shard_dir = work / "demo", work / "run", work / "tubes"
    print(f"[demo] workdir {work}", flush=True)

    patch_config_for_demo()
    C.assert_agnostic()
    assert_no_benchmark_references()

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
    # Stage 0 is §37's script, run verbatim — §40 changes nothing about the tube split.
    run("prepare_oes90_pure.py", "--shard_dir", str(shard_dir),
        "--manifest_path", manifest, "--hvg_path", str(hvg),
        "--exclude_donors", *common)
    run("train_oes90_dc_encoder.py", *common, "--device", "cpu")
    run("encode_oes90_dc_tubes.py", *common, "--device", "cpu")

    # --- guard 1: a mismatched encoder must stop stage 3 before it trains ---------
    good_enc = (run_dir / "encoder_sha256.txt").read_text()
    (run_dir / "encoder_sha256.txt").write_text("0" * 64 + "\n")
    expect_failure("encoder sha256 mismatch", "train_oes90_dc_chunk.py",
                   *common, "--chunk_id", "0", "--n_chunks", str(DEMO_CHUNKS),
                   "--device", "cpu")
    (run_dir / "encoder_sha256.txt").write_text(good_enc)

    # --- guard 2: a mismatched embedding cache must stop stage 3 too --------------
    good_emb = (run_dir / "embeddings_sha256.txt").read_text()
    (run_dir / "embeddings_sha256.txt").write_text("f" * 64 + "\n")
    expect_failure("embedding cache mismatch", "train_oes90_dc_chunk.py",
                   *common, "--chunk_id", "0", "--n_chunks", str(DEMO_CHUNKS),
                   "--device", "cpu")
    (run_dir / "embeddings_sha256.txt").write_text(good_emb)

    for k in range(DEMO_CHUNKS):
        run("train_oes90_dc_chunk.py", *common, "--chunk_id", str(k),
            "--n_chunks", str(DEMO_CHUNKS), "--device", "cpu")
    for k in range(DEMO_CHUNKS):
        run("ig_oes90_dc.py", *common, "--chunk_id", str(k),
            "--n_chunks", str(DEMO_CHUNKS), "--device", "cpu", "--top_n", str(DEMO_TOP_N))

    # --- guard 3: merge must refuse chunks that disagree on the encoder -----------
    meta_path = run_dir / "ig_chunk_0_meta.json"
    saved = meta_path.read_text()
    blob = json.loads(saved)
    blob["encoder_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(blob))
    expect_failure("split-encoder provenance", "merge_oes90_dc_signatures.py",
                   *common, "--n_chunks", str(DEMO_CHUNKS))
    meta_path.write_text(saved)

    run("merge_oes90_dc_signatures.py", *common, "--n_chunks", str(DEMO_CHUNKS))

    for arm in ("curated", "raw"):
        run("run_oes90_dc_coupling.py", *common, "--arm", arm)
        for k in range(DEMO_DIR_SHARDS):
            run("run_oes90_dc_direction.py", *common, "--arm", arm,
                "--pairs_shard", str(k), "--n_shards", str(DEMO_DIR_SHARDS))
        run("merge_oes90_dc_direction.py", *common, "--arm", arm,
            "--n_shards", str(DEMO_DIR_SHARDS))

    # --- guard 4: the shard merge must refuse an incomplete arm ------------------
    stray = run_dir / f"direction_table_curated_shard{DEMO_DIR_SHARDS - 1}.csv"
    stash = stray.read_text()
    stray.unlink()
    expect_refusal("incomplete direction shards", "merge_oes90_dc_direction.py",
                   *common, "--arm", "curated", "--n_shards", str(DEMO_DIR_SHARDS))
    stray.write_text(stash)

    # --- checks ------------------------------------------------------------------
    import pandas as pd

    from cascadir.signatures import null_calibrated_max_occurrences

    prep_meta = json.loads((run_dir / "prepare_meta.json").read_text())
    n_stim = len(prep_meta["shards"]["conditions"]) - 1  # minus the control

    print("\n===== checks =====", flush=True)

    # -- §40: the encoder is the FINAL-epoch checkpoint, and it records its dropout
    import torch

    enc_meta = json.loads((run_dir / "encoder_meta.json").read_text())
    assert enc_meta["dropout"] == C.ENCODER_DROPOUT, enc_meta["dropout"]
    assert enc_meta["restore_best"] is False
    assert enc_meta["epochs_run"] == DEMO_STAGE1_EPOCHS, enc_meta["epochs_run"]
    sd_best = torch.load(run_dir / "encoder.pt", map_location="cpu")
    sd_last = torch.load(run_dir / "encoder_last.pt", map_location="cpu")
    assert all(torch.equal(sd_best[k], sd_last[k]) for k in sd_best), \
        "encoder.pt is not the final-epoch checkpoint — restore_best leaked"
    print(f"  encoder: dropout={enc_meta['dropout']}, {enc_meta['epochs_run']} fixed "
          f"epochs, final-epoch checkpoint OK (best was epoch {enc_meta['best_epoch']})",
          flush=True)

    hist = pd.read_csv(run_dir / "encoder_history.csv")
    assert {"train_loss", "train_acc", "val_loss", "val_acc"} <= set(hist.columns)
    assert hist.val_loss.notna().all(), "encoder history has no validation curve"
    assert len(hist) == DEMO_STAGE1_EPOCHS, "early stopping ran despite patience=None"
    print(f"  encoder_history.csv: {len(hist)} epochs, val curve recorded OK", flush=True)

    # -- §40: the curation cap is derived, and the curation actually happened
    cur_meta = json.loads((run_dir / "curation_meta.json").read_text())
    n_genes = len(gene_names)
    expected_cap = null_calibrated_max_occurrences(
        n_stim, DEMO_TOP_N, n_genes, target_null_removal=DEMO_CURATION_TARGET
    )
    assert cur_meta["cap"] == expected_cap, (cur_meta["cap"], expected_cap)
    print(f"  curation cap: {cur_meta['cap']} matches an independent recomputation "
          f"(K={n_stim}, top_n={DEMO_TOP_N}, G={n_genes}) OK", flush=True)
    print(f"  curation: removed {cur_meta['observed_removal_frac']:.1%} vs null "
          f"{cur_meta['expected_null_removal']:.1%} -> excess "
          f"{cur_meta['excess_removal']:+.1%}; {cur_meta['n_conditions_dropped']} "
          f"conditions dropped", flush=True)

    report = pd.read_csv(run_dir / "curation_report.csv")
    assert len(report) == n_stim, f"curation report covers {len(report)} of {n_stim}"
    assert (report.n_before == DEMO_TOP_N).all()
    assert (report.n_after <= report.n_before).all()
    print(f"  curation_report.csv: all {n_stim} input conditions accounted for OK",
          flush=True)

    raw_sig = pd.read_parquet(run_dir / "signatures_main.parquet")
    cur_sig = pd.read_parquet(run_dir / "signatures_main_curated.parquet")
    assert raw_sig.cytokine.nunique() == n_stim
    assert (raw_sig.groupby("cytokine").size() == DEMO_TOP_N).all()
    over = cur_sig.groupby("gene").cytokine.nunique()
    assert (over <= cur_meta["cap"]).all(), \
        f"curated signatures still contain over-cap genes: {over[over > cur_meta['cap']]}"
    assert len(cur_sig) < len(raw_sig), "curation removed nothing — the demo is not exercising it"
    # rank_ig must be re-densified 0..n-1 within each curated signature
    for _cond, g in cur_sig.groupby("cytokine"):
        assert list(g.sort_values("rank_ig").rank_ig) == list(range(len(g)))
    print(f"  signatures: raw {len(raw_sig)} rows -> curated {len(cur_sig)} rows, "
          f"no gene over cap, ranks re-densified OK", flush=True)

    for which in ("main", "reserve"):
        sig = pd.read_parquet(run_dir / f"signatures_{which}.parquet")
        assert sig.cytokine.nunique() == n_stim
        assert (sig.groupby("cytokine").size() == DEMO_TOP_N).all()
    stab = pd.read_csv(run_dir / "signature_stability.csv")
    assert len(stab) == n_stim and stab.jaccard.between(0, 1).all()
    cur_stab = pd.read_csv(run_dir / "signature_stability_curated.csv")
    assert len(cur_stab) <= n_stim and cur_stab.jaccard.between(0, 1).all()
    print(f"  stability: raw median {stab.jaccard.median():.3f}, curated median "
          f"{cur_stab.jaccard.median():.3f} OK", flush=True)

    # -- §40: BOTH arms produce a full coupling + direction table
    for arm in ("curated", "raw"):
        n_cond = int(json.loads(
            (run_dir / f"coupling_meta_{arm}.json").read_text())["n_conditions"])
        expected_pairs = n_cond * (n_cond - 1) // 2
        coup = pd.read_csv(run_dir / f"coupling_donor_degree_{arm}.csv")
        assert len(coup) == expected_pairs, f"{arm} coupling: {len(coup)}/{expected_pairs}"
        for col in ("q_donor", "coupled_q05", "coupled_q10", "donor_sign_p"):
            assert col in coup.columns, f"{arm} coupling table missing {col}"
        direction = pd.read_csv(run_dir / f"direction_table_{arm}.csv")
        assert len(direction) == expected_pairs, f"{arm} direction: {len(direction)}/{expected_pairs}"
        eng = pd.read_parquet(run_dir / f"engagement_per_celltype_{arm}.parquet")
        for col in ("condition_a", "condition_b", "sA_PB_norm", "sB_PA_norm", "cross_asym"):
            assert col in eng.columns, f"{arm} engagement table missing {col}"
        print(f"  arm={arm}: {n_cond} conditions, {expected_pairs} pairs, coupling + "
              f"direction (merged from {DEMO_DIR_SHARDS} shards) + {len(eng)} engagement "
              "rows OK", flush=True)

    n_heads = len(list((run_dir / "models").glob("*_head.pt")))
    n_hist = len(list((run_dir / "history").glob("*_train.csv")))
    assert n_heads == n_stim and n_hist == n_stim
    assert (run_dir / "embeddings" / "embeddings_meta.json").exists()
    print(f"  saved models/histories: {n_heads}/{n_hist}, encoded tubes present OK",
          flush=True)

    # --- agnosticism: no §40 module may pull in the benchmark constants -----------
    leaked = [m for m in sys.modules if m.endswith("_full90_config")]
    assert not leaked, f"benchmark config leaked into the §40 run: {leaked}"
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
