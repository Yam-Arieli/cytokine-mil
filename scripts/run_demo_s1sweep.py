"""Local end-to-end de-risk for the Stage-1 CONSTRUCTION sweep — HARNESS ONLY, not biology.

Runs prepare -> encoder -> train -> ig -> analysis on `tests/make_demo_data.py`'s simulated
data at toy scale, and asserts the things that would silently invalidate the real sweep:

  * the replica arms really are built one-tube-per-condition (ONE donor per condition), and
    the volume arms really are donor-balanced — if both were built the same way, the
    donor-structure contrast measures nothing;
  * `pub_replica` really does contain the held-out donors and `pub_replica_clean` really
    does not — that pair IS the leakage measurement, so getting it backwards would invert
    the headline;
  * `vol_small` really is smaller than `vol_large`;
  * the four arms produce DIFFERENT encoders;
  * a head trained under one arm's encoder REFUSES to reload under another's;
  * no stage pulls in the benchmark constants (`assert_agnostic` stays true throughout).

The demo's numbers mean nothing — the genes are simulated. This checks plumbing only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import _encsweep_config as C  # noqa: E402
from run_demo_encsweep import assert_no_benchmark_references  # noqa: E402
from run_demo_oes90_pure import build_demo_shards, run  # noqa: E402

DEMO_TUBES_PER_GROUP = 2
ARMS = ("pub_replica", "pub_replica_clean", "vol_small", "vol_large")
DEMO_VAL_DONORS = ["Donor3"]


def patch_config_for_demo() -> None:
    """Toy scale. runpy executes each stage in-process, so these patches are visible."""
    C.MAIN_TUBE_INDICES = [0, 1]
    C.RESERVE_TUBE_INDICES = []
    # Unlike the breadth demo, VAL_DONORS must be NON-empty here: the whole point of the
    # pub_replica / pub_replica_clean pair is that one includes held-out donors.
    C.VAL_DONORS = list(DEMO_VAL_DONORS)
    C.EMBED_DIM = 64
    C.HIDDEN_DIMS = (64, 64)
    C.ATTENTION_HIDDEN_DIM = 16
    C.STAGE1_EPOCHS = 3
    C.STAGE2_EPOCHS = 3
    C.TOP_N = 20
    C.N_CHUNKS = 2
    C.PANEL_SIZE = 4
    C.PBS_CAP_PER_DONOR = 400
    C.STIM_CAP_PER_DONOR = 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="s1sweep_demo_"))
    demo, run_dir, shard_dir = work / "demo", work / "run", work / "tubes"
    print(f"[demo] workdir {work}", flush=True)

    patch_config_for_demo()
    C.assert_agnostic()
    assert_no_benchmark_references()

    import scanpy as sc

    import make_demo_data as mdd

    mdd.N_PSEUDO_TUBES = DEMO_TUBES_PER_GROUP
    manifest = mdd.make_demo_data(str(demo))
    entries = json.loads(Path(manifest).read_text())
    tube0 = entries[0]["path"]
    gene_names = [str(g) for g in sc.read_h5ad(tube0).var_names]
    (Path(manifest).parent / "hvg_list.json").write_text(json.dumps(gene_names))

    # Mirror the real setup: the SHARDS exclude the held-out donors, while the source
    # manifest still carries them — which is exactly what `pub_replica` needs to read.
    clean_manifest = Path(manifest).parent / "manifest_clean.json"
    clean_manifest.write_text(json.dumps(
        [e for e in entries if e["donor"] not in set(DEMO_VAL_DONORS)]))

    print("\n===== building demo tube shards (held-out donors excluded) =====", flush=True)
    shard_meta = build_demo_shards(str(clean_manifest), gene_names, shard_dir)
    print(f"  {shard_meta['n_tubes']} tubes, sha={shard_meta['shards_sha256'][:16]}...")

    run("prepare_s1sweep.py", "--out_dir", str(run_dir), "--shard_dir", str(shard_dir),
        "--manifest", manifest, "--vol_small", "200", "--vol_large", "600")

    meta = json.loads((run_dir / "encsweep_meta.json").read_text())
    a = meta["arms"]
    print("\n[check] arm construction:")
    for arm in ARMS:
        print(f"    {arm:18s} n_cells={a[arm]['n_cells']:5d} "
              f"donors/cond={a[arm]['donors_per_condition']} "
              f"val_donors_in={a[arm]['includes_val_donors']}")

    for arm in ("pub_replica", "pub_replica_clean"):
        assert a[arm]["donors_per_condition"] == 1, (
            f"{arm} should contribute ONE donor per condition (build_stage1_manifest's "
            f"rule); got {a[arm]['donors_per_condition']}")
    for arm in ("vol_small", "vol_large"):
        assert a[arm]["donors_per_condition"] > 1, (
            f"{arm} should be donor-BALANCED; got {a[arm]['donors_per_condition']} donors")

    assert a["pub_replica"]["includes_val_donors"], (
        "pub_replica must contain the held-out donors — that arm IS the leakage "
        "measurement, and without them it measures nothing.")
    assert not a["pub_replica_clean"]["includes_val_donors"], (
        "pub_replica_clean must NOT contain held-out donors, or the leakage contrast is "
        "between two leaky arms.")
    for arm in ("vol_small", "vol_large"):
        assert not a[arm]["includes_val_donors"], f"{arm} leaked held-out donors"
    assert a["vol_small"]["n_cells"] < a["vol_large"]["n_cells"], (
        "vol_small is not smaller than vol_large — the volume contrast is empty.")

    for arm in ARMS:
        run("train_encsweep_encoder.py", "--out_dir", str(run_dir),
            "--arm", arm, "--device", "cpu")
    shas = {x: (run_dir / x / "encoder_sha256.txt").read_text().strip() for x in ARMS}
    print("\n[check] encoder digests: " + ", ".join(f"{k}={v[:10]}..." for k, v in shas.items()))
    assert len(set(shas.values())) == len(ARMS), (
        "two arms produced IDENTICAL encoders — an independent variable is not varying.")

    for arm in ARMS:
        for chunk in range(C.N_CHUNKS):
            run("train_encsweep_chunk.py", "--out_dir", str(run_dir), "--arm", arm,
                "--chunk_id", str(chunk), "--n_chunks", str(C.N_CHUNKS), "--device", "cpu")
            run("ig_encsweep.py", "--out_dir", str(run_dir), "--arm", arm,
                "--chunk_id", str(chunk), "--n_chunks", str(C.N_CHUNKS),
                "--device", "cpu", "--top_n", str(C.TOP_N))

    print("\n===== cross-arm guard =====", flush=True)
    import _oes90_pure_estimator as E

    enc_a, meta_a = E.load_encoder(run_dir / ARMS[0], device="cpu", verbose=False)
    try:
        E.load_model(run_dir / ARMS[1], meta["panel"][0], enc_a, meta_a["sha256"],
                     device="cpu")
    except AssertionError as exc:
        print(f"[guard] cross-arm reload correctly refused: {exc}"[:160], flush=True)
    else:
        raise AssertionError(
            "a head trained under one arm's encoder loaded happily under another's — "
            "the §27.6 digest guard is broken and arms could silently mix.")

    run("analyze_encsweep.py", "--out_dir", str(run_dir), "--top_n", str(C.TOP_N))

    import pandas as pd

    div = pd.read_csv(run_dir / "arm_diversity.csv")
    assert len(div) == len(ARMS), f"expected {len(ARMS)} arms in the table, got {len(div)}"
    assert (run_dir / "ARM_COMPARISON.md").exists(), "ARM_COMPARISON.md was not written"
    print("\n" + div[["arm", "n_stage1_cells", "top5_pool", "mean_jaccard",
                      "distinct_genes"]].to_string(index=False))

    C.assert_agnostic()
    print("\n[PASS] Stage-1 construction sweep runs end-to-end; replica arms are "
          "one-donor-per-condition and volume arms are balanced; the leakage pair is "
          "correctly signed; encoders differ; cross-arm guard fires.")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"[demo] kept {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
