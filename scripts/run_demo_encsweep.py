"""Local end-to-end de-risk for the encoder-breadth sweep — HARNESS ONLY, not biology.

Runs prepare -> encoder -> train -> ig -> analysis on `tests/make_demo_data.py`'s simulated
data at toy scale, and asserts the things that would silently corrupt the real sweep:

  * the arms really do get DIFFERENT encoders (if they did not, the sweep measures nothing);
  * each arm's Stage-1 set has the arm's condition count but the SAME cell budget — the
    whole design rests on breadth being the only thing that varies;
  * a head trained under one arm's encoder REFUSES to be reloaded under another's;
  * no stage pulls in the benchmark constants (`assert_agnostic` stays true throughout).

The demo's numbers mean nothing — the genes are simulated. This checks plumbing only.
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

import _encsweep_config as C  # noqa: E402
from run_demo_oes90_pure import build_demo_shards, run  # noqa: E402

DEMO_TUBES_PER_GROUP = 2
DEMO_ARMS = ("pbs_only", "all90")


def patch_config_for_demo() -> None:
    """Toy scale. runpy executes each stage in-process, so these patches are visible."""
    C.MAIN_TUBE_INDICES = [0, 1]
    C.RESERVE_TUBE_INDICES = []
    C.VAL_DONORS = []            # the demo has 3 donors; holding two out would leave one
    C.EMBED_DIM = 64
    C.HIDDEN_DIMS = (64, 64)
    C.ATTENTION_HIDDEN_DIM = 16
    C.STAGE1_EPOCHS = 3
    C.STAGE2_EPOCHS = 3
    C.TOP_N = 20
    C.N_CHUNKS = 2
    C.PANEL_SIZE = 4
    C.ARMS = DEMO_ARMS
    C.ARM_N_CONDITIONS = {"pbs_only": 0, "all90": None}
    C.STAGE1_TOTAL_CELLS = 600
    C.PBS_CAP_PER_DONOR = 400
    C.STIM_CAP_PER_DONOR = 100


def assert_no_benchmark_references() -> None:
    """Static check: does any sweep stage REFERENCE a benchmark artefact?

    `assert_agnostic()` catches the runtime version (a stage importing `_full90_config`).
    This catches the static one, and walks the parsed tree rather than grepping, because a
    grep also matches prose — `_encsweep_config`'s own docstring explains that it does NOT
    import `_full90_config`. It carries a positive control: if the check does not fire on
    a file that genuinely uses these constants, the check itself is broken.
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
    stages = ["_encsweep_config", "prepare_encsweep", "prepare_s1sweep",
              "train_encsweep_encoder", "train_encsweep_chunk", "ig_encsweep",
              "analyze_encsweep"]
    offenders = {n: h for n in stages if (h := scan(scripts / f"{n}.py"))}
    if offenders:
        raise AssertionError(f"sweep stages reference benchmark artefacts: {offenders}")
    if not scan(scripts / "_full90_config.py"):
        raise AssertionError(
            "the benchmark-reference check did not fire on _full90_config.py, which "
            "genuinely uses those constants — the check itself is broken."
        )
    print(f"[guard] {len(stages)} sweep stages reference no benchmark artefact "
          "(positive control fired)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="encsweep_demo_"))
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
    # prepare_encsweep reads the HVG list from the manifest's directory.
    (Path(manifest).parent / "hvg_list.json").write_text(json.dumps(gene_names))

    print("\n===== building demo tube shards =====", flush=True)
    shard_meta = build_demo_shards(manifest, gene_names, shard_dir)
    print(f"  {shard_meta['n_tubes']} tubes, sha={shard_meta['shards_sha256'][:16]}...")

    run("prepare_encsweep.py", "--out_dir", str(run_dir),
        "--shard_dir", str(shard_dir), "--manifest", manifest)

    meta = json.loads((run_dir / "encsweep_meta.json").read_text())
    budgets = {a: meta["arms"][a]["n_cells"] for a in DEMO_ARMS}
    conds = {a: meta["arms"][a]["n_encoder_conditions"] for a in DEMO_ARMS}
    print(f"\n[check] per-arm Stage-1 cells {budgets}, encoder conditions {conds}")
    assert conds["pbs_only"] == 1, f"pbs_only should see 1 group, saw {conds['pbs_only']}"
    assert conds["all90"] > conds["pbs_only"], "arms do not differ in condition breadth"
    spread = max(budgets.values()) - min(budgets.values())
    assert spread <= 0.15 * max(budgets.values()), (
        f"Stage-1 cell budgets differ by {spread} cells across arms — breadth would be "
        "confounded with gradient exposure and the sweep would measure nothing."
    )

    for arm in DEMO_ARMS:
        run("train_encsweep_encoder.py", "--out_dir", str(run_dir),
            "--arm", arm, "--device", "cpu")
    shas = {a: (run_dir / a / "encoder_sha256.txt").read_text().strip() for a in DEMO_ARMS}
    print(f"\n[check] encoder digests: " +
          ", ".join(f"{a}={s[:12]}..." for a, s in shas.items()))
    assert len(set(shas.values())) == len(DEMO_ARMS), (
        "the arms produced IDENTICAL encoders — the sweep's independent variable is not "
        "actually varying."
    )

    for arm in DEMO_ARMS:
        for chunk in range(C.N_CHUNKS):
            run("train_encsweep_chunk.py", "--out_dir", str(run_dir), "--arm", arm,
                "--chunk_id", str(chunk), "--n_chunks", str(C.N_CHUNKS), "--device", "cpu")
            run("ig_encsweep.py", "--out_dir", str(run_dir), "--arm", arm,
                "--chunk_id", str(chunk), "--n_chunks", str(C.N_CHUNKS),
                "--device", "cpu", "--top_n", str(C.TOP_N))

    # A head must refuse to be recombined with a different arm's encoder.
    print("\n===== cross-arm guard =====", flush=True)
    import _oes90_pure_estimator as E

    a, b = DEMO_ARMS
    enc_a, meta_a = E.load_encoder(run_dir / a, device="cpu", verbose=False)
    panel = meta["panel"]
    try:
        E.load_model(run_dir / b, panel[0], enc_a, meta_a["sha256"], device="cpu")
    except AssertionError as exc:
        print(f"[guard] cross-arm reload correctly refused: {exc}"[:160], flush=True)
    else:
        raise AssertionError(
            "a head trained under one arm's encoder loaded happily under another's — "
            "the §27.6 digest guard is broken and arms could silently mix."
        )

    run("analyze_encsweep.py", "--out_dir", str(run_dir), "--top_n", str(C.TOP_N))

    import pandas as pd

    div = pd.read_csv(run_dir / "arm_diversity.csv")
    assert len(div) == len(DEMO_ARMS), f"expected {len(DEMO_ARMS)} arms, got {len(div)}"
    assert (run_dir / "ARM_COMPARISON.md").exists(), "ARM_COMPARISON.md was not written"
    print("\n" + div.to_string(index=False))

    C.assert_agnostic()
    print("\n[PASS] encoder-breadth sweep runs end-to-end; arms differ; budgets matched; "
          "cross-arm guard fires; both agnosticism guards hold.")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"[demo] kept {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
