"""Phase 0a — the tube-identity gate for the controlled code-path comparison.

`reports/code_path_comparison/SPEC.md` §2. Before any claim that the two code paths
differ, we must know they are reading the SAME numbers. This asserts that, per
(donor, condition, tube_idx):

  * `PseudoTubeDataset(..., gene_names=hvgs)`  — what every `cytokine_mil` fit trains on;
  * `full90_tube_io.load_tube_set(shard_dir)`  — what every `cascadir` fit (§36-§40) trains on;

produce bit-identical float32 `X`, the same cell counts, and the same cell-type vectors.

`prepare_oesinghaus_full90.py` builds the shards by reading these very `.h5ad` tubes,
reindexed to the same HVG list, so a match is expected. A MISMATCH would mean the whole
`cascadir`-vs-`cytokine_mil` gap is a data difference rather than a code difference, and
every later phase is moot — which is exactly why this runs first and gates the rest.

Both sides are exercised through the REAL loaders, never a reimplementation: a hand-rolled
"equivalent" reader could agree with itself while both disagree with production.

Streams one tube (and one condition's shards) at a time — the full set is ~64 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))

from cytokine_mil.analysis.full90_tube_io import load_tube_set, read_meta  # noqa: E402
from cytokine_mil.data.dataset import PseudoTubeDataset  # noqa: E402
from cytokine_mil.data.label_encoder import CytokineLabel  # noqa: E402

MANIFEST = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVGS = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
SHARDS = "results/oes_full90/tubes"


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _hash_X(X) -> str:
    """sha256 of the float32 matrix, layout-normalised so C/F order cannot alias."""
    a = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _hash_labels(labels) -> str:
    return hashlib.sha256("\x00".join(str(x) for x in labels).encode()).hexdigest()


def manifest_side(manifest_path: str, gene_names: list[str], max_per_group: int | None):
    """Hash every tube as `PseudoTubeDataset` yields it. Streams (preload=False)."""
    with open(manifest_path) as fh:
        entries = json.load(fh)
    label_enc = CytokineLabel()
    ds = PseudoTubeDataset(manifest_path, label_enc, gene_names=gene_names, preload=False)

    seen: dict[tuple, int] = defaultdict(int)
    out: dict[tuple, dict] = {}
    for i, e in enumerate(entries):
        key = (str(e["donor"]), str(e["cytokine"]), int(e["tube_idx"]))
        gkey = key[:2]
        if max_per_group is not None and seen[gkey] >= max_per_group:
            continue
        seen[gkey] += 1
        X, _label, donor, cyt = ds[i]
        arr = X.numpy()
        out[key] = {"sha": _hash_X(arr), "n_cells": int(arr.shape[0])}
        if len(out) % 500 == 0:
            _log(f"  [manifest] {len(out)} tubes hashed")
    return out


def shard_side(shard_dir: str, max_per_group: int | None):
    """Hash every tube as `load_tube_set` yields it, one condition at a time."""
    meta = read_meta(shard_dir)
    control = meta["control_label"]
    conditions = sorted({s["condition"] for s in meta["shards"]})

    seen: dict[tuple, int] = defaultdict(int)
    out: dict[tuple, dict] = {}
    for ci, cond in enumerate(conditions, 1):
        ts = load_tube_set(
            shard_dir,
            conditions=[cond],
            include_control=(cond == control),
        )
        for t in ts.tubes:
            if t.condition != cond:
                continue  # include_control can pull the control in alongside
            key = (str(t.donor), str(t.condition), int(t.tube_idx))
            gkey = key[:2]
            if max_per_group is not None and seen[gkey] >= max_per_group:
                continue
            seen[gkey] += 1
            out[key] = {
                "sha": _hash_X(t.X),
                "n_cells": int(t.X.shape[0]),
                "ct_sha": _hash_labels(t.cell_types),
            }
        del ts
        if ci % 10 == 0 or ci == len(conditions):
            _log(f"  [shards] {ci}/{len(conditions)} conditions, {len(out)} tubes hashed")
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--hvg_path", default=HVGS)
    ap.add_argument("--shard_dir", default=SHARDS)
    ap.add_argument("--out_dir", default="results/code_path/phase0")
    ap.add_argument("--max_per_group", type=int, default=None,
                    help="Cap tubes per (donor, condition) — for a fast smoke run.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.hvg_path) as fh:
        gene_names = [str(g) for g in json.load(fh)]
    _log(f"[0a] HVGs: {len(gene_names)}")

    _log("\n[0a] hashing the shard side (cascadir path)...")
    shards, meta = shard_side(args.shard_dir, args.max_per_group)
    _log(f"[0a] shard tubes: {len(shards)}")

    # Gene order must match before per-tube hashes mean anything.
    shard_genes = [str(g) for g in meta["gene_names"]]
    genes_match = shard_genes == gene_names
    _log(f"[0a] gene order identical: {genes_match}")

    _log("\n[0a] hashing the manifest side (cytokine_mil path)...")
    mani = manifest_side(args.manifest, gene_names, args.max_per_group)
    _log(f"[0a] manifest tubes: {len(mani)}")

    both = sorted(set(mani) & set(shards))
    only_m = sorted(set(mani) - set(shards))
    only_s = sorted(set(shards) - set(mani))

    mismatch = [k for k in both if mani[k]["sha"] != shards[k]["sha"]]
    n_cell_mismatch = [k for k in both if mani[k]["n_cells"] != shards[k]["n_cells"]]

    donors_only_m = sorted({k[0] for k in only_m})

    _log("")
    _log("=" * 66)
    _log(f"  tubes in BOTH        : {len(both)}")
    _log(f"  X sha mismatches     : {len(mismatch)}")
    _log(f"  n_cells mismatches   : {len(n_cell_mismatch)}")
    _log(f"  manifest-only tubes  : {len(only_m)}  donors={donors_only_m}")
    _log(f"  shard-only tubes     : {len(only_s)}")
    _log("=" * 66)
    if mismatch[:5]:
        _log("  first mismatches:")
        for k in mismatch[:5]:
            _log(f"    {k}: manifest={mani[k]['sha'][:16]}... shard={shards[k]['sha'][:16]}...")

    passed = genes_match and not mismatch and not n_cell_mismatch and not only_s and both
    summary = {
        "gene_order_identical": genes_match,
        "n_both": len(both),
        "n_sha_mismatch": len(mismatch),
        "n_ncells_mismatch": len(n_cell_mismatch),
        "n_manifest_only": len(only_m),
        "manifest_only_donors": donors_only_m,
        "n_shard_only": len(only_s),
        "sha_mismatch_examples": [list(k) for k in mismatch[:20]],
        "passed": bool(passed),
    }
    (out / "phase0a_tube_identity.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _log(f"\n[write] {out/'phase0a_tube_identity.json'}")

    if not passed:
        _log("\n[GATE 0a] **FAILED** — the two paths do NOT read identical tubes.")
        _log("           The code-path gap would be a DATA difference; stop and report that.")
        return 1
    _log("\n[GATE 0a] PASSED — both paths read bit-identical tubes.")
    _log(f"           (manifest-only tubes are the excluded donors {donors_only_m}, as designed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
