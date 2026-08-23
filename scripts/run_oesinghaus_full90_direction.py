#!/usr/bin/env python
"""Stage 5 of the Oesinghaus full-90 DAG — cross_asym direction for all 4005 pairs.

`est.direction_table()` is the ONLY direction API (cascadir-values SKILL.md, CLAUDE.md
§28): the `cross_asym` column on the coupling table is the `M-M^T` difference-of-medians
approximation and differs in sign on some pairs, so direction is never read off it.

Also runs `est.benchmark(...)` on the 17 audited pairs
(`reports/cascade_pairs/cytokine_axes_audited.csv`, `counts_in_benchmark=True`) as a
REGRESSION CHECK — reported, never aborting. Two faithful fresh fits have already landed
near 0.65 rather than the published 15/17 = 0.88 (CLAUDE.md §27.6 at 6/11; the §31
recurrent-IG run at 11/17 despite a single shared encoder and the wide config), so a hard
88% gate would abort on run-to-run variance rather than on a bug. The symmetric
`directional_score` control is reported alongside and should sit near chance — that
contrast is the evidence the antisymmetric statistic is doing the work.

`scripts/retally_pipeline_against_audit.py` is deliberately NOT used: it consumes a
per-cell-type CSV that `direction_table` does not emit, and producing one would mean
calling the module-level `directional_asymmetry_test`. `est.benchmark` scores the same
signs on the same denominator and adds the symmetric control.

Usage (cluster, CPU):
  python scripts/run_oesinghaus_full90_direction.py --output_dir results/oes_full90
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _full90_config as C  # noqa: E402
from _full90_estimator import build_estimator  # noqa: E402

PUBLISHED_CORRECT, PUBLISHED_TOTAL = 15, 17


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="results/oes_full90")
    ap.add_argument("--pairs_shard", type=int, default=None,
                    help="Optional: only score pairs i where i %% n_shards == this.")
    ap.add_argument("--n_shards", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.output_dir)
    est, provenance = build_estimator(out)

    conds = sorted(est.signatures)
    pairs = [(conds[i], conds[j]) for i in range(len(conds)) for j in range(i + 1, len(conds))]
    suffix = ""
    if args.pairs_shard is not None and args.n_shards > 1:
        pairs = [p for k, p in enumerate(pairs) if k % args.n_shards == args.pairs_shard]
        suffix = f"_shard{args.pairs_shard}"
        print(f"[shard] {args.pairs_shard}/{args.n_shards}", flush=True)
    print(f"[direction] scoring {len(pairs)} pairs "
          f"(n_null_perms={C.N_NULL_PERMS})", flush=True)

    t0 = time.time()
    direction = est.direction_table(pairs=pairs)
    elapsed = time.time() - t0
    dir_path = out / f"direction_table{suffix}.csv"
    direction.to_csv(dir_path, index=False)
    print(f"[direction] done in {elapsed/60:.1f} min -> {dir_path}", flush=True)

    meta = {
        **provenance,
        "n_pairs": int(len(direction)),
        "n_null_perms": C.N_NULL_PERMS,
        "elapsed_min": round(elapsed / 60, 1),
        "classification_counts": direction["classification"].value_counts().to_dict(),
    }

    if suffix:  # sharded run: benchmark belongs to the full-table job
        C.write_json(out / f"direction_meta{suffix}.json", meta)
        return 0

    labels = C.load_audited_labels()
    have = set(est.signatures)
    usable = [(u, d) for (u, d) in labels if u in have and d in have]
    dropped = [p for p in labels if p not in usable]
    if dropped:
        # Not expected on the real panel (the audited cytokines are a subset of the 90);
        # if it fires, a name in the audit does not match the manifest spelling.
        print(f"[benchmark] WARNING: {len(dropped)} audited pairs have no signature and "
              f"are excluded from the denominator: {dropped}", flush=True)
    print(f"\n[benchmark] regression check on {len(usable)} of {len(labels)} audited pairs "
          f"(published anchor {PUBLISHED_CORRECT}/{PUBLISHED_TOTAL})", flush=True)
    if not usable:
        print("[benchmark] no audited pair is scorable on this panel — skipping", flush=True)
        meta["benchmark"] = {"n_labeled": len(labels), "n_scorable": 0,
                             "note": "no audited cytokine present in this fit"}
        C.write_json(out / "direction_meta.json", meta)
        return 0
    bench = est.benchmark(usable)
    print(bench.summary(), flush=True)
    bench.table.to_csv(out / "benchmark_regression.csv", index=False)

    lines = [
        bench.summary(),
        "",
        f"published anchor: {PUBLISHED_CORRECT}/{PUBLISHED_TOTAL} = "
        f"{PUBLISHED_CORRECT/PUBLISHED_TOTAL:.3f} (binary_ig_all24, 24-cytokine panel)",
        f"this fit:         {bench.cross_accuracy:.3f} over {bench.n_scored} non-AMBIGUOUS "
        "of the same audited pairs",
        "",
        "REPORTED, NOT A GATE. Prior fresh fits: CLAUDE.md §27.6 = 6/11; §31 recurrent-IG "
        "= 11/17 (0.65) despite a shared encoder and the wide config. A shortfall here is "
        "run-to-run variance unless the symmetric control also rises toward the "
        "cross_asym number.",
        "",
        "This is a SEPARATE fit from the published 24/45-cytokine runs. Never average or "
        "mix the two (CLAUDE.md §26.3, the Sheu mixed-provenance lesson).",
    ]
    (out / "benchmark_summary.txt").write_text("\n".join(lines) + "\n")

    meta["benchmark"] = {
        "n_audited_total": len(labels),
        "n_audited_dropped": len(dropped),
        "n_labeled": bench.n_labeled,
        "n_found": bench.n_found,
        "n_scored": bench.n_scored,
        "cross_accuracy": bench.cross_accuracy,
        "cross_accuracy_all": bench.cross_accuracy_all,
        "dirscore_accuracy": bench.dirscore_accuracy,
        "n_null_pass": bench.n_null_pass,
        "classification_counts": bench.classification_counts,
        "published_anchor": f"{PUBLISHED_CORRECT}/{PUBLISHED_TOTAL}",
    }
    C.write_json(out / "direction_meta.json", meta)
    print(f"\n[done] {dir_path}, benchmark_regression.csv, benchmark_summary.txt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
