#!/usr/bin/env python
"""Benchmark cascadir on one forged large-cascade snapshot (one h5ad -> metrics).

Reads the planted ground truth from ``adata.uns["cascade_forge"]``, fits cascadir, and
scores: direction accuracy (cross_asym vs the symmetric directional_score control) on the
signed direct edges (feedback pair excluded), and signature-space coupling — recall on the
truly-coupled pairs and false-positive rate (incl. the isolated negative-control labels).

Usage:
    python scripts/benchmark_large_cascade.py --h5ad results/.../snapshot_t6.h5ad
Outputs (next to the h5ad): <stem>.metrics.json, <stem>.direction.csv, <stem>.coupling.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _edges(gt, key):
    return [tuple(str(x) for x in e) for e in gt.get(key, [])]


def _json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--coupling_alpha", type=float, default=0.05)
    p.add_argument("--out", default=None, help="output dir (default: alongside the h5ad)")
    p.add_argument("--encoder_epochs", type=int, default=None, help="override (smoke tests)")
    p.add_argument("--binary_epochs", type=int, default=None, help="override (smoke tests)")
    return p.parse_args(argv)


def main(argv=None):
    import cascadir as cd

    args = parse_args(argv)
    h5 = Path(args.h5ad)
    out_dir = Path(args.out) if args.out else h5.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = h5.stem

    adata = ad.read_h5ad(str(h5))
    gt = dict(adata.uns["cascade_forge"])
    cfg = dict(gt.get("config", {}))
    assume = str(cfg.get("output", "raw"))
    snapshot_time = float(gt.get("snapshot_time", float("nan")))

    direct = _edges(gt, "direct_edges")
    reachable = _edges(gt, "reachable_edges")
    bidir = {frozenset(p) for p in _edges(gt, "bidirectional_pairs")}
    labels = [str(x) for x in gt.get("labels", [])]
    in_edge = {x for e in direct for x in e}
    isolated = sorted(l for l in labels if l not in in_edge)
    signed_edges = [(a, b) for (a, b) in direct if frozenset((a, b)) not in bidir]

    print(f"[bench] {stem}: {adata.n_obs} cells x {adata.n_vars} genes | assume={assume} | "
          f"{len(signed_edges)} signed edges | isolated={isolated}", flush=True)

    train_config = None
    if args.encoder_epochs is not None or args.binary_epochs is not None:
        base = cd.TrainConfig()
        train_config = cd.TrainConfig(
            encoder_epochs=args.encoder_epochs if args.encoder_epochs is not None
            else base.encoder_epochs,
            binary_epochs=args.binary_epochs if args.binary_epochs is not None
            else base.binary_epochs,
        )
    est = cd.CascadeDirection(
        condition_col="condition", donor_col="donor", celltype_col="cell_type",
        control_label="PBS", train_config=train_config,
    ).fit(adata, assume=assume)

    # ---- direction ----
    bench = est.benchmark(signed_edges)
    bench.table.to_csv(out_dir / f"{stem}.direction.csv", index=False)

    # ---- coupling (existence) + false positives ----
    coupling = est.signature_coupling(donor_level=True, coupling_alpha=args.coupling_alpha)
    coupling.to_csv(out_dir / f"{stem}.coupling.csv", index=False)

    truth_pairs = {frozenset(e) for e in reachable}          # a<->b reachable = truly coupled
    all_pairs = {frozenset(p) for p in combinations(labels, 2)}
    non_truth = all_pairs - truth_pairs
    coupled_pred = {
        frozenset((str(r["condition_a"]), str(r["condition_b"])))
        for _, r in coupling.iterrows() if bool(r["coupled"])
    }
    tp = len(truth_pairs & coupled_pred)
    fp = len(non_truth & coupled_pred)
    isolated_pairs = {p for p in all_pairs if p & set(isolated)}
    isolated_fp = len(isolated_pairs & coupled_pred)

    metrics = {
        "snapshot": stem,
        "snapshot_time": snapshot_time,
        "responder_mode": cfg.get("responder_mode"),
        "effect_size": cfg.get("effect_size"),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_donors": cfg.get("n_donors"),
        # direction
        "n_signed_edges": len(signed_edges),
        "n_found": int(bench.n_found),
        "cross_accuracy": float(bench.cross_accuracy_all),
        "cross_accuracy_nonambig": float(bench.cross_accuracy),
        "dirscore_accuracy": float(bench.dirscore_accuracy),
        "classification_counts": {str(k): int(v) for k, v in bench.classification_counts.items()},
        # coupling
        "n_truth_coupled_pairs": len(truth_pairs),
        "coupling_recall": (tp / len(truth_pairs)) if truth_pairs else float("nan"),
        "coupling_false_positive_rate": (fp / len(non_truth)) if non_truth else float("nan"),
        "isolated_labels": isolated,
        "isolated_false_positives": isolated_fp,
        "n_isolated_pairs": len(isolated_pairs),
    }
    with open(out_dir / f"{stem}.metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    print(f"[bench] {stem}: direction cross_acc={metrics['cross_accuracy']:.0%} "
          f"(dirscore {metrics['dirscore_accuracy']:.0%}) | "
          f"coupling recall={metrics['coupling_recall']:.0%} "
          f"FP={metrics['coupling_false_positive_rate']:.0%} "
          f"isolated_FP={isolated_fp}/{len(isolated_pairs)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
