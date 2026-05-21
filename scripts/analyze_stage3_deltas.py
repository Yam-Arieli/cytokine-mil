"""Extract per-class p_correct AUC for Stage 2 baseline vs Stage 3 CA, both seeds.

Outputs a single CSV:
  seed,cytokine,stage2_train_auc,stage2_val_auc,stage3_train_auc,stage3_val_auc,
  train_delta,val_delta
"""
import pickle
import csv
import gc
import numpy as np
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results")
OUT_CSV = Path("/tmp/stage3_ca_deltas.csv")
SEEDS = [42, 123]


def auc_trajectory(traj):
    traj = np.asarray(traj, dtype=np.float64)
    T = len(traj)
    if T < 2:
        return float(traj[0]) if T else float("nan")
    # mean over epochs (= normalized AUC over a unit x-axis)
    return float(traj.mean())


def per_cytokine_aucs(records):
    per_cyt = defaultdict(list)
    for r in records:
        cyt = r.get("cytokine")
        if cyt is None:
            continue
        if "p_correct_trajectory" in r and r["p_correct_trajectory"] is not None:
            traj = r["p_correct_trajectory"]
            if len(traj) > 0:
                per_cyt[cyt].append(auc_trajectory(traj))
    return {c: float(np.mean(v)) for c, v in per_cyt.items() if v}


def main():
    rows = []
    for seed in SEEDS:
        stage2_dir = RESULTS_DIR / "oesinghaus_full_v2" / f"seed_{seed}"
        stage3_dir = RESULTS_DIR / "oesinghaus_stage3_ca" / f"seed_{seed}"

        print(f"=== seed {seed} ===", flush=True)
        s2_pkl = stage2_dir / "dynamics.pkl"
        s3_pkl = stage3_dir / "dynamics_stage3.pkl"

        print(f"  loading Stage 2 ({s2_pkl.stat().st_size/1e9:.1f} GB)...", flush=True)
        with open(s2_pkl, "rb") as f:
            d2 = pickle.load(f)
        print(f"  keys: {list(d2.keys())}", flush=True)
        if d2.get("records"):
            print(f"  sample record keys: {list(d2['records'][0].keys())}", flush=True)
        s2_train_auc = per_cytokine_aucs(d2.get("records", []))
        s2_val_auc = per_cytokine_aucs(d2.get("val_records", []))
        print(f"  Stage 2: {len(s2_train_auc)} train cyts, {len(s2_val_auc)} val cyts", flush=True)
        del d2
        gc.collect()

        print(f"  loading Stage 3 ({s3_pkl.stat().st_size/1e9:.1f} GB)...", flush=True)
        with open(s3_pkl, "rb") as f:
            d3 = pickle.load(f)
        print(f"  keys: {list(d3.keys())}", flush=True)
        if d3.get("records"):
            print(f"  sample record keys: {list(d3['records'][0].keys())}", flush=True)
        s3_train_auc = per_cytokine_aucs(d3.get("records", []))
        s3_val_auc = per_cytokine_aucs(d3.get("val_records", []))
        print(f"  Stage 3: {len(s3_train_auc)} train cyts, {len(s3_val_auc)} val cyts", flush=True)
        del d3
        gc.collect()

        all_cyts = sorted(set(s2_train_auc) | set(s3_train_auc) | set(s2_val_auc) | set(s3_val_auc))
        for c in all_cyts:
            s2t = s2_train_auc.get(c, float("nan"))
            s2v = s2_val_auc.get(c, float("nan"))
            s3t = s3_train_auc.get(c, float("nan"))
            s3v = s3_val_auc.get(c, float("nan"))
            rows.append({
                "seed": seed,
                "cytokine": c,
                "stage2_train_auc": s2t,
                "stage2_val_auc": s2v,
                "stage3_train_auc": s3t,
                "stage3_val_auc": s3v,
                "train_delta": s3t - s2t,
                "val_delta": s3v - s2v,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} with {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
