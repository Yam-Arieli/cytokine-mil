"""Quick script to compare final training accuracy across MIL seeds."""
import pickle
import numpy as np
from pathlib import Path

BASE = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full")

seeds = {
    "42":  "run_20260412_161758_seed42",
    "123": "run_20260412_161803_seed123",
    "7":   "run_20260412_161803_seed7",
}

for seed, run in seeds.items():
    path = BASE / run / "dynamics.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)

    train_recs = d["records"]
    val_recs   = d.get("val_records", [])

    train_final = np.mean([r["p_correct_trajectory"][-1] for r in train_recs])
    train_mean  = np.mean([np.mean(r["p_correct_trajectory"]) for r in train_recs])

    if val_recs:
        val_final = np.mean([r["p_correct_trajectory"][-1] for r in val_recs])
        val_mean  = np.mean([np.mean(r["p_correct_trajectory"]) for r in val_recs])
    else:
        val_final = val_mean = float("nan")

    print(f"Seed {seed:>3} | train_final={train_final:.4f}  train_mean={train_mean:.4f}"
          f"  | val_final={val_final:.4f}  val_mean={val_mean:.4f}"
          f"  | n_train={len(train_recs)}  n_val={len(val_recs)}")
