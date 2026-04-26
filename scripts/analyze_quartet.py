"""
Quartet extraction: A → X → B → Y

For each cytokine pair (A→B) with strong cross-seed asymmetry:
  - X(A,B) = argmax_T [ attention_A(T, late) × bias(A,B,T, late) ]
             cell type that is both a primary A-responder and displaced toward B
  - Y(A,B) = argmax_T [ bias(A,B,T, late) ]
             cell type most displaced toward B's centroid in A-tubes
  - Temporal: epoch of peak attention_A(X) vs epoch of peak attention_A(Y)
  - X ≠ Y → true quartet; X == Y → triplet only

Requires per-seed:
  - geo_trajectory.pkl  (from run_geo_trajectory.py with --checkpoint_subdir checkpoints_stage3)
  - attention_trajectory.pkl  (from extract_attention_trajectory.py with --checkpoint_subdir checkpoints_stage3)

Usage:
    python scripts/analyze_quartet.py --seeds_dir results/oesinghaus_full --seeds 11 12 13 14 15
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
import sys; sys.path.insert(0, str(REPO_ROOT))

LATE_FRACTION = 0.3   # final 30% of epochs used for "late" averages
ASYM_THRESHOLD = 2.0  # minimum mean-final asymmetry to consider a pair
MIN_SEEDS = 4         # X and Y must agree in at least this many seeds

# Pre-registered pairs to always report (from .claude/preregistration_quartet.md)
PREREGISTERED = [
    ("IL-12",    "IFN-gamma",  "NK CD56bright", "CD14 Mono",    "positive control"),
    ("TNF-alpha","IL-6",       "CD14 Mono",     "B Naive",       "positive prediction"),
    ("IL-1-beta","IL-6",       "CD14 Mono",     "B Naive",       "positive prediction (if in dataset)"),
    ("IFN-beta", "IL-27",      "pDC",           "NK",            "positive prediction"),
    ("IL-6",     "IL-10",      None,            None,            "negative control — expect no quartet"),
    ("IFN-gamma","IL-12",      None,            None,            "negative control — expect weaker"),
]


def _log(msg=""):
    print(msg, flush=True)


def _load_geo(seed_dir: Path) -> dict:
    p = seed_dir / "geo_trajectory.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _load_attn(seed_dir: Path) -> dict:
    p = seed_dir / "attention_trajectory.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _late_mean(arr, late_fraction=LATE_FRACTION):
    """Mean over the last late_fraction of the array."""
    n = max(1, int(len(arr) * late_fraction))
    return float(np.mean(arr[-n:]))


def _peak_epoch(arr, epochs):
    """Return the epoch (actual number) at which arr peaks."""
    idx = int(np.argmax(arr))
    return epochs[idx]


def extract_quartet_one_seed(geo: dict, attn: dict) -> dict:
    """
    For one seed, extract X and Y for every (A,B) pair above threshold.

    Returns: {(a_name, b_name): {"X": str, "Y": str, "joint_X": float,
                                  "bias_Y": float, "asym": float,
                                  "X_peak_epoch": int, "Y_peak_epoch": int,
                                  "is_quartet": bool}}
    """
    epochs        = geo["epochs"]
    asym_traj     = geo["asymmetry_traj"]    # (T, K, K)
    bias_traj     = geo["bias_traj"]         # (T, K, K, n_ct)
    cyt_names     = geo["cytokine_names"]
    cell_types    = geo["cell_types"]        # list, matches last dim of bias_traj
    K             = len(cyt_names)
    n_epochs      = len(epochs)
    late_start    = max(0, int(n_epochs * (1 - LATE_FRACTION)))

    cyt_to_idx = {c: i for i, c in enumerate(cyt_names)}
    ct_to_idx  = {c: i for i, c in enumerate(cell_types)}

    results = {}

    for a_idx, a_name in enumerate(cyt_names):
        if a_name not in attn.get("trajectory", {}):
            continue
        attn_a = attn["trajectory"][a_name]  # {cell_type: np.array(n_epochs)}

        for b_idx, b_name in enumerate(cyt_names):
            if a_idx == b_idx:
                continue

            # Final asymmetry for this pair
            final_asym = float(np.mean(asym_traj[late_start:, a_idx, b_idx]))
            if final_asym < ASYM_THRESHOLD:
                continue

            # Per-cell-type values averaged over late epochs
            bias_late = np.mean(bias_traj[late_start:, a_idx, b_idx, :], axis=0)  # (n_ct,)

            # Attention over late epochs for cytokine A
            attn_late = {}
            for ct, traj_arr in attn_a.items():
                if ct not in ct_to_idx:
                    continue
                if len(traj_arr) != n_epochs:
                    continue
                attn_late[ct] = float(np.mean(traj_arr[late_start:]))

            if not attn_late:
                continue

            # Joint score: attention_A(T) × bias(A,B,T) for each T
            joint_scores = {}
            for ct, attn_val in attn_late.items():
                ct_i = ct_to_idx[ct]
                joint_scores[ct] = attn_val * float(bias_late[ct_i])

            # X = argmax joint score (must be positive)
            best_joint = max(joint_scores.values()) if joint_scores else 0.0
            X = max(joint_scores, key=joint_scores.get) if joint_scores else None
            if X is None or best_joint <= 0:
                continue

            # Y = argmax raw bias(A,B,T)
            best_bias_idx = int(np.argmax(bias_late))
            best_bias_val = float(bias_late[best_bias_idx])
            Y = cell_types[best_bias_idx] if best_bias_val > 0 else None

            if Y is None:
                continue

            # Temporal: peak attention epoch for X and Y within A-tubes
            x_traj = attn_a.get(X)
            y_traj = attn_a.get(Y)
            x_peak = _peak_epoch(x_traj, epochs) if x_traj is not None else None
            y_peak = _peak_epoch(y_traj, epochs) if y_traj is not None else None

            results[(a_name, b_name)] = {
                "X":            X,
                "Y":            Y,
                "joint_X":      best_joint,
                "bias_Y":       best_bias_val,
                "asym":         final_asym,
                "X_peak_epoch": x_peak,
                "Y_peak_epoch": y_peak,
                "is_quartet":   (X != Y),
            }

    return results


def aggregate_across_seeds(per_seed_results: list, seeds: list) -> dict:
    """
    Aggregate per-seed quartet results.

    For each (A,B) pair present in >= MIN_SEEDS seeds:
      - X_consensus: most common X across seeds
      - Y_consensus: most common Y across seeds
      - seeds_agree_X: count of seeds agreeing on X_consensus
      - seeds_agree_Y: count of seeds agreeing on Y_consensus
      - mean_asym, std_asym
    """
    all_pairs = set()
    for seed_res in per_seed_results:
        all_pairs.update(seed_res.keys())

    aggregated = {}
    for pair in all_pairs:
        seed_data = [r[pair] for r in per_seed_results if pair in r]
        if len(seed_data) < MIN_SEEDS:
            continue

        X_votes = [d["X"] for d in seed_data]
        Y_votes = [d["Y"] for d in seed_data]
        X_consensus = max(set(X_votes), key=X_votes.count)
        Y_consensus = max(set(Y_votes), key=Y_votes.count)
        seeds_agree_X = X_votes.count(X_consensus)
        seeds_agree_Y = Y_votes.count(Y_consensus)

        asym_vals = [d["asym"] for d in seed_data]
        x_peaks   = [d["X_peak_epoch"] for d in seed_data if d["X_peak_epoch"] is not None]
        y_peaks   = [d["Y_peak_epoch"] for d in seed_data if d["Y_peak_epoch"] is not None]

        is_quartet_votes = [d["is_quartet"] for d in seed_data]

        aggregated[pair] = {
            "A":              pair[0],
            "B":              pair[1],
            "X_consensus":    X_consensus,
            "Y_consensus":    Y_consensus,
            "seeds_agree_X":  seeds_agree_X,
            "seeds_agree_Y":  seeds_agree_Y,
            "n_seeds":        len(seed_data),
            "mean_asym":      float(np.mean(asym_vals)),
            "std_asym":       float(np.std(asym_vals)),
            "is_quartet":     (X_consensus != Y_consensus),
            "quartet_votes":  sum(is_quartet_votes),
            "X_peak_median":  int(np.median(x_peaks)) if x_peaks else None,
            "Y_peak_median":  int(np.median(y_peaks)) if y_peaks else None,
            "temporal_order_ok": (
                (np.median(x_peaks) < np.median(y_peaks))
                if x_peaks and y_peaks else None
            ),
        }

    return aggregated


def write_report(aggregated: dict, seeds: list, out_dir: Path):
    lines = []
    lines.append("=" * 72)
    lines.append("QUARTET ANALYSIS: A → X → B → Y")
    lines.append(f"Seeds: {seeds}  |  ASYM threshold: {ASYM_THRESHOLD}"
                 f"  |  Min seeds agree: {MIN_SEEDS}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Extraction rule:")
    lines.append("  X(A,B) = argmax_T [ attention_A(T,late) × bias(A,B,T,late) ]")
    lines.append("  Y(A,B) = argmax_T [ bias(A,B,T,late) ]")
    lines.append("  late = final 30% of Stage 3 epochs")
    lines.append("")

    # Pre-registered pairs first
    lines.append("─" * 72)
    lines.append("PRE-REGISTERED PREDICTIONS")
    lines.append("─" * 72)
    for a, b, exp_x, exp_y, note in PREREGISTERED:
        key = (a, b)
        res = aggregated.get(key)
        lines.append(f"\n  {a} → {b}  [{note}]")
        lines.append(f"  Expected X: {exp_x}   Expected Y: {exp_y}")
        if res is None:
            lines.append(f"  RESULT: not found (asym below threshold or missing data)")
        else:
            is_q = "QUARTET" if res["is_quartet"] else "TRIPLET (X==Y)"
            x_match = "✓" if exp_x and res["X_consensus"] == exp_x else ("?" if exp_x is None else "✗")
            y_match = "✓" if exp_y and res["Y_consensus"] == exp_y else ("?" if exp_y is None else "✗")
            temp = ("X before Y ✓" if res["temporal_order_ok"]
                    else ("X after Y ✗" if res["temporal_order_ok"] is False
                          else "temporal order unknown"))
            lines.append(f"  RESULT: {is_q}")
            lines.append(f"    X = {res['X_consensus']} ({res['seeds_agree_X']}/{res['n_seeds']} seeds) {x_match}")
            lines.append(f"    Y = {res['Y_consensus']} ({res['seeds_agree_Y']}/{res['n_seeds']} seeds) {y_match}")
            lines.append(f"    ASYM: mean={res['mean_asym']:.3f}  std={res['std_asym']:.3f}")
            lines.append(f"    Temporal: X peaks ep{res['X_peak_median']}  Y peaks ep{res['Y_peak_median']}  → {temp}")
            lines.append(f"    Quartet votes: {res['quartet_votes']}/{res['n_seeds']} seeds say X≠Y")

    # All stable quartets ranked by asymmetry
    stable_quartets = sorted(
        [(k, v) for k, v in aggregated.items()
         if v["is_quartet"] and v["seeds_agree_X"] >= MIN_SEEDS and v["seeds_agree_Y"] >= MIN_SEEDS],
        key=lambda kv: kv[1]["mean_asym"], reverse=True
    )

    lines.append("")
    lines.append("─" * 72)
    lines.append(f"ALL STABLE QUARTETS (X≠Y, both agreed ≥{MIN_SEEDS} seeds, ranked by ASYM)")
    lines.append("─" * 72)
    if not stable_quartets:
        lines.append("  None found.")
    else:
        header = f"  {'A':<18} {'X':<22} {'B':<18} {'Y':<22} {'ASYM':>6}  {'Temp':>12}  Seeds"
        lines.append(header)
        lines.append("  " + "-" * 100)
        for (a, b), res in stable_quartets[:30]:
            temp_str = (f"X<Y ep{res['X_peak_median']}<{res['Y_peak_median']}"
                        if res["temporal_order_ok"]
                        else (f"X>Y ep{res['X_peak_median']}>{res['Y_peak_median']}"
                              if res["temporal_order_ok"] is False else "?"))
            lines.append(
                f"  {a:<18} {res['X_consensus']:<22} {b:<18} {res['Y_consensus']:<22}"
                f" {res['mean_asym']:>6.2f}  {temp_str:>12}  "
                f"{res['seeds_agree_X']}/{res['n_seeds']}"
            )

    # Stable triplets (X==Y)
    stable_triplets = sorted(
        [(k, v) for k, v in aggregated.items()
         if not v["is_quartet"] and v["n_seeds"] >= MIN_SEEDS],
        key=lambda kv: kv[1]["mean_asym"], reverse=True
    )
    lines.append("")
    lines.append("─" * 72)
    lines.append(f"STABLE TRIPLETS (X==Y, ranked by ASYM)")
    lines.append("─" * 72)
    if not stable_triplets:
        lines.append("  None found.")
    for (a, b), res in stable_triplets[:15]:
        lines.append(
            f"  {a} → {res['X_consensus']} → {b}   ASYM={res['mean_asym']:.2f}"
            f"  ({res['seeds_agree_X']}/{res['n_seeds']} seeds)"
        )

    report = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(report)
    print(report)

    # Summary JSON
    summary = {
        "preregistered": {},
        "stable_quartets": [],
        "stable_triplets": [],
    }
    for a, b, exp_x, exp_y, note in PREREGISTERED:
        key = (a, b)
        res = aggregated.get(key)
        summary["preregistered"][f"{a}→{b}"] = {
            "found": res is not None,
            "result": res,
            "expected_X": exp_x,
            "expected_Y": exp_y,
            "note": note,
        }
    for (a, b), res in stable_quartets:
        summary["stable_quartets"].append({f"{a}→{b}": res})
    for (a, b), res in stable_triplets:
        summary["stable_triplets"].append({f"{a}→{b}": res})

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _log(f"\nSaved: {out_dir / 'report.txt'} and {out_dir / 'summary.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds_dir", required=True,
                   help="Directory containing stage3_seed* subdirectories.")
    p.add_argument("--seeds", nargs="+", type=int, required=True,
                   help="Seed numbers to aggregate, e.g. 11 12 13 14 15")
    p.add_argument("--seed_prefix", default="stage3_seed",
                   help="Prefix for seed subdirectory names.")
    args = p.parse_args()

    seeds_dir = Path(args.seeds_dir)
    seeds     = args.seeds

    per_seed_results = []
    for seed in seeds:
        seed_dir = seeds_dir / f"{args.seed_prefix}{seed}"
        _log(f"\nLoading seed {seed} from {seed_dir} ...")
        geo  = _load_geo(seed_dir)
        attn = _load_attn(seed_dir)
        if geo is None:
            _log(f"  WARNING: geo_trajectory.pkl not found, skipping seed {seed}")
            continue
        if attn is None:
            _log(f"  WARNING: attention_trajectory.pkl not found, skipping seed {seed}")
            continue
        _log(f"  geo epochs: {geo['epochs']}")
        _log(f"  attn cytokines: {len(attn.get('cytokines', []))}")
        res = extract_quartet_one_seed(geo, attn)
        _log(f"  Pairs above threshold: {len(res)}")
        per_seed_results.append(res)

    if not per_seed_results:
        _log("ERROR: No seeds loaded successfully.")
        sys.exit(1)

    _log(f"\nAggregating {len(per_seed_results)} seeds...")
    aggregated = aggregate_across_seeds(per_seed_results, seeds)
    _log(f"Pairs after aggregation: {len(aggregated)}")

    out_dir = seeds_dir / "quartet_analysis"
    write_report(aggregated, seeds, out_dir)


if __name__ == "__main__":
    main()
