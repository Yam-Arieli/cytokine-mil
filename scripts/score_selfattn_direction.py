"""
Score §34 self-attention direction calls on the SAME audited benchmark as the
IG cross_asym 88% (CLAUDE.md §34.4).

For each audited labeled pair (axis_a, axis_b) with counts_in_benchmark=True in
reports/cascade_pairs/cytokine_axes_audited.csv, we derive a SIGNED self-attention
direction statistic and compare its sign to `expected_sign` (+1 ⇒ a_to_b,
axis_a upstream; −1 ⇒ b_to_a) — identical convention and denominator to
scripts/retally_pipeline_against_audit.py (the 15/17 = 88% headline).

Two self-attention direction statistics (both: + ⇒ a_to_b):
  - relay-lag       : sign of relay_recruitment_lag mean_lag (§33 pooling timing).
  - interaction-asym: sign of relay_interaction_direction D (§34 cell×cell).

Output: a comparison table with the IG cross_asym reference row → SELFATTN_RESULTS.md.
Averaged over seeds (per-seed accuracy then mean; plus majority-vote-over-seeds).

Usage:
    python scripts/score_selfattn_direction.py --base_dir results/selfattn --seeds 42 123 7
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.attention_dynamics import attention_primary, relay_recruitment_lag
from cytokine_mil.analysis.attention_interaction import relay_interaction_direction

AUDIT_CSV = REPO_ROOT / "reports/cascade_pairs/cytokine_axes_audited.csv"
IG_ACCURACY_MD = REPO_ROOT / "reports/cascade_pairs/pipeline_accuracy_audited.md"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True, help="dir containing seed_<s>/ subdirs")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--audit_csv", default=str(AUDIT_CSV))
    p.add_argument("--out", default=None)
    return p.parse_args()


def _safe_int(x):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return 0


def _load_benchmark(audit_csv: str) -> pd.DataFrame:
    df = pd.read_csv(audit_csv)
    df = df[df["counts_in_benchmark"].astype(str).str.lower() == "true"].copy()
    df["expected_sign_int"] = df["expected_sign"].apply(_safe_int)
    return df[["axis_a", "axis_b", "expected_sign_int"]].reset_index(drop=True)


def _seed_signs(run_dir: Path, bench: pd.DataFrame):
    """Return per-pair (relay_sign, inter_sign) for one seed; NaN sign = no call."""
    with open(run_dir / "attention_trajectory.pkl", "rb") as f:
        pool = pickle.load(f)
    with open(run_dir / "interaction_trajectory.pkl", "rb") as f:
        inter = pickle.load(f)
    traj, per_donor, epochs = pool["trajectory"], pool["trajectory_per_donor"], pool["epochs"]
    rows = []
    for _, r in bench.iterrows():
        A, B = r["axis_a"], r["axis_b"]
        relay_sign, inter_sign = np.nan, np.nan
        if A in traj and B in traj:
            lag = relay_recruitment_lag(traj, per_donor, epochs, A, B)
            if np.isfinite(lag["mean_lag"]) and lag["mean_lag"] != 0:
                relay_sign = float(np.sign(lag["mean_lag"]))
            T_A = attention_primary(traj.get(A, {}))
            T_B = attention_primary(traj.get(B, {}))
            if T_A and T_B:
                d = relay_interaction_direction(inter["interaction"], A, B, T_A, T_B)
                if np.isfinite(d["D"]) and d["D"] != 0:
                    inter_sign = float(np.sign(d["D"]))
        rows.append({"axis_a": A, "axis_b": B, "expected": r["expected_sign_int"],
                     "relay_sign": relay_sign, "inter_sign": inter_sign})
    return pd.DataFrame(rows)


def _accuracy(signs: pd.Series, expected: pd.Series):
    """(n_called, n_correct_of_called, n_correct_of_all, n_total)."""
    called = signs.notna() & (signs != 0)
    n_total = len(signs)
    n_called = int(called.sum())
    correct_called = int(((signs == expected) & called).sum())
    correct_all = int((signs == expected).sum())   # NaN != expected -> counted wrong
    return n_called, correct_called, correct_all, n_total


def _ig_reference():
    """Read the standing IG cross_asym headline accuracy if present; else cite."""
    if IG_ACCURACY_MD.exists():
        txt = IG_ACCURACY_MD.read_text()
        m = re.search(r"([0-9]+)\s*/\s*([0-9]+).{0,40}?(?:accuracy|correct)", txt, re.I)
        if m:
            n, d = int(m.group(1)), int(m.group(2))
            return f"{n}/{d} = {100*n/d:.0f}%", "reproduced from pipeline_accuracy_audited.md"
    return "15/17 = 88%", "standing committed §26 number (cited; not reproduced this run)"


def main():
    args = _parse_args()
    base = Path(args.base_dir)
    bench = _load_benchmark(args.audit_csv)

    per_seed = {}
    for s in args.seeds:
        rd = base / f"seed_{s}"
        if not (rd / "attention_trajectory.pkl").exists():
            print(f"skip seed {s}: no attention_trajectory.pkl")
            continue
        per_seed[s] = _seed_signs(rd, bench)

    if not per_seed:
        print("ERROR: no seed results found."); sys.exit(1)

    # per-seed accuracy for each statistic
    def stat_rows(col):
        out = []
        for s, df in per_seed.items():
            nc, cc, ca, nt = _accuracy(df[col], df["expected"])
            out.append({"seed": s, "n_called": nc, "correct_called": cc,
                        "correct_all": ca, "n_total": nt})
        return pd.DataFrame(out)

    relay_acc = stat_rows("relay_sign")
    inter_acc = stat_rows("inter_sign")

    # majority-vote-over-seeds sign per pair, then accuracy
    def majority_acc(col):
        mat = np.vstack([per_seed[s][col].to_numpy() for s in per_seed])  # (n_seed, n_pair)
        maj = []
        for j in range(mat.shape[1]):
            vals = mat[:, j]
            vals = vals[np.isfinite(vals) & (vals != 0)]
            maj.append(np.sign(vals.sum()) if vals.size else np.nan)
        maj = pd.Series(maj)
        exp = next(iter(per_seed.values()))["expected"]
        return _accuracy(maj, exp)

    ig_str, ig_note = _ig_reference()

    L = ["# §34 self-attention — cascade-direction accuracy vs the IG benchmark\n\n",
         f"Benchmark: {len(bench)} audited labeled pairs (`counts_in_benchmark=True`) from "
         "`cytokine_axes_audited.csv`; sign vs `expected_sign` (+1 ⇒ a_to_b). Same denominator "
         "as the IG cross_asym 88%.\n\n",
         "## Headline comparison (accuracy over benchmark pairs)\n\n",
         "| method | seeds | accuracy (correct / benchmark) | accuracy (of called) |\n",
         "|---|---|---|---|\n",
         f"| IG `cross_asym` (§26 reference) | — | **{ig_str}** | — |  <!-- {ig_note} -->\n"]

    for name, acc_df in [("self-attn relay-lag", relay_acc), ("self-attn interaction-asym", inter_acc)]:
        mean_all = acc_df["correct_all"].sum() / acc_df["n_total"].sum()
        mean_called = (acc_df["correct_called"].sum() / max(acc_df["n_called"].sum(), 1))
        seeds_str = ",".join(str(s) for s in per_seed)
        L.append(f"| {name} (per-seed mean) | {seeds_str} | {mean_all:.2f} "
                 f"({acc_df['correct_all'].sum()}/{acc_df['n_total'].sum()}) | "
                 f"{mean_called:.2f} ({acc_df['correct_called'].sum()}/{acc_df['n_called'].sum()}) |\n")

    for name, col in [("self-attn relay-lag", "relay_sign"), ("self-attn interaction-asym", "inter_sign")]:
        nc, cc, ca, nt = majority_acc(col)
        L.append(f"| {name} (majority-vote seeds) | {','.join(str(s) for s in per_seed)} | "
                 f"{ca/nt:.2f} ({ca}/{nt}) | {cc}/{nc} |\n")

    def _md_table(acc_df):
        s = ["| seed | n_called | correct_called | correct_all | n_total |\n",
             "|---|---|---|---|---|\n"]
        for _, r in acc_df.iterrows():
            s.append(f"| {r['seed']} | {r['n_called']} | {r['correct_called']} | "
                     f"{r['correct_all']} | {r['n_total']} |\n")
        return "".join(s)

    L.append("\n## Per-seed detail\n\n### relay-lag\n")
    L.append(_md_table(relay_acc) + "\n### interaction-asymmetry\n")
    L.append(_md_table(inter_acc))
    L.append("\n> Direction ≠ existence (Path A) ≠ causation. This is ADDITIVE to IG "
             "`cross_asym` (§26), which remains the primary coupling+direction method. "
             "Small n (benchmark pairs); attention is task-driven, seed-noisy.\n")

    out = Path(args.out) if args.out else (REPO_ROOT / "reports/selfattn_dynamics/SELFATTN_RESULTS.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(L))
    print(f"Saved: {out}")
    print(f"IG reference: {ig_str}")
    for name, acc_df in [("relay-lag", relay_acc), ("interaction-asym", inter_acc)]:
        print(f"  {name}: correct_all {acc_df['correct_all'].sum()}/{acc_df['n_total'].sum()}")


if __name__ == "__main__":
    main()
