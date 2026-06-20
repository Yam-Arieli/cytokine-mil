"""
Aggregate the Oesinghaus donor-count coupling test (reflective-crunching-yeti plan):
is the vsPBS-vs-vsPanel COUPLING-gate preference driven by donor COUNT or the dataset?

Reads the 4 cell-level-degree runs (high_all10 + low_g1/g2/g3), each with both
signature variants, and compares IG_vsPBS vs IG_vsPanel on the HUB (degree-corrected)
mode. Primary readout = the coupling_hub RANKING of the 17 benchmark-positive pairs
(the binary null is permissive at the cell level — §28.2 magnitude-floor); secondary =
hub recall (null p<0.05) and over-call (coupled_frac). Applies the pre-registered
decision rule and writes reports/cascade_pairs/COUPLING_DONOR_COUNT_OES.md.

numpy/pandas only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

RUNS = ["high_all10", "low_g1", "low_g2", "low_g3"]
LOW_RUNS = ["low_g1", "low_g2", "low_g3"]
VARIANTS = ["IG_vsPBS", "IG_vsPanel"]


def _md(df: pd.DataFrame, cols) -> str:
    cols = list(cols)
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append("NaN" if np.isnan(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _metrics(df: pd.DataFrame) -> Dict[str, float]:
    """HUB-mode metrics for one (run, variant) per-pair table."""
    d = df.copy()
    d["is_pos"] = _as_bool(d["is_pos"])
    d["coupled_hub"] = _as_bool(d["coupled_hub"])
    d = d.sort_values("coupling_hub", ascending=False).reset_index(drop=True)
    d["rank_hub"] = np.arange(1, len(d) + 1)
    posd = d[d["is_pos"]]
    n_pos = len(posd)
    return {
        "n_pairs": int(len(d)),
        "n_pos": n_pos,
        "mean_rank_pos": float(posd["rank_hub"].mean()) if n_pos else float("nan"),
        "median_rank_pos": float(posd["rank_hub"].median()) if n_pos else float("nan"),
        "pos_in_top20": int((posd["rank_hub"] <= 20).sum()),
        "recall_hub": int(posd["coupled_hub"].sum()),
        "overcall_hub": float(d["coupled_hub"].mean()),
    }


def _winner(pbs: Optional[Dict], panel: Optional[Dict]) -> str:
    """Which variant ranks the benchmark pairs higher (lower mean_rank_pos)? Primary
    readout. Returns 'vsPBS' | 'vsPanel' | 'tie' | 'n/a'."""
    if not pbs or not panel:
        return "n/a"
    a, b = pbs["mean_rank_pos"], panel["mean_rank_pos"]
    if np.isnan(a) or np.isnan(b):
        return "n/a"
    if abs(a - b) < 0.5:
        return "tie"
    return "vsPBS" if a < b else "vsPanel"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="results/coupling_donor_count_oes")
    ap.add_argument("--out", default=str(
        REPO_ROOT / "reports" / "cascade_pairs" / "COUPLING_DONOR_COUNT_OES.md"))
    args = ap.parse_args()
    base = Path(args.base_dir)

    per: Dict[str, Dict[str, Optional[Dict]]] = {}
    rows: List[Dict] = []
    for run in RUNS:
        per[run] = {}
        for v in VARIANTS:
            f = base / run / f"cell_degree_{v}.csv"
            if not f.exists():
                per[run][v] = None
                print(f"WARN missing {f}")
                continue
            m = _metrics(pd.read_csv(f))
            per[run][v] = m
            rows.append({"run": run, "variant": v, **m})

    if not rows:
        print("FATAL: no run outputs found under", base)
        return
    tab = pd.DataFrame(rows)[
        ["run", "variant", "n_pairs", "n_pos", "mean_rank_pos",
         "median_rank_pos", "pos_in_top20", "recall_hub", "overcall_hub"]]

    win = {run: _winner(per[run]["IG_vsPBS"], per[run]["IG_vsPanel"]) for run in RUNS}
    high_win = win["high_all10"]
    low_wins = [win[r] for r in LOW_RUNS]
    n_low_pbs = sum(w == "vsPBS" for w in low_wins)
    n_low_panel = sum(w == "vsPanel" for w in low_wins)

    # ---- pre-registered decision rule ----
    if high_win == "vsPBS":
        verdict = ("**It's the cell-level PATH, not donor count per se.** HIGH (full "
                   "10 donors, cell-level) already favors IG_vsPBS — so switching to the "
                   "cell-level path is what flips the variant preference, not the number "
                   "of donors. The rule 'few donors → vsPBS' holds *operationally* (few "
                   "donors force the cell-level path) but the mechanism is the path.")
        tag = "PATH"
    elif n_low_pbs >= 2 and high_win in ("vsPanel", "tie"):
        verdict = ("**Donor COUNT drives the preference — rule validated.** HIGH favors "
                   f"IG_vsPanel/tie while {n_low_pbs}/3 LOW (3-donor) groups favor "
                   "IG_vsPBS. Same dataset, same path, same signatures — only donor count "
                   "changed. Confirms 'few donors → vsPBS; many donors → vsPanel'.")
        tag = "DONOR_COUNT"
    elif n_low_panel >= 2:
        verdict = ("**NOT donor count — the ID divergence was dataset-specific.** The "
                   f"3-donor Oes groups still favor IG_vsPanel ({n_low_panel}/3). Retract "
                   "'few donors → vsPBS'; the ID result (vsPBS best) reflects ID's biology/"
                   "panel, not donor count.")
        tag = "DATASET"
    else:
        verdict = ("**Inconclusive.** LOW groups split between variants; no clean "
                   f"majority (vsPBS {n_low_pbs}/3, vsPanel {n_low_panel}/3). Larger "
                   "donor sweep or more groups needed.")
        tag = "INCONCLUSIVE"

    L: List[str] = []
    L.append("# Oesinghaus donor-count test — is the vsPBS/vsPanel COUPLING "
             "preference about donor COUNT or the dataset?")
    L.append("")
    L.append(f"**Verdict ({tag}):** {verdict}")
    L.append("")
    L.append("Held constant: dataset (Oes), method (cell-level degree coupling), "
             "signatures (binary_ig_all24, all donors), benchmark "
             "(cytokine_axes_audited.csv, 17 counts_in_benchmark positives). Varied: "
             "donors entering the coupling. Primary readout = coupling_hub RANKING of "
             "the benchmark pairs (binary null is permissive — trust the rank).")
    L.append("")
    L.append("## Per-run variant winner (by benchmark-pair mean coupling_hub rank; "
             "lower rank = ranked higher = better)")
    L.append("")
    wtab = pd.DataFrame([{
        "run": run,
        "vsPBS_mean_rank": (per[run]["IG_vsPBS"] or {}).get("mean_rank_pos", float("nan")),
        "vsPanel_mean_rank": (per[run]["IG_vsPanel"] or {}).get("mean_rank_pos", float("nan")),
        "vsPBS_recall": (per[run]["IG_vsPBS"] or {}).get("recall_hub", float("nan")),
        "vsPanel_recall": (per[run]["IG_vsPanel"] or {}).get("recall_hub", float("nan")),
        "vsPBS_overcall": (per[run]["IG_vsPBS"] or {}).get("overcall_hub", float("nan")),
        "vsPanel_overcall": (per[run]["IG_vsPanel"] or {}).get("overcall_hub", float("nan")),
        "winner": win[run],
    } for run in RUNS])
    L.append(_md(wtab, ["run", "vsPBS_mean_rank", "vsPanel_mean_rank", "vsPBS_recall",
                        "vsPanel_recall", "vsPBS_overcall", "vsPanel_overcall", "winner"]))
    L.append("")
    L.append(f"LOW groups favoring vsPBS: **{n_low_pbs}/3**; favoring vsPanel: "
             f"**{n_low_panel}/3**. HIGH (anchor): **{high_win}**.")
    L.append("")
    L.append("## Full metrics (HUB mode)")
    L.append("")
    L.append(_md(tab, tab.columns))
    L.append("")
    L.append("## Caveats")
    L.append("- Reused all-donor signatures → isolates the donor-count effect on the "
             "coupling PATH; does NOT test signature degradation at low donor count.")
    L.append("- Cell-level null is permissive → ranking is the readout, not recall_hub.")
    L.append("- 3 LOW groups = a consistency check (3 disjoint donor triples), not a CI.")
    L.append("- recall_hub denominator = benchmark-positive pairs scored "
             f"(n_pos≈{int(tab['n_pos'].max())}); HIGH should track the §28.2 donor-level "
             "Oes numbers (11/17) — large divergence is itself a 'path' signal.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"VERDICT [{tag}]: high={high_win}, low_pbs={n_low_pbs}/3, "
          f"low_panel={n_low_panel}/3")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
