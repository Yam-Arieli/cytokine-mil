"""Stage-1 encoder training curves — accuracy and loss by epoch, across every fit.

Context (CLAUDE.md §39.5): the encoder's gene-space rank tracks Stage-1 cell VOLUME at
Spearman -0.894, and an earlier observation had low final Stage-1 loss predicting a worse
signature. Both point at the same quantity — how completely Stage 1 solved the cell-type
task — so the per-epoch curve is the thing to look at directly.

Two sources, because the two training paths record differently:

  * `cascadir` fits write `encoder_history.csv` (epoch, train_loss, train_acc, val_*).
  * `cytokine_mil` fits do NOT — `cytokine_mil/training/train_encoder.py` returns only the
    encoder. Their per-epoch numbers exist only as stdout lines in `run_log.txt`
    (`[Stage 1] Epoch N/M | loss=X | acc=Y`), so those are parsed back out. A fit whose log
    was not captured simply has no curve, and is reported as missing rather than guessed.

Descriptive only: reads training logs, plots them. No method math.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _encoder_geometry_fits as R  # noqa: E402

LOG_RE = re.compile(r"\[Stage 1\]\s+Epoch\s+(\d+)/(\d+)\s*\|\s*loss=([\d.eE+-]+)\s*\|"
                    r"\s*acc=([\d.eE+-]+)")


def curve_for(fit: dict) -> pd.DataFrame | None:
    run = (REPO_ROOT / fit["encoder"]).parent

    hist = run / "encoder_history.csv"
    if hist.exists():
        df = pd.read_csv(hist)
        df["source"] = "encoder_history.csv"
        return df

    log = run / "run_log.txt"
    if log.exists():
        rows = [{"epoch": int(m[0]), "train_loss": float(m[2]), "train_acc": float(m[3])}
                for m in LOG_RE.findall(log.read_text(errors="ignore"))]
        if rows:
            df = pd.DataFrame(rows).drop_duplicates("epoch").sort_values("epoch")
            df["source"] = "run_log.txt"
            return df
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "results" / "encoder_geometry"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vol_p = out / "stage1_volume.csv"
    vol = pd.read_csv(vol_p).set_index("fit") if vol_p.exists() else None

    frames, missing = [], []
    for fit in R.FITS:
        if not (REPO_ROOT / fit["encoder"]).exists():
            continue
        c = curve_for(fit)
        if c is None:
            missing.append(fit["key"])
            continue
        c["fit"] = fit["key"]
        c["code_path"] = fit["code_path"]
        if vol is not None and fit["key"] in vol.index:
            c["stage1_cells"] = vol.loc[fit["key"], "stage1_cells"]
        frames.append(c)

    if not frames:
        raise SystemExit("no Stage-1 curves found")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "stage1_curves.csv", index=False)

    print(f"[curves] {df.fit.nunique()} fits with a Stage-1 curve")
    if missing:
        print(f"[missing] no per-epoch record saved for: {', '.join(missing)}")
    summary = (df.sort_values("epoch").groupby(["fit", "code_path"])
                 .agg(epochs=("epoch", "max"), final_acc=("train_acc", "last"),
                      final_loss=("train_loss", "last"),
                      cells=("stage1_cells", "first"),
                      source=("source", "first")).reset_index())
    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    colour = {"cytokine_mil": "#1f77b4", "cascadir": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for key, g in df.groupby("fit"):
        g = g.sort_values("epoch")
        cp = g.code_path.iloc[0]
        cells = g.stage1_cells.iloc[0] if "stage1_cells" in g else float("nan")
        lbl = f"{key} ({int(cells/1000)}K)" if pd.notna(cells) else key
        # low-volume fits dashed: Stage-1 volume is the variable the geometry tracks
        ls = "--" if pd.notna(cells) and cells < 15000 else "-"
        axes[0].plot(g.epoch, g.train_acc, color=colour.get(cp, "#999"), ls=ls,
                     alpha=0.85, lw=1.4, label=lbl)
        axes[1].semilogy(g.epoch, g.train_loss, color=colour.get(cp, "#999"), ls=ls,
                         alpha=0.85, lw=1.4)
    axes[0].set_xlabel("Stage-1 epoch"); axes[0].set_ylabel("train accuracy (cell type)")
    axes[0].set_title("Stage-1 cell-type accuracy")
    axes[1].set_xlabel("Stage-1 epoch"); axes[1].set_ylabel("train loss")
    axes[1].set_title("Stage-1 loss (log scale)")
    axes[0].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle("Stage-1 encoder training — blue = cytokine_mil path, red = cascadir; "
                 "dashed = low Stage-1 volume (<15K cells)", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "stage1_training_curves.png", dpi=150)
    plt.close(fig)
    print(f"\n[write] {plots / 'stage1_training_curves.png'}")
    print(f"[write] {out / 'stage1_curves.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
