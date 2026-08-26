"""Stage 4 of the encoder-breadth sweep — merge each arm's signatures and compare arms.

The readout is signature DIVERSITY across cytokines. Every statistic here is a descriptive
property of the emitted gene sets (set sizes, overlaps, ranks) — no engagement scores, no
PBS-normalised values, no per-cell-type matrices. Coupling and direction are deliberately
not computed: this sweep decides which encoder to fit with, not what the biology is.

Primary readouts, in the order they discriminate:
  * top-5 gene-pool size, and the largest number of cytokines sharing one top-5 gene.
    §37's collapse is worst at the TOP of the ranking (SLC8A1 in the top-5 of 56 of 90),
    which is why trimming top_n cannot fix it and why this is the sharpest measure.
  * mean pairwise Jaccard at top-50, and distinct genes used out of n x 50.

Reference points, on 24 cytokines at top-50: published anchor 81 top-5 pool / worst gene
4-of-24 / meanJ 0.065 / 504 distinct; §37 PURE 40 / 14-of-24 / 0.241 / 261. The panel here
is a different (seeded-random) 24, so the decision is read off the LADDER ACROSS ARMS, not
off absolute agreement with those numbers.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _encsweep_config as C  # noqa: E402


def merge_chunks(adir: Path) -> pd.DataFrame | None:
    parts = sorted(adir.glob("signatures_chunk_*.parquet"))
    if not parts:
        return None
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values(["cytokine", "rank_ig"]).reset_index(drop=True)
    df.to_parquet(adir / "signatures.parquet", index=False)
    return df


def diversity(df: pd.DataFrame, top_n: int) -> dict:
    """Descriptive diversity of a set of signatures. No method math."""
    d = df[df.rank_ig < top_n]
    sets = {c: set(g.gene) for c, g in d.groupby("cytokine")}
    cy = sorted(sets)
    pair_j = [
        len(sets[a] & sets[b]) / len(sets[a] | sets[b])
        for a, b in itertools.combinations(cy, 2)
    ]
    distinct = len(set().union(*sets.values())) if sets else 0
    t5 = df[df.rank_ig < 5]
    c5 = Counter(t5.gene)
    worst_gene, worst_n = c5.most_common(1)[0] if c5 else ("-", 0)
    n_genes_total = 4000  # the HVG universe the signatures are drawn from
    chance = (top_n * top_n / n_genes_total) / (2 * top_n - top_n * top_n / n_genes_total)
    return {
        "n_cytokines": len(cy),
        "top5_pool": int(t5.gene.nunique()),
        "top5_worst_gene": worst_gene,
        "top5_worst_n": int(worst_n),
        "top5_worst_frac": worst_n / len(cy) if cy else np.nan,
        "mean_jaccard": float(np.mean(pair_j)) if pair_j else np.nan,
        "median_jaccard": float(np.median(pair_j)) if pair_j else np.nan,
        "jaccard_vs_chance": float(np.mean(pair_j) / chance) if pair_j else np.nan,
        "distinct_genes": int(distinct),
        "slots": int(len(cy) * top_n),
        "collapse_x": len(cy) * top_n / distinct if distinct else np.nan,
    }


def seen_contrast(df: pd.DataFrame, seen: set, top_n: int) -> dict:
    """Within-arm contrast: panel cytokines the encoder SAW vs ones it did not.

    Breadth is not the only thing that changes across arms — as breadth falls, so does the
    chance that a given panel cytokine was itself in the encoder's training set. Those two
    make OPPOSITE predictions:

      * invariance (this sweep's hypothesis): seeing more conditions is worse, so within an
        arm, seen and unseen panel cytokines should look about the SAME;
      * familiarity (closer to the published anchor's leakage story): seeing a cytokine
        helps it specifically, so seen should look BETTER than unseen.

    This contrast is within-arm, so it is free of the between-arm confound. It is only
    informative for the mixed arms (rand18, rand45) — `pbs_only` has no seen cytokines and
    `all90` has no unseen ones.
    """
    d = df[df.rank_ig < top_n]
    sets = {c: set(g.gene) for c, g in d.groupby("cytokine")}
    out = {}
    for label, group in (("seen", [c for c in sets if c in seen]),
                         ("unseen", [c for c in sets if c not in seen])):
        pair_j = [
            len(sets[a] & sets[b]) / len(sets[a] | sets[b])
            for a, b in itertools.combinations(sorted(group), 2)
        ]
        t5 = d[(d.rank_ig < 5) & (d.cytokine.isin(group))]
        out[f"n_{label}"] = len(group)
        out[f"meanJ_{label}"] = float(np.mean(pair_j)) if pair_j else np.nan
        out[f"top5_pool_{label}"] = int(t5.gene.nunique()) if len(t5) else 0
    return out


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--top_n", type=int, default=C.TOP_N)
    args = ap.parse_args()

    out = Path(args.out_dir)
    meta = C.read_json(out / "encsweep_meta.json")

    # Arms come from the meta the prepare stage wrote, not from C.ARMS, so this same
    # analyzer serves both sweeps (§38.1 breadth, §38.4 Stage-1 construction).
    arms = list(meta["arms"].keys()) or list(C.ARMS)
    rows, sign_rows, loss_rows = [], [], []
    for arm in arms:
        adir = C.arm_dir(out, arm)
        df = merge_chunks(adir)
        if df is None:
            C.log(f"[skip] {arm}: no signature chunks")
            continue
        rec = {"arm": arm,
               "n_encoder_conditions": meta["arms"][arm]["n_encoder_conditions"],
               "n_stage1_cells": meta["arms"][arm]["n_cells"]}
        rec.update(diversity(df, args.top_n))
        rec.update(seen_contrast(df, set(meta["encoder_subsets"][arm]), args.top_n))
        rows.append(rec)

        sd = adir / "signature_sign_diagnosis.csv"
        if sd.exists():
            s = pd.read_csv(sd)
            sign_rows.append({"arm": arm,
                              "frac_up_median": float(s.frac_up.median()),
                              "frac_up_max": float(s.frac_up.max()),
                              "igrank_vs_delta_median": float(
                                  s.spearman_igrank_vs_delta.median()),
                              "mean_delta_median": float(s.mean_delta.median())})
        losses = [pd.read_csv(f).loss.iloc[-1] for f in (adir / "history").glob("*_train.csv")]
        if losses:
            loss_rows.append({"arm": arm, "n_models": len(losses),
                              "loss_final_median": float(np.median(losses)),
                              "loss_final_min": float(np.min(losses)),
                              "loss_final_max": float(np.max(losses))})

    if not rows:
        C.log("[abort] no arm produced signatures")
        return 1

    # The construction sweep gives every arm all 91 conditions, so sorting by breadth
    # would be arbitrary there; fall back to the meta's arm order (the design order).
    div = pd.DataFrame(rows)
    div = (div.sort_values("n_encoder_conditions")
           if div.n_encoder_conditions.nunique() > 1 else div)
    div.to_csv(out / "arm_diversity.csv", index=False)
    sign = pd.DataFrame(sign_rows) if sign_rows else None
    loss = pd.DataFrame(loss_rows) if loss_rows else None

    # Two sweeps share this analyzer and describe their arms differently: the breadth
    # sweep pins one cell budget for every arm, the construction sweep varies volume and
    # donor structure on purpose. Report whichever the prepare stage actually recorded.
    is_construction = meta.get("sweep") == "stage1_construction"
    if is_construction:
        budget_line = (
            "Stage-1 sets differ BY DESIGN: `pub_replica*` are one tube per condition "
            "(one donor each, `build_stage1_manifest`'s rule); `vol_*` are donor-balanced "
            f"at {meta.get('vol_small_cells')} and {meta.get('vol_large_cells')} cells.  "
        )
    else:
        budget_line = (f"Stage-1 budget: {meta['stage1_budget']} cells per arm "
                       f"(target {meta['stage1_budget_target']}).  ")

    lines = [
        "# Stage-1 construction sweep — arm comparison" if is_construction
        else "# Encoder condition-breadth sweep — arm comparison",
        "",
        f"Panel: seeded-random {len(meta['panel'])} of 90 (seed {meta['panel_seed']}).  ",
        budget_line,
        f"Pinned: embed {meta['pinned']['embed_dim']}, hidden "
        f"{tuple(meta['pinned']['hidden_dims'])}, Stage-1 "
        f"{meta['pinned']['stage1_epochs']} epochs (no early stopping), "
        f"k={len(meta['main_tube_indices'])} tubes, top_n={meta['pinned']['top_n']}.",
        "",
        ("Contrasts — each pair differs in ONE variable: "
         "`pub_replica` vs `pub_replica_clean` = D2/D3 leakage; "
         "`pub_replica_clean` vs `vol_large` = donor structure; "
         "`vol_small` vs `vol_large` = Stage-1 volume. "
         "`pub_replica` deliberately breaks §16 to size the leakage — diagnostic only.")
        if is_construction else "Encoder arms are nested: rand18 ⊂ rand45 ⊂ all90.",
        "",
        "## Signature diversity (the readout)",
        "",
        "| arm | encoder conds | top-5 pool | worst top-5 gene | meanJ | xchance | distinct/slots | collapse |",
        "|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for _, r in div.iterrows():
        lines.append(
            f"| `{r.arm}` | {r.n_encoder_conditions} | **{r.top5_pool}** | "
            f"{r.top5_worst_gene} {r.top5_worst_n}/{r.n_cytokines} | "
            f"**{r.mean_jaccard:.3f}** | {r.jaccard_vs_chance:.0f}x | "
            f"{r.distinct_genes}/{r.slots} | {r.collapse_x:.1f}x |"
        )
    lines += [
        "",
        "Reference (different 24-cytokine panels, so compare the ladder, not the absolute "
        "values): published anchor top-5 pool 81, worst 4/24, meanJ 0.065, 504/1200 "
        "(2.4x); §37 PURE 40, 14/24, 0.241, 261/1200 (4.6x).",
        "",
    ]
    mixed = (div[(div.n_seen > 1) & (div.n_unseen > 1)]
             if {"n_seen", "n_unseen"}.issubset(div.columns) else div.iloc[0:0])
    if len(mixed):
        lines += [
            "## Within-arm contrast: panel cytokines the encoder saw vs did not",
            "",
            "Only the mixed arms are informative here (`pbs_only` saw none of the panel,",
            "`all90` saw all of it). Similar seen/unseen columns support the invariance",
            "reading; markedly better `seen` would instead point at familiarity with the",
            "specific cytokine — closer to the published anchor's Stage-1 leakage.",
            "",
            "| arm | n seen | n unseen | meanJ seen | meanJ unseen | top-5 pool seen | unseen |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in mixed.iterrows():
            lines.append(
                f"| `{r.arm}` | {int(r.n_seen)} | {int(r.n_unseen)} | "
                f"{r.meanJ_seen:.3f} | {r.meanJ_unseen:.3f} | "
                f"{int(r.top5_pool_seen)} | {int(r.top5_pool_unseen)} |"
            )
        lines.append("")

    if sign is not None:
        lines += ["## Signature sign validity", "",
                  "| arm | median frac_up | max frac_up | median rho(IG rank, Δ) | median mean_Δ |",
                  "|---|---:|---:|---:|---:|"]
        for _, r in sign.iterrows():
            lines.append(f"| `{r.arm}` | {r.frac_up_median:.3f} | {r.frac_up_max:.3f} | "
                         f"{r.igrank_vs_delta_median:+.3f} | {r.mean_delta_median:+.4f} |")
        lines += ["", "§37 reference: median frac_up 0.655, max 0.800, "
                      "median rho(IG rank, Δ) −0.162.", ""]
    if loss is not None:
        lines += ["## Stage-2 training", "",
                  "| arm | models | median final loss | min | max |",
                  "|---|---:|---:|---:|---:|"]
        for _, r in loss.iterrows():
            lines.append(f"| `{r.arm}` | {int(r.n_models)} | {r.loss_final_median:.5f} | "
                         f"{r.loss_final_min:.5f} | {r.loss_final_max:.5f} |")
        lines.append("")

    lines += [
        "## How to read this",
        "",
        "If top-5 pool rises and meanJ falls as encoder breadth drops, condition breadth is",
        "confirmed causal: Stage-1's cell-type objective was training the encoder to be",
        "invariant to the very perturbation signal the method needs, and showing it fewer",
        "cytokines leaves more of that signal in the representation.",
        "",
        "If `pbs_only` wins, prefer it for the production fit — it is canonical, uses zero",
        "cytokine cells, and avoids a random subset's arbitrariness. If it instead does",
        "*worse* than `rand18`, that is the distribution-shift failure mode: an encoder that",
        "has never seen a stimulated cell extrapolating badly onto one.",
        "",
        "If all four arms look like §37, breadth is not the lever, and the remaining",
        "suspects are encoder width, tube count k, and the published anchor's Stage-1",
        "leakage.",
    ]
    (out / "ARM_COMPARISON.md").write_text("\n".join(lines) + "\n")

    C.log("\n" + "\n".join(lines[8:8 + 6 + len(div)]))
    C.log(f"\n[done] {out/'ARM_COMPARISON.md'}")
    C.mark_done(out, "analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
