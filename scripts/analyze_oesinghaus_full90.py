#!/usr/bin/env python
"""Stage 6 of the Oesinghaus full-90 DAG — join, score, and write RESULTS.md.

Pure CSV work: it joins the coupling and direction tables cascadir produced, attaches
labels, and counts. **No engagement scores, PBS-normalised values or per-cell-type
matrices are summed, subtracted or averaged anywhere here** — every coupling and
`cross_asym` value is read as cascadir emitted it (cascadir-values SKILL.md).

What it produces:
  per_pair_summary.csv    one row per unordered pair, coupling + direction + labels
  RESULTS.md              regression check, enrichment, over-call, novel candidates

The scientific point of the run is the DENOMINATOR. Every published Oesinghaus coupling
number is measured on a 24-cytokine panel whose members were chosen because they appear in
a literature benchmark pair (276 pairs, 76 coupled). Over-call here is recomputed over the
full 4005-pair family, which is neutral.

Enrichment is subtler and is reported three ways, because the literature LABELS are not
neutral even when the pair family is: only the 121 axes in `cytokine_axes.csv` carry an
adjudicated `literature_status`, and those 121 were themselves selected by Path A on the
old panel. Unlabeled is not the same as unsupported.

Usage: python scripts/analyze_oesinghaus_full90.py --output_dir results/oes_full90
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _full90_config as C  # noqa: E402

AXES_CSV = REPO_ROOT / "reports" / "cascade_pairs" / "cytokine_axes.csv"
LIT_SUPPORTED = {"KNOWN_DIRECTIONAL", "KNOWN_COREGULATED", "PARTIAL", "PRE_REGISTERED"}
LIT_UNSUPPORTED = {"NOVEL"}
LIT_EXCLUDED = {"NAME_AMBIGUOUS"}


def key(a, b):
    return tuple(sorted((str(a), str(b))))


def load_labels():
    """Literature status per pair (121 adjudicated axes) + the published 276-pair family."""
    lit = {}
    with open(AXES_CSV) as fh:
        for row in csv.DictReader(fh):
            lit[key(row["axis_a"], row["axis_b"])] = row.get("literature_status") or ""

    published, published_coupled = set(), set()
    if C.PUBLISHED_COUPLING_CSV.exists():
        with open(C.PUBLISHED_COUPLING_CSV) as fh:
            for row in csv.DictReader(fh):
                k = key(row["axis_a"], row["axis_b"])
                published.add(k)
                if row.get("coupled_q05") == "True":
                    published_coupled.add(k)

    audited = {}
    with open(C.AUDITED_CSV) as fh:
        for row in csv.DictReader(fh):
            if str(row.get("counts_in_benchmark")).lower() == "true":
                audited[key(row["axis_a"], row["axis_b"])] = int(float(row["expected_sign"]))
    return lit, published, published_coupled, audited


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]; returns (odds_ratio, p)."""
    try:
        from scipy.stats import fisher_exact

        return fisher_exact([[a, b], [c, d]])
    except Exception:  # pragma: no cover - scipy is present in biovenv
        return (float("nan"), float("nan"))


def pct(x, n):
    return f"{x}/{n} = {x/n:.1%}" if n else f"{x}/0 = n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="results/oes_full90")
    ap.add_argument("--report_path",
                    default=str(REPO_ROOT / "reports" / "oesinghaus_full90" / "RESULTS.md"))
    ap.add_argument("--top_k", type=int, default=25)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    out = Path(args.output_dir)
    coupling = pd.read_csv(out / "coupling_donor_degree.csv")
    direction = pd.read_csv(out / "direction_table.csv")
    lit, published, published_coupled, audited = load_labels()

    for df in (coupling, direction):
        df["pair"] = [key(a, b) for a, b in zip(df["condition_a"], df["condition_b"])]

    dir_cols = ["pair", "cross_asym_median", "directional_score_median", "sign_consensus",
                "n_pos", "n_neg", "classification", "direction", "upstream", "null_p"]
    df = coupling.merge(direction[dir_cols], on="pair", how="outer", validate="one_to_one")
    if df["coupling"].isna().any() or df["cross_asym_median"].isna().any():
        n_c = int(df["coupling"].isna().sum())
        n_d = int(df["cross_asym_median"].isna().sum())
        print(f"[warn] {n_c} pairs without coupling, {n_d} without direction "
              "(kept as NaN in per_pair_summary.csv)", flush=True)

    df["literature_status"] = [lit.get(k, "") for k in df["pair"]]
    df["lit_labeled"] = df["literature_status"].isin(
        LIT_SUPPORTED | LIT_UNSUPPORTED | LIT_EXCLUDED)
    df["lit_supported"] = df["literature_status"].isin(LIT_SUPPORTED)
    df["in_published_276"] = [k in published for k in df["pair"]]
    df["published_coupled_q05"] = [k in published_coupled for k in df["pair"]]
    df["in_audited_17"] = [k in audited for k in df["pair"]]
    df["expected_sign"] = [audited.get(k, np.nan) for k in df["pair"]]
    df["sign_correct"] = np.where(
        df["in_audited_17"],
        np.sign(df["cross_asym_median"]) == df["expected_sign"],
        np.nan,
    )
    df["condition_a"] = [k[0] for k in df["pair"]]
    df["condition_b"] = [k[1] for k in df["pair"]]
    df = df.drop(columns=["pair"])

    summary_path = out / "per_pair_summary.csv"
    df.sort_values("coupling", ascending=False).to_csv(summary_path, index=False)
    print(f"[write] {summary_path} ({len(df)} pairs)", flush=True)

    n_all = len(df)
    gates = {}
    for q in C.FDR_QS:
        tag = f"coupled_q{int(round(q*100)):02d}"
        if tag in df.columns:
            gates[tag] = df[tag].fillna(False).astype(bool)
    if "coupled" in df.columns:
        gates["coupled_alpha05"] = df["coupled"].fillna(False).astype(bool)

    primary = gates.get("coupled_q05", next(iter(gates.values())))

    # ---- enrichment, three ways -----------------------------------------
    labeled = df["lit_labeled"] & ~df["literature_status"].isin(LIT_EXCLUDED)
    sup = df["lit_supported"]
    enr = {}
    for tag, g in gates.items():
        a = int((g & sup).sum())            # coupled & supported
        b = int((g & ~sup).sum())           # coupled & (unsupported or unlabeled)
        c = int((~g & sup).sum())
        d = int((~g & ~sup).sum())
        orat, p = fisher(a, b, c, d)
        la = int((g & labeled & sup).sum())
        lb = int((g & labeled & ~sup).sum())
        lc = int((~g & labeled & sup).sum())
        ld = int((~g & labeled & ~sup).sum())
        lorat, lp = fisher(la, lb, lc, ld)
        enr[tag] = {
            "n_coupled": int(g.sum()),
            "frac_coupled_of_4005": float(g.mean()),
            "full_family": {"a": a, "b": b, "c": c, "d": d,
                            "supported_frac_coupled": a / (a + b) if a + b else float("nan"),
                            "supported_frac_all": float(sup.mean()),
                            "odds_ratio": float(orat), "p": float(p)},
            "within_labeled": {"a": la, "b": lb, "c": lc, "d": ld,
                               "supported_frac_coupled": la / (la + lb) if la + lb else float("nan"),
                               "supported_frac_labeled": float(sup[labeled].mean()) if labeled.any() else float("nan"),
                               "odds_ratio": float(lorat), "p": float(lp)},
        }

    # ---- novel candidates ------------------------------------------------
    novel = df[
        primary
        & ~df["in_published_276"]
        & ~df["lit_labeled"]
    ].sort_values("coupling", ascending=False)
    novel_path = out / "novel_candidates.csv"
    novel.to_csv(novel_path, index=False)

    bench = {}
    bp = out / "direction_meta.json"
    if bp.exists():
        bench = json.loads(bp.read_text()).get("benchmark", {})

    C.write_json(out / "analysis_meta.json",
                 {"n_pairs": n_all, "enrichment": enr, "n_novel_candidates": int(len(novel))})

    # ---- report ----------------------------------------------------------
    prep = json.loads((out / "prepare_meta.json").read_text())
    sig_meta = json.loads((out / "signatures_meta.json").read_text())
    cm = json.loads((out / "coupling_meta.json").read_text())
    n_cond = sig_meta["n_conditions"]

    L = []
    A = L.append
    A("# Oesinghaus full-90 — coupling and cascade direction on a neutral background")
    A("")
    A("_Auto-generated by `scripts/analyze_oesinghaus_full90.py`. All coupling and "
      "`cross_asym` values come from `cascadir` (`CascadeDirection.signature_coupling` / "
      "`.direction_table` / `.benchmark`); nothing here re-derives the method's math._")
    A("")
    A("## What this run is, and what it is not")
    A("")
    A(f"A single fresh `cascadir` fit over **{n_cond} cytokines + PBS**, giving coupling "
      f"and direction for all **{n_all} unordered pairs**.")
    A("")
    A("Every published Oesinghaus coupling number is measured on a 24-cytokine panel whose "
      "members were selected because they appear in a literature benchmark pair (276 pairs, "
      "76 coupled). That background is non-neutral. **The over-call figures below are the "
      "first measured against a neutral pair family.**")
    A("")
    A("**This is a separate fit.** 66 of the 90 cytokines had no published signature, so "
      "these numbers must never be averaged or mixed with the published 24/45-cytokine "
      "results (CLAUDE.md §26.3, the Sheu mixed-provenance lesson).")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| donors | {', '.join(prep['stage1_cells']['donors'])} (D2/D3 held out everywhere, "
      "Stage-1 included — CLAUDE.md §16) |")
    A(f"| tubes | {prep['tubes']['n_tubes']} ({prep['tubes']['cells_per_tube_mean']:.0f} "
      f"cells/tube mean), {len(prep['tubes']['cell_types'])} cell types |")
    A(f"| signatures | {n_cond} x top-{C.TOP_N} by Integrated Gradients, "
      f"{sig_meta['n_unique_genes']} distinct genes |")
    A(f"| encoder | ONE shared Stage-1 encoder, sha256 `{sig_meta['encoder_sha256'][:16]}...`, "
      "verified identical in every training chunk (the CLAUDE.md §27.6 guard) |")
    A(f"| hyperparameters | published \"wide\": embed={C.EMBED_DIM}, "
      f"hidden={C.HIDDEN_DIMS}, attn={C.ATTENTION_HIDDEN_DIM}, Stage-1 "
      f"{C.STAGE1_EPOCHS}@{C.STAGE1_LR}, Stage-2 {C.STAGE2_EPOCHS}@{C.STAGE2_LR} |")
    A(f"| coupling gate | `signature_coupling(donor_level=True, degree_correct=True)`, "
      f"then BH-FDR over all {n_all} pairs |")
    A("")

    A("## 1. Regression check on the 17 audited pairs (reported, not a gate)")
    A("")
    if bench:
        A(f"- `cross_asym` signed accuracy (non-AMBIGUOUS): **{bench['cross_accuracy']:.3f}** "
          f"over {bench['n_scored']} pairs")
        A(f"- `cross_asym` signed accuracy (all found): {bench['cross_accuracy_all']:.3f} "
          f"over {bench['n_found']} pairs")
        A(f"- symmetric `directional_score` control: **{bench['dirscore_accuracy']:.3f}** "
          "(should sit near chance — that contrast is the evidence the antisymmetric "
          "statistic is doing the work)")
        A(f"- published anchor: {bench['published_anchor']} = 0.882 on the 24-cytokine panel")
        A(f"- classifications: {bench['classification_counts']}")
    else:
        A("_`direction_meta.json` carried no benchmark block — re-run stage 5 unsharded._")
    A("")
    A("Not a stop gate, by decision: two earlier faithful fresh fits already landed near "
      "0.65 (CLAUDE.md §27.6 at 6/11; the §31 recurrent-IG run at 11/17 **despite** a single "
      "shared encoder and the wide config), so a hard ~88% gate would abort on run-to-run "
      "variance rather than on a bug. Read the symmetric control alongside: a shortfall "
      "matters only if the control rises with it.")
    A("")

    A(f"## 2. Over-call on the neutral {n_all}-pair background")
    A("")
    A("| gate | coupled pairs | fraction of all pairs |")
    A("|---|---:|---:|")
    for tag, g in gates.items():
        A(f"| `{tag}` | {int(g.sum())} | {g.mean():.1%} |")
    A("")
    A(f"For comparison, the published 24-cytokine panel: 76/276 = 27.5% at q≤0.05 "
      "(`reports/coupling_figures_draft/donor_coupling_hub_IG_vsPBS.csv`), and the §28.2 "
      "over-call figures (77% raw → 31% degree-corrected) were measured on that same "
      "prior-selected family. **The fractions above are the neutral-background version.**")
    A("")
    A(f"Sign-test resolution: with {int(cm['n_donors_per_pair_median'])} donors the "
      f"one-sided binomial p is quantized to {cm['sign_test_p_levels'][:3]}, so the BH "
      "threshold lands on one of those steps rather than at an arbitrary cut.")
    A("")

    A("## 3. Enrichment — and why it needs two denominators")
    A("")
    A(f"The pair family is neutral, but the **labels are not**: only the "
      f"{int(labeled.sum())} pairs carrying an adjudicated `literature_status` in "
      "`cytokine_axes.csv` have any verdict at all, and those axes were themselves selected "
      "by Path A on the old panel. Unlabeled is not the same as unsupported, so both "
      "readings are given.")
    A("")
    A(f"| gate | supported/coupled (full {n_all} family) | supported fraction, all pairs | OR | p | "
      "supported/coupled (labeled only) | supported fraction, labeled | OR | p |")
    A("|---|---|---:|---:|---:|---|---:|---:|---:|")
    for tag, e in enr.items():
        f_, w = e["full_family"], e["within_labeled"]
        A(f"| `{tag}` | {pct(f_['a'], f_['a']+f_['b'])} | {f_['supported_frac_all']:.1%} | "
          f"{f_['odds_ratio']:.2f} | {f_['p']:.2g} | {pct(w['a'], w['a']+w['b'])} | "
          f"{w['supported_frac_labeled']:.1%} | {w['odds_ratio']:.2f} | {w['p']:.2g} |")
    A("")
    A(f"- **Full-{n_all} columns** count every unlabeled pair as unsupported. That is the "
      "denominator asked for, and it is a **lower bound** on the true support rate among "
      "coupled pairs — most unlabeled pairs have simply never been looked up.")
    A("- **Labeled-only columns** are the like-for-like 2x2 within the adjudicated axes; "
      "valid internally, but the adjudicated set is Path-A-selected.")
    A("- Neither is the CLAUDE.md §0 \"~50% lit-supported vs ~1% chance\" number, which was "
      "computed on the 121 Path A axes. Do not quote them interchangeably.")
    A("")

    A(f"## 4. Prior-free candidates (top {args.top_k} of {len(novel)})")
    A("")
    A(f"Pairs passing `coupled_q05` that are **not** in the published 276-pair family and "
      "carry **no** literature adjudication — the part of the run no prior could have "
      "picked out. Full list: `results/oes_full90/novel_candidates.csv`.")
    A("")
    A("| # | pair | coupling | donor consensus | q | cross_asym | direction | class |")
    A("|---:|---|---:|---:|---:|---:|---|---|")
    for i, (_, r) in enumerate(novel.head(args.top_k).iterrows(), 1):
        A(f"| {i} | {r['condition_a']} — {r['condition_b']} | {r['coupling']:.4f} | "
          f"{r.get('donor_consensus', float('nan')):.2f} | {r['q_donor']:.2g} | "
          f"{r['cross_asym_median']:.4f} | {r.get('direction','')} | "
          f"{r.get('classification','')} |")
    A("")
    A("Direction is read only on coupled pairs: `cross_asym` gives direction, never "
      "existence — a non-coupled pair can still score large (CLAUDE.md §26.4).")
    A("")

    A("## 5. Honest caveats")
    A("")
    A("- **Direction ≠ existence ≠ causation.** Coupling is the existence call; "
      "`cross_asym` only orders a pair; neither is interventional evidence.")
    A(f"- **The labels are still panel-limited.** The pair family is neutral; the literature "
      f"verdicts cover {labeled.sum()/n_all:.0%} of it ({int(labeled.sum())} pairs). "
      "Section 3 reports both readings rather than picking one.")
    A("- **Single seed.** This is one fit. The coupling-recall pipeline is known to be "
      "seed-noisy, so every count here is a point estimate.")
    A("- **`retally_pipeline_against_audit.py` was not used** — it consumes a per-cell-type "
      "CSV `direction_table` does not emit, and producing one would mean calling a "
      "module-level function instead of the orchestrator. `est.benchmark` scores the same "
      f"signs on the same {len(audited)}-pair audited denominator and adds the symmetric "
      "control.")
    A("- **Binary training ran the full 250 epochs.** Checked before the run: at epoch 50 "
      "only ~81% of each final top-50 signature has appeared and the top-50 coupled-pair "
      "set has Jaccard 0.52 against the final one, so a shorter schedule would have been a "
      "different result, not a cheaper one. The per-epoch loss plateau is **unmeasured** — "
      "no binary-MIL p_correct trajectory is persisted anywhere in the repo.")
    A("- **Stage-1 excluded D2/D3**, which some earlier Oesinghaus runs did not. Stricter, "
      "per CLAUDE.md §16, and one more reason this fit is not the published one.")
    A("")

    rp = Path(args.report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L) + "\n")
    print(f"[write] {rp}", flush=True)
    print(f"[write] {novel_path} ({len(novel)} prior-free candidates)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
