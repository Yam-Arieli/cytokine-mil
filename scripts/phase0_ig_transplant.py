"""Phase 0b — the IG transplant, run on the published anchor itself.

`reports/code_path_comparison/SPEC.md` §2. §38.5 left "which training code ran" as the
only variable separating a 0.065-0.077 `cytokine_mil` fit from a 0.178-0.394 `cascadir`
one. This asks the cheapest version of that question — is the difference in the ATTRIBUTION
step? — and answers it with **no training at all**, because the published run's weights
survive on the cluster.

The published head + its shared Stage-1 encoder are plain `state_dict`s whose
``encoder.* / attention.* / classifier.*`` keys match `cascadir.models.AbMil` exactly (the
two encoder modules differ only by §40's parameter-free ``pre_embed_drop``). So the very
same weights can be pushed through both attribution paths.

FIVE configurations. The first is production `cytokine_mil`; the rest are
`cascadir.derive_signature` over a 2x2 of {which tubes are attributed} x {which tubes build
the PBS baseline}:

  cm_prod              production run_binary_ig_probe.py, invoked VERBATIM (not re-implemented)
  cd_cond.cm_base.cm   cascadir IG on exactly cm_prod's tubes and baseline
  cd_cond.cm_base.main
  cd_cond.main_base.cm
  cd_cond.main_base.main   cascadir IG as §36-§40 actually run it

Reading them:

  * ``cm_prod`` vs ``cd_cond.cm_base.cm`` isolates **the algorithm** — identical weights,
    identical cells, identical baseline, so any gap is the attribution code itself.
  * ``cm_prod`` vs ``cd_cond.main_base.main`` is the **as-run** comparison, which also
    carries the configuration difference.
  * the 2x2 then says WHICH knob moves it, which a straight A/B cannot.

That configuration difference is real and was not previously noted: production
`run_binary_ig_probe.py` attributes over ``by_cyt[cyt][:max_tubes_per_cytokine]`` — the
first 10 tubes in MANIFEST order, which is grouped by donor — and builds its PBS baseline
from the first 10 PBS tubes the same way. `cascadir.derive_signature` instead attributes
over every tube of the condition in the set and averages the baseline over every control
tube in it (§36-§40: 4 tubes x 10 donors, D2/D3 excluded). The per-config donor
composition is reported so this is visible rather than inferred.

Scope: run B's 16 cytokines — the only single-encoder `cytokine_mil` reference (meanJ
0.079). Phase 1's published-24 panel is a separate, later question.
"""

from __future__ import annotations

import argparse
import itertools
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import anndata  # noqa: E402

from analyze_encsweep import diversity  # noqa: E402
from cascadir.models import AbMil, AttentionModule, BagClassifier, InstanceEncoder  # noqa: E402
from cascadir.signatures import derive_signature  # noqa: E402
from cascadir.types import PseudoTube, PseudoTubeSet  # noqa: E402
from cytokine_mil.analysis.full90_tube_io import load_tube_set  # noqa: E402
from run_binary_ig_probe import _infer_hps_from_state_dict, _load_binary_model  # noqa: E402

MANIFEST = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVGS = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
SHARDS = "results/oes_full90/tubes"
RUN_B = ("results/oesinghaus_binary_missing16/"
         "run_20260530_191127_pid213865")
CONTROL = "PBS"
MAIN_TUBE_INDICES = [0, 1, 2, 3]   # §37/§40's main split, so cd_prod is the real config


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _safe(name: str) -> str:
    return str(name).replace("/", "_")


# ---------------------------------------------------------------------------
# Tube loading
# ---------------------------------------------------------------------------


def _manifest_tube(entry: dict, gene_names: list[str]) -> PseudoTube:
    """One tube as production `run_binary_ig_probe._load_tube_X` aligns it."""
    ad = anndata.read_h5ad(entry["path"])
    ad = ad[:, gene_names]
    X = ad.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    ct = (tuple(str(c) for c in ad.obs["cell_type"])
          if "cell_type" in ad.obs.columns else tuple("NA" for _ in range(X.shape[0])))
    return PseudoTube(
        X=X, condition=str(entry["cytokine"]), donor=str(entry["donor"]),
        cell_types=ct, cell_types_included=tuple(sorted(set(ct))),
        tube_idx=int(entry["tube_idx"]),
    )


def _describe(tubes: list[PseudoTube]) -> dict:
    return {
        "n_tubes": len(tubes),
        "n_cells": int(sum(t.X.shape[0] for t in tubes)),
        "donors": sorted({t.donor for t in tubes}),
    }


# ---------------------------------------------------------------------------
# Model transplant
# ---------------------------------------------------------------------------


def build_cascadir_model(state: dict, n_genes: int, device: str) -> AbMil:
    """Rebuild the published binary AB-MIL as a `cascadir.AbMil` from its state_dict.

    Asserts the key mapping is TOTAL before loading. A silently partial `load_state_dict`
    would leave randomly-initialised weights in place and fabricate the whole result, so
    this uses strict=True and additionally compares the key sets itself for a readable
    error.
    """
    # Check the shape-defining keys FIRST. Without this a missing one surfaces as a raw
    # KeyError from deep inside HP inference, which reads like a bug in this script rather
    # than what it is: a state_dict that does not describe a binary AB-MIL.
    required = ("encoder.input_proj.0.weight", "encoder.down1.fc1.weight",
                "encoder.down2.fc1.weight", "attention.V.weight",
                "encoder.cell_type_head.weight", "classifier.classifier.weight")
    absent = [k for k in required if k not in state]
    if absent:
        raise AssertionError(
            "state_dict is missing shape-defining keys, so it cannot map 1:1 onto "
            f"cascadir.AbMil: {absent}"
        )
    hps = _infer_hps_from_state_dict(state, n_genes)
    enc = InstanceEncoder(
        input_dim=n_genes, embed_dim=hps["embed_dim"],
        n_cell_types=hps["n_cell_types"], hidden_dims=(hps["h0"], hps["h1"]),
    )
    model = AbMil(
        enc,
        AttentionModule(embed_dim=hps["embed_dim"],
                        attention_hidden_dim=hps["attention_hidden_dim"]),
        BagClassifier(embed_dim=hps["embed_dim"], n_classes=hps["n_classes"]),
        encoder_frozen=True,
    )
    want, have = set(model.state_dict()), set(state)
    if want != have:
        raise AssertionError(
            "state_dict keys do not map 1:1 onto cascadir.AbMil — refusing to load a "
            f"partial transplant.\n  missing: {sorted(want - have)[:8]}\n"
            f"  unexpected: {sorted(have - want)[:8]}"
        )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def assert_forward_equivalent(cm_model, cd_model, n_genes: int, device: str,
                              seed: int = 0, tol: float = 1e-5) -> float:
    """The transplant is only meaningful if both objects compute the same function."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(64, n_genes, generator=g).to(device)
    with torch.no_grad():
        a = cm_model(X)[0].detach().cpu().numpy()
        b = cd_model(X)[0].detach().cpu().numpy()
    d = float(np.max(np.abs(a - b)))
    if not np.isfinite(d) or d > tol:
        raise AssertionError(
            f"transplanted models disagree on the same input (max|dlogit|={d:.3e} > {tol}). "
            "Any IG comparison built on this would be meaningless."
        )
    return d


# ---------------------------------------------------------------------------
# Signature tables
# ---------------------------------------------------------------------------


def _sig_rows(cyt: str, sig, keep: int) -> list[dict]:
    return [{"cytokine": cyt, "gene": g, "ig": float(s), "rank_ig": r}
            for r, (g, s) in enumerate(zip(sig.genes, sig.ig_scores)) if r < keep]


def run_cm_production(run_dir: Path, out_dir: Path, targets: list[str], hvg: str,
                      manifest: str, top_n: int, n_steps: int, cap: int,
                      device: str) -> pd.DataFrame:
    """Invoke `run_binary_ig_probe.py` VERBATIM and read its parquet.

    Driving the real script (rather than re-implementing its loop here) is the point: a
    re-implementation could agree with cascadir for reasons that have nothing to do with
    what production actually computes.
    """
    d = out_dir / "cm_prod"
    d.mkdir(parents=True, exist_ok=True)
    argv = [
        "run_binary_ig_probe.py",
        "--binary_run_dir", str(run_dir), "--output_dir", str(d),
        "--manifest_path", manifest, "--hvg_path", hvg,
        "--targets", *targets,
        "--top_n", str(top_n), "--n_ig_steps", str(n_steps),
        "--max_tubes_per_cytokine", str(cap), "--device", device,
    ]
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(REPO_ROOT / "scripts" / "run_binary_ig_probe.py"),
                       run_name="__main__")
    finally:
        sys.argv = old
    df = pd.read_parquet(d / "binary_ig.parquet")
    return df[["cytokine", "gene", "ig", "rank_ig"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary_run_dir", default=RUN_B)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--hvg_path", default=HVGS)
    ap.add_argument("--shard_dir", default=SHARDS)
    ap.add_argument("--out_dir", default="results/code_path/phase0")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Default: every cytokine with a model in the run dir.")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--deep_n", type=int, default=200,
                    help="Signature depth derived; comparisons are cut to --top_n.")
    ap.add_argument("--n_ig_steps", type=int, default=20)
    ap.add_argument("--max_tubes_per_cytokine", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_cm", action="store_true",
                    help="Reuse an existing cm_prod/binary_ig.parquet.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.binary_run_dir)

    with open(args.hvg_path) as fh:
        gene_names = [str(g) for g in json.load(fh)]
    n_genes = len(gene_names)

    targets = args.targets or sorted(
        p.name[len("model_"):-len(".pt")] for p in run_dir.glob("model_*.pt")
    )
    _log(f"[0b] run dir : {run_dir}")
    _log(f"[0b] targets : {len(targets)} -> {targets}")
    _log(f"[0b] HVGs    : {n_genes}")

    # ---- tube inventories -------------------------------------------------
    with open(args.manifest) as fh:
        manifest = json.load(fh)
    by_cyt: dict[str, list[dict]] = defaultdict(list)
    for e in manifest:
        by_cyt[str(e["cytokine"])].append(e)

    cap = args.max_tubes_per_cytokine
    _log(f"\n[0b] loading cm-selected tubes (first {cap} in manifest order)...")
    cm_ctrl = [_manifest_tube(e, gene_names) for e in by_cyt[CONTROL][:cap]]
    _log(f"  baseline(cm)   : {_describe(cm_ctrl)}")

    _log(f"[0b] loading main tubes from shards (tube_idx={MAIN_TUBE_INDICES})...")
    main_ctrl_set = load_tube_set(args.shard_dir, conditions=[CONTROL],
                                  include_control=True, tube_indices=MAIN_TUBE_INDICES)
    main_ctrl = [t for t in main_ctrl_set.tubes if t.condition == CONTROL]
    _log(f"  baseline(main) : {_describe(main_ctrl)}")

    # ---- models + transplant validation -----------------------------------
    tables: dict[str, list[dict]] = defaultdict(list)
    inventory: dict[str, dict] = {}
    max_logit_delta = 0.0

    cond_sources = ("cm", "main")
    base_sources = ("cm", "main")

    for cyt in targets:
        mp = run_dir / f"model_{_safe(cyt)}.pt"
        if not mp.exists():
            _log(f"  SKIP {cyt}: no model at {mp}")
            continue
        state = torch.load(mp, map_location="cpu", weights_only=False)
        cm_model = _load_binary_model(mp, n_genes, args.device)
        cd_model = build_cascadir_model(state, n_genes, args.device)
        max_logit_delta = max(
            max_logit_delta,
            assert_forward_equivalent(cm_model, cd_model, n_genes, args.device),
        )

        cond_tubes = {
            "cm": [_manifest_tube(e, gene_names) for e in by_cyt[cyt][:cap]],
            "main": [t for t in load_tube_set(
                args.shard_dir, conditions=[cyt], include_control=False,
                tube_indices=MAIN_TUBE_INDICES).tubes if t.condition == cyt],
        }
        for cs in cond_sources:
            inventory[f"{cyt}|cond.{cs}"] = _describe(cond_tubes[cs])

        for cs, bs in itertools.product(cond_sources, base_sources):
            ts = PseudoTubeSet(
                tubes=list(cond_tubes[cs]) + list(cm_ctrl if bs == "cm" else main_ctrl),
                gene_names=tuple(gene_names), control_label=CONTROL,
            )
            sig = derive_signature(cd_model, ts, cyt, top_n=args.deep_n,
                                   n_steps=args.n_ig_steps, device=args.device)
            tables[f"cd_cond.{cs}_base.{bs}"] += _sig_rows(cyt, sig, args.deep_n)
        _log(f"  [{cyt}] 4 cascadir configs done "
             f"(cond.cm={len(cond_tubes['cm'])} tubes, cond.main={len(cond_tubes['main'])})")
        del cm_model, cd_model, cond_tubes

    _log(f"\n[0b] transplant validated: max |Δlogit| across all models = {max_logit_delta:.3e}")

    # ---- production cytokine_mil ------------------------------------------
    cm_parquet = out / "cm_prod" / "binary_ig.parquet"
    if args.skip_cm and cm_parquet.exists():
        _log(f"[0b] reusing {cm_parquet}")
        cm_df = pd.read_parquet(cm_parquet)[["cytokine", "gene", "ig", "rank_ig"]]
    else:
        _log("\n[0b] running PRODUCTION run_binary_ig_probe.py ...")
        cm_df = run_cm_production(run_dir, out, targets, args.hvg_path, args.manifest,
                                  args.deep_n, args.n_ig_steps, cap, args.device)

    configs: dict[str, pd.DataFrame] = {"cm_prod": cm_df}
    for k, rows in tables.items():
        configs[k] = pd.DataFrame(rows)

    # ---- readouts ---------------------------------------------------------
    div_rows = []
    for name, df in configs.items():
        d = df[df.cytokine.isin(targets)]
        rec = {"config": name}
        rec.update(diversity(d, args.top_n))
        div_rows.append(rec)
    div = pd.DataFrame(div_rows)

    tops = {n: {c: set(g.nsmallest(args.top_n, "rank_ig").gene)
                for c, g in df.groupby("cytokine")} for n, df in configs.items()}
    jrows = []
    for a, b in itertools.combinations(configs, 2):
        shared = sorted(set(tops[a]) & set(tops[b]))
        js = [len(tops[a][c] & tops[b][c]) / len(tops[a][c] | tops[b][c]) for c in shared]
        jrows.append({"config_a": a, "config_b": b, "n_cytokines": len(shared),
                      "mean_jaccard": float(np.mean(js)) if js else np.nan,
                      "min_jaccard": float(np.min(js)) if js else np.nan})
    jac = pd.DataFrame(jrows)

    div.to_csv(out / "phase0b_diversity.csv", index=False)
    jac.to_csv(out / "phase0b_config_agreement.csv", index=False)
    for name, df in configs.items():
        df.to_parquet(out / f"phase0b_signatures_{name}.parquet", index=False)
    (out / "phase0b_tube_inventory.json").write_text(
        json.dumps({"baseline_cm": _describe(cm_ctrl),
                    "baseline_main": _describe(main_ctrl),
                    "per_cytokine": inventory,
                    "max_logit_delta": max_logit_delta}, indent=2, sort_keys=True))

    _log("\n" + "=" * 78)
    _log(f"DIVERSITY at top-{args.top_n} (lower mean_jaccard = more specific)")
    _log(div.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    _log("")
    _log("PER-CYTOKINE TOP-N AGREEMENT BETWEEN CONFIGS")
    _log(jac.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    _log("=" * 78)

    algo = jac.query("config_a == 'cm_prod' and config_b == 'cd_cond.cm_base.cm'")
    if len(algo):
        j = float(algo.mean_jaccard.iloc[0])
        _log(f"\n[GATE 0b] algorithm-only agreement (identical weights, cells, baseline): "
             f"mean top-{args.top_n} Jaccard = {j:.3f}")
        if j >= 0.95:
            _log("           PASSED (>= 0.95) — the two IG implementations agree; the")
            _log("           attribution CODE is not the difference. Any gap in the as-run")
            _log("           comparison is configuration, which the 2x2 above localises.")
        else:
            _log("           FAILED (< 0.95) — the IG implementations disagree on identical")
            _log("           inputs. The attribution step is at least part of the answer;")
            _log("           Phase 1 should not be run before this is understood.")
    _log(f"\n[write] {out}/phase0b_*.csv|parquet|json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
