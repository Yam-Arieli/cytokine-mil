"""Shared loaders for the Oesinghaus 90-cytokine PURE run.

Every stage after the encoder needs the same four things — the shared encoder (with its
digest verified), the MAIN tube subset, a saved binary model, and a fitted
`cascadir.CascadeDirection` — so they live here rather than being re-derived per script.

`CascadeDirection.from_artifacts` rebuilds exactly the state `fit()` leaves, which is what
keeps coupling and direction on the orchestrator. The bare module-level
`signature_coupling()` silently falls back to an over-powered cell-level null while still
returning a `coupled` column (cascadir/MANUAL.md §3.1) — never call it.

No statistics here: this loads files and hands them to cascadir.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _oes90_pure_config as C  # noqa: E402
from cytokine_mil.analysis.full90_tube_io import load_tube_set, read_meta  # noqa: E402


# ---------------------------------------------------------------------------
# Encoder — the CLAUDE.md §27.6 guard
# ---------------------------------------------------------------------------


def load_encoder(out_dir, device: str = "cpu", verbose: bool = True):
    """Rebuild the shared Stage-1 encoder and ASSERT its digest matches stage 1's.

    Signatures derived under different encoders are not comparable, and both coupling and
    `cross_asym` compare signatures across cytokines — so a task that silently trained on
    its own encoder would corrupt every downstream number without erroring. This refuses
    to proceed on a mismatch.
    """
    import torch

    from cascadir.models import InstanceEncoder

    out = Path(out_dir)
    meta = C.read_json(out / "encoder_meta.json")
    expected = (out / "encoder_sha256.txt").read_text().strip()
    if meta["sha256"] != expected:
        raise AssertionError(
            f"encoder_meta.json sha ({meta['sha256'][:16]}...) disagrees with "
            f"encoder_sha256.txt ({expected[:16]}...) — the artifacts are inconsistent."
        )

    state = torch.load(out / "encoder.pt", map_location="cpu")
    actual = C.state_dict_sha256(state)
    if actual != expected:
        raise AssertionError(
            "ENCODER MISMATCH (CLAUDE.md §27.6 guard): encoder.pt hashes to "
            f"{actual[:16]}... but stage 1 recorded {expected[:16]}.... Every chunk must "
            "train on the identical shared encoder; refusing to run."
        )

    encoder = InstanceEncoder(
        input_dim=int(meta["n_genes"]),
        embed_dim=int(meta["embed_dim"]),
        n_cell_types=int(meta["n_cell_types"]),
        hidden_dims=tuple(meta["hidden_dims"]),
    )
    encoder.load_state_dict(state)
    encoder = encoder.to(device).eval()
    if verbose:
        C.log(
            f"[encoder] verified sha256={expected[:16]}...  embed_dim={meta['embed_dim']} "
            f"hidden={tuple(meta['hidden_dims'])} (best epoch {meta['best_epoch']})"
        )
    return encoder, meta


# ---------------------------------------------------------------------------
# Tubes
# ---------------------------------------------------------------------------


def _split_info(out_dir) -> dict:
    return C.read_json(Path(out_dir) / "tube_split.json")


def load_tubes(out_dir, which: str = "main", conditions=None, verbose: bool = True):
    """Load the MAIN or RESERVE tube subset, asserting the shard digest is stage 0's."""
    import time

    split = _split_info(out_dir)
    meta = read_meta(split["shard_dir"])
    if meta["shards_sha256"] != split["shards_sha256"]:
        raise AssertionError(
            "TUBE MISMATCH: the shard set hashes to "
            f"{meta['shards_sha256'][:16]}... but stage 0 recorded "
            f"{split['shards_sha256'][:16]}.... Refusing to run."
        )
    key = {"main": "main_tube_indices", "reserve": "reserve_tube_indices"}[which]
    idx = split[key]

    t0 = time.time()
    ts = load_tube_set(
        split["shard_dir"],
        conditions=conditions,
        include_control=True,
        tube_indices=idx,
    )
    if verbose:
        n_cells = sum(t.n_cells for t in ts.tubes)
        C.log(
            f"[tubes:{which}] tube_idx={idx} -> {len(ts.tubes)} tubes, {n_cells} cells, "
            f"{len(ts.conditions)} conditions, {len(ts.donors)} donors, "
            f"{len(ts.cell_types)} cell types ({time.time()-t0:.0f}s)"
        )
    return ts, meta


# ---------------------------------------------------------------------------
# Binary models
# ---------------------------------------------------------------------------


def model_path(out_dir, condition: str) -> Path:
    """Filesystem-safe path for one condition's saved head."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(condition))
    return Path(out_dir) / "models" / f"{safe}_head.pt"


def save_model_head(model, out_dir, condition: str, encoder_sha: str) -> Path:
    """Persist ONLY the attention + classifier weights.

    The encoder is already saved and digest-verified, and is frozen throughout, so storing
    it again inside all 90 models would multiply the same 25 MB by 90 for nothing. The
    digest is written alongside so a head can never be recombined with the wrong encoder.
    """
    import torch

    p = model_path(out_dir, condition)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "condition": condition,
            "encoder_sha256": encoder_sha,
            "attention": model.attention.state_dict(),
            "classifier": model.classifier.state_dict(),
            "attention_hidden_dim": C.ATTENTION_HIDDEN_DIM,
        },
        p,
    )
    return p


def load_model(out_dir, condition: str, encoder, encoder_sha: str, device: str = "cpu"):
    """Rebuild the `AbMil` for one condition from the shared encoder + its saved head."""
    import torch

    from cascadir.models import AbMil, AttentionModule, BagClassifier

    blob = torch.load(model_path(out_dir, condition), map_location="cpu")
    if blob["encoder_sha256"] != encoder_sha:
        raise AssertionError(
            f"head for {condition!r} was trained on encoder "
            f"{blob['encoder_sha256'][:16]}... but the loaded encoder is "
            f"{encoder_sha[:16]}.... Refusing to recombine."
        )
    attention = AttentionModule(
        embed_dim=encoder.embed_dim,
        attention_hidden_dim=int(blob["attention_hidden_dim"]),
    )
    attention.load_state_dict(blob["attention"])
    classifier = BagClassifier(embed_dim=encoder.embed_dim, n_classes=2)
    classifier.load_state_dict(blob["classifier"])
    model = AbMil(encoder, attention, classifier, encoder_frozen=True).to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Signatures + the fitted estimator
# ---------------------------------------------------------------------------


def load_signatures(parquet_path, top_n: int = C.TOP_N) -> dict:
    """Read a signatures parquet into cascadir's `Signature` dataclass.

    Selection is by the precomputed `rank_ig` written by `derive_signatures` — the IG
    values are read, never recomputed or re-ranked.
    """
    import pandas as pd

    from cascadir.types import Signature

    df = pd.read_parquet(parquet_path)
    out = {}
    for cond, sub in df.groupby("cytokine"):
        sub = sub.sort_values("rank_ig").head(top_n)
        out[str(cond)] = Signature(
            condition=str(cond),
            genes=tuple(str(g) for g in sub["gene"]),
            ig_scores=tuple(float(v) for v in sub["ig"]),
            top_n=top_n,
        )
    return out


def build_estimator(out_dir, signatures_name: str = "signatures_main.parquet",
                    verbose: bool = True, conditions=None):
    """Load MAIN tubes + signatures and return a fitted `CascadeDirection` (+ provenance).

    `conditions` restricts the fit to a subset of stimuli (the control is always kept).
    Tubes and signatures are restricted together — a mismatch between them would leave
    the orchestrator scoring signatures whose cells are absent. Restricting changes the
    coupling degree terms (they are computed over whatever matrix is passed), which is
    exactly why a subset analysis has to be re-run rather than filtered after the fact;
    `cross_asym` is pairwise and is unaffected either way.
    """
    import cascadir as cd

    out = Path(out_dir)
    keep = None if conditions is None else sorted({str(c) for c in conditions})
    tube_set, tube_meta = load_tubes(out, which="main", conditions=keep, verbose=verbose)
    signatures = load_signatures(out / signatures_name)
    if keep is not None:
        allowed = set(keep) | {C.CONTROL}
        missing = [c for c in keep if c not in signatures and c != C.CONTROL]
        if missing:
            raise ValueError(
                f"{len(missing)} requested conditions have no signature: {missing[:8]}"
            )
        signatures = {k: v for k, v in signatures.items() if k in allowed}
    if verbose:
        C.log(f"[signatures] {len(signatures)} conditions x {C.TOP_N} genes "
              f"({signatures_name})")

    est = cd.CascadeDirection.from_artifacts(
        tube_set,
        signatures,
        condition_col=C.CONDITION_COL,
        donor_col=C.DONOR_COL,
        celltype_col=C.CELLTYPE_COL,
        control_label=C.CONTROL,
        cross_asym_config=C.cross_asym_config(),
        device="cpu",
        seed=C.SEED,
    )
    provenance = {
        "tubes_shards_sha256": tube_meta["shards_sha256"],
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "n_tubes": len(tube_set.tubes),
        "n_conditions": len(signatures),
        "donors": list(tube_set.donors),
        "cell_types": list(tube_set.cell_types),
        "top_n": C.TOP_N,
        "signatures_file": signatures_name,
    }
    return est, provenance
