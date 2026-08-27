"""Artifact loaders + the orchestrator builder for the §40 dropout+curation run.

THE HARD RULE (cascadir/MANUAL.md §3.1, CLAUDE.md §40.5): coupling and direction go
through `CascadeDirection.from_artifacts` and its methods. The bare module-level
`signature_coupling()` silently falls back to the over-powered cell-level null while
still returning a `coupled` column — **never call it**.

Most of §37's loaders take everything they need as arguments and are reused here
verbatim rather than copied: `load_tubes`, `load_signatures`, `model_path`,
`save_model_head` and `load_model` carry no §37-specific constants. Only the two
functions that bind run configuration are re-implemented — `load_encoder` (which must
carry §40's dropout through to the rebuilt module) and `build_estimator` (which must use
§40's `CrossAsymConfig`, i.e. top_n=200, and accept either signature arm).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "cascadir" / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _oes90_dc_config as C  # noqa: E402

# Reused unchanged — these are config-free (everything comes in as an argument).
from _oes90_pure_estimator import (  # noqa: E402,F401
    load_model,
    load_signatures,
    load_tubes,
    model_path,
    save_model_head,
)


def load_encoder(out_dir, device: str = "cpu", verbose: bool = True):
    """Rebuild the shared Stage-1 encoder and ASSERT its digest matches stage 1's.

    Same §27.6 guard as §37 — signatures derived under different encoders are not
    comparable, and both coupling and `cross_asym` compare signatures across cytokines,
    so a task that silently trained on its own encoder would corrupt every downstream
    number without erroring.

    §40 addition: `dropout` is read from the meta and passed to the rebuilt module. It is
    inert here (the encoder is returned in `eval()` mode, and `nn.Dropout` carries no
    parameters, so the state_dict and its digest are identical either way) — it is
    carried so the reconstructed object self-describes the fit rather than quietly
    looking like a no-dropout encoder.
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
        dropout=float(meta.get("dropout", 0.0)),
    )
    encoder.load_state_dict(state)
    encoder = encoder.to(device).eval()
    if verbose:
        C.log(
            f"[encoder] verified sha256={expected[:16]}...  embed_dim={meta['embed_dim']} "
            f"hidden={tuple(meta['hidden_dims'])} dropout={meta.get('dropout', 0.0)} "
            f"(ran {meta.get('n_epochs_run')} epochs, "
            f"restore_best={meta.get('restore_best')})"
        )
    return encoder, meta


def build_estimator(out_dir, arm: str = "curated", verbose: bool = True, conditions=None):
    """Load MAIN tubes + one signature arm; return a fitted `CascadeDirection` + provenance.

    `arm` selects which signature parquet is read: "curated" (the §40 result) or "raw"
    (the uncurated top-200 control). The two arms may cover **different condition sets** —
    curation drops conditions it empties — so any comparison between them must be made on
    the intersection, not row-by-row.

    `conditions` restricts the fit to a subset of stimuli (the control is always kept).
    Tubes and signatures are restricted together; a mismatch would leave the orchestrator
    scoring signatures whose cells are absent. Restricting changes the coupling degree
    terms (computed over whatever matrix is passed), which is why a subset analysis must
    be re-run rather than filtered afterwards; `cross_asym` is pairwise and unaffected.
    """
    import cascadir as cd

    out = Path(out_dir)
    signatures_name = C.arm_signatures(arm)
    keep = None if conditions is None else sorted({str(c) for c in conditions})
    tube_set, tube_meta = load_tubes(out, which="main", conditions=keep, verbose=verbose)
    signatures = load_signatures(out / signatures_name, top_n=C.TOP_N)
    if keep is not None:
        allowed = set(keep) | {C.CONTROL}
        missing = [c for c in keep if c not in signatures and c != C.CONTROL]
        if missing:
            raise ValueError(
                f"{len(missing)} requested conditions have no signature: {missing[:8]}"
            )
        signatures = {k: v for k, v in signatures.items() if k in allowed}
    if not signatures:
        raise ValueError(f"no signatures loaded from {signatures_name} — refusing to run.")

    sizes = sorted(len(s.genes) for s in signatures.values())
    if verbose:
        C.log(
            f"[signatures] arm={arm} ({signatures_name}): {len(signatures)} conditions, "
            f"gene count min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]} "
            f"(requested top_n={C.TOP_N})"
        )

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
        "arm": arm,
        "tubes_shards_sha256": tube_meta["shards_sha256"],
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "n_tubes": len(tube_set.tubes),
        "n_conditions": len(signatures),
        "donors": list(tube_set.donors),
        "cell_types": list(tube_set.cell_types),
        "top_n": C.TOP_N,
        "signature_sizes": {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]},
        "signatures_file": signatures_name,
    }
    return est, provenance
