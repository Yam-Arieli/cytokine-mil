"""Shared constants + helpers for the Oesinghaus 90-cytokine PURE run (CLAUDE.md §37).

Written from scratch rather than importing `scripts/_full90_config.py`, deliberately.
This run is **cytokine-agnostic**: no stage may consult the audited pair list or the
published 24-cytokine panel, so the constants that point at them (`AUDITED_CSV`,
`PUBLISHED_COUPLING_CSV`, `load_audited_labels`) are not defined here and are not in
scope for anything that imports this module. :func:`assert_agnostic` makes that a runtime
check rather than a convention.

This module holds no method math. Every statistic comes from `cascadir`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- data ----------------------------------------------------------------
# The tube shards are REUSED from the §36 prepare stage: 9100 tubes over 90 cytokines +
# PBS across the 10 training donors, materialised once from the committed pseudo-tubes
# with D2/D3 already excluded. They are read-only here and sha-verified at every stage.
SHARD_DIR = "/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/tubes"
OUT_DIR = REPO_ROOT / "results" / "oes90_pure"

# The shards carry no barcodes, so the Stage-1 unique-cell set is read from the source
# pseudo-tubes (same files the shards were built from) where obs_names are available.
MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"

CONTROL = "PBS"
VAL_DONORS = ["Donor2", "Donor3"]  # CLAUDE.md §16 — held out EVERYWHERE, Stage-1 included

CONDITION_COL = "cytokine"
DONOR_COL = "donor"
CELLTYPE_COL = "cell_type"

# ---- tube split ----------------------------------------------------------
# The SAME indices in every (donor, condition) group, so the split encodes no
# per-cytokine choice. MAIN drives the whole method; RESERVE is used only to re-derive
# the signatures a second time for the memorisation check (CLAUDE.md §37).
MAIN_TUBE_INDICES = [0, 1, 2, 3]
RESERVE_TUBE_INDICES = [4, 5, 6, 7]

# ---- Stage-1 cell budget -------------------------------------------------
# Equal weight per cytokine and per donor: cap each cytokine at CELLS_PER_CYTOKINE unique
# cells drawn evenly across the donors. Without the cap, cytokines whose tubes happen to
# draw from larger cell pools would dominate the encoder.
CELLS_PER_CYTOKINE = 2000
STAGE1_CELL_SEED = 42

# ---- model / training ----------------------------------------------------
# 2x the published "wide" config on the encoder (512 -> 1024). Everything else matches
# the published wide config; see CLAUDE.md §37 for the recorded deviations.
EMBED_DIM = 1024
HIDDEN_DIMS = (1024, 1024)
ATTENTION_HIDDEN_DIM = 128
STAGE1_EPOCHS = 200          # an upper bound; early stopping decides where it lands
STAGE1_LR = 0.005
STAGE1_VAL_FRACTION = 0.10
STAGE1_PATIENCE = 10
STAGE1_MIN_DELTA = 1e-4
STAGE1_EXTRA_EPOCHS = 20     # keep going past the plateau so overfitting is recorded
STAGE2_EPOCHS = 250
STAGE2_LR = 0.00003
MOMENTUM = 0.9
SEED = 42

# ---- signatures / statistic ---------------------------------------------
TOP_N = 100
N_IG_STEPS = 20
MIN_CELLS = 10
N_NULL_PERMS = 100
NULL_SEED = 42

# ---- DAG ----------------------------------------------------------------
N_CHUNKS = 9  # 90 cytokines / 9 tasks = 10 per task
COUPLING_ALPHA = 0.05
FDR_QS = (0.05, 0.10)

# Files this run must never read (the mechanical agnosticism guard).
FORBIDDEN_INPUTS = (
    "cytokine_axes_audited.csv",
    "cytokine_axes.csv",
    "donor_coupling_hub_IG_vsPBS.csv",
    "literature_review_aggregate.json",
    "binary_ig_all24",
)


def assert_agnostic() -> None:
    """Fail loudly if a benchmark artefact has been imported or opened by this process.

    The point of this run is that no cytokine is privileged. The most likely way that
    breaks is a stage quietly reading the audited pair list to "sanity check" something,
    which would let benchmark knowledge steer a choice. Checking the loaded module set is
    cheap and catches the realistic version of the mistake (importing `_full90_config`,
    which carries those paths).
    """
    if "_full90_config" in sys.modules or "scripts._full90_config" in sys.modules:
        raise RuntimeError(
            "_oes90_pure_config.assert_agnostic: `_full90_config` is imported. It carries "
            "AUDITED_CSV / PUBLISHED_COUPLING_CSV, which this run must not see."
        )


def cross_asym_config():
    """The `CrossAsymConfig` every stage must share (signature size drives everything)."""
    from cascadir.config import CrossAsymConfig

    return CrossAsymConfig(
        top_n=TOP_N,
        n_ig_steps=N_IG_STEPS,
        min_cells=MIN_CELLS,
        n_null_perms=N_NULL_PERMS,
        null_seed=NULL_SEED,
    )


def state_dict_sha256(state_dict) -> str:
    """Stable digest of a torch state_dict — the CLAUDE.md §27.6 guard.

    Every chunk task recomputes this over the encoder it loaded and asserts it equals the
    digest Stage 1 wrote. Signatures from different encoders are not comparable, and both
    coupling and `cross_asym` compare signatures across cytokines, so a sharded encoder
    silently corrupts every downstream number.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode())
        t = state_dict[key]
        arr = t.detach().cpu().numpy()
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def chunk_conditions(all_conditions, chunk_id: int, n_chunks: int) -> list:
    """Deterministic round-robin slice of the sorted stimulus list for one array task.

    Round-robin rather than contiguous so a task never gets a block of unusually
    large/small conditions; the union over all chunks is exactly the input. The list is
    sorted first, so the assignment depends on nothing but the names.
    """
    conds = sorted(all_conditions)
    if not 0 <= chunk_id < n_chunks:
        raise ValueError(f"chunk_id={chunk_id} out of range for n_chunks={n_chunks}")
    return [c for i, c in enumerate(conds) if i % n_chunks == chunk_id]


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def log(msg: str) -> None:
    print(msg, flush=True)


def mark_done(out_dir, stage: str) -> None:
    """Touch the DONE marker the DAG's watchdog and sentinels look for."""
    p = Path(out_dir) / f"DONE_{stage}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
