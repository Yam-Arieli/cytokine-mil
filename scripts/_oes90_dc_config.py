"""Shared constants + helpers for the Oesinghaus-90 dropout+curation run (CLAUDE.md §40).

§40 keeps §37 PURE's condition set, tube split and Stage-2 schedule and changes four
things: 50% dropout on the input of the encoder's final block, `top_n` 100 -> 200,
null-calibrated promiscuous-gene curation after the IG merge, and a fixed 20-epoch
Stage 1 with the validation curve recorded but **not** used to pick the checkpoint.

Like §38's encsweep this imports `_oes90_pure_config` for **plumbing only** — the digest,
chunking, json and DONE-marker helpers, and the data paths. That module is itself written
from scratch precisely so it carries no benchmark knowledge (`AUDITED_CSV`,
`PUBLISHED_COUPLING_CSV`, `load_audited_labels` are not in scope there), which is what
keeps :func:`assert_agnostic` meaningful when re-exported here.

This module holds no method math. Every statistic comes from `cascadir`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "cascadir" / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _oes90_pure_config as P  # noqa: E402  (plumbing only — see the module docstring)

# ---- re-exported plumbing (identical semantics; no benchmark knowledge) ----
assert_agnostic = P.assert_agnostic
state_dict_sha256 = P.state_dict_sha256
chunk_conditions = P.chunk_conditions
write_json = P.write_json
read_json = P.read_json
log = P.log
mark_done = P.mark_done

# ---- data (the §36 shards, read-only and sha-verified at every stage) -----
SHARD_DIR = P.SHARD_DIR
MANIFEST_PATH = P.MANIFEST_PATH
HVG_PATH = P.HVG_PATH
OUT_DIR = REPO_ROOT / "results" / "oes90_dc"

CONTROL = P.CONTROL
VAL_DONORS = P.VAL_DONORS  # §16 — held out EVERYWHERE, Stage-1 included

CONDITION_COL = P.CONDITION_COL
DONOR_COL = P.DONOR_COL
CELLTYPE_COL = P.CELLTYPE_COL

# ---- tube split ----------------------------------------------------------
# Identical to §37: k=4 main, a disjoint k=4 reserve, the SAME tube_idx values in every
# (donor, condition) group so the split encodes no per-cytokine choice. Stage 0 is
# literally §37's prepare script, so these must not drift from it.
MAIN_TUBE_INDICES = P.MAIN_TUBE_INDICES
RESERVE_TUBE_INDICES = P.RESERVE_TUBE_INDICES
CELLS_PER_CYTOKINE = P.CELLS_PER_CYTOKINE
STAGE1_CELL_SEED = P.STAGE1_CELL_SEED

# ---- model / training ----------------------------------------------------
# Width is UNCHANGED from §37 (2x the published wide config). It is listed here rather
# than re-exported so the run is self-describing, but note in any writeup that width is
# therefore *not* a new variable relative to §37 — see CLAUDE.md §40.3.
EMBED_DIM = 1024
HIDDEN_DIMS = (1024, 1024)
ATTENTION_HIDDEN_DIM = 128

# §40 intervention 1: dropout immediately before the embedding layer, aimed at the
# gene-space collapse §39.5 measured (4000 genes -> ~85-108 effective dimensions).
ENCODER_DROPOUT = 0.5

# Stage 1: a FIXED schedule. §37 early-stopped on val loss and restored the best
# checkpoint, which landed at epoch 4 — so "how long Stage 1 ran" moved together with
# every other change. Here the val split is still held out and recorded (so the
# overfitting §37 saw is still observable) but it does not pick the checkpoint.
STAGE1_EPOCHS = 20
STAGE1_LR = 0.005
STAGE1_VAL_FRACTION = 0.10
STAGE1_PATIENCE = None      # no early stopping
STAGE1_RESTORE_BEST = False  # keep the final-epoch weights

# Stage 2: unchanged from §37 / the published wide config.
STAGE2_EPOCHS = 250
STAGE2_LR = 0.00003
MOMENTUM = 0.9
SEED = 42

# ---- signatures ----------------------------------------------------------
TOP_N = 200
N_IG_STEPS = P.N_IG_STEPS
MIN_CELLS = P.MIN_CELLS
N_NULL_PERMS = P.N_NULL_PERMS
NULL_SEED = P.NULL_SEED

# ---- curation (CLAUDE.md §40.2) ------------------------------------------
# The cap is DERIVED from (n_conditions, top_n, n_genes) at merge time, never hardcoded:
# a fixed cap means very different things at different scales. This target is the
# stringency that a cap of 3 carries on a 24-condition panel at top-200, i.e.
# null_expected_removal(24, 200, 4000, cap=3).
CURATION_TARGET_NULL_REMOVAL = 0.1052
CURATION_MIN_GENES = 1  # drop only conditions curated away entirely

# ---- orchestration -------------------------------------------------------
N_CHUNKS = 9              # 90 cytokines / 9 = 10 per array task
N_DIRECTION_SHARDS = 4    # the per-pair null scales with top_n; see CLAUDE.md §40.6
COUPLING_ALPHA = 0.05
FDR_QS = (0.05, 0.10)

# The signature arms carried through coupling and direction. "curated" is the §40
# result; "raw" is the within-run control that says whether the curation did anything.
ARMS = {"curated": "signatures_main_curated.parquet", "raw": "signatures_main.parquet"}


def arm_signatures(arm: str) -> str:
    """Parquet filename for a signature arm. Raises on an unknown arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    return ARMS[arm]


def cross_asym_config():
    """The `CrossAsymConfig` every stage must share (signature size drives everything).

    `max_signature_occurrences` is deliberately left at ``None``: curation happens once,
    at merge time, over all 90 conditions at once (an occurrence count cannot be computed
    per chunk), and the curated signatures are then read from parquet. Setting it here as
    well would curate an already-curated set.
    """
    from cascadir.config import CrossAsymConfig

    return CrossAsymConfig(
        top_n=TOP_N,
        n_ig_steps=N_IG_STEPS,
        min_cells=MIN_CELLS,
        n_null_perms=N_NULL_PERMS,
        null_seed=NULL_SEED,
    )
