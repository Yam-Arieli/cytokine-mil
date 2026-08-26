"""Shared constants for the Stage-1 encoder condition-breadth sweep.

Why this sweep exists
---------------------
On the identical 24 cytokines at the identical top-50 cut, mean between-cytokine signature
Jaccard runs 0.065 (published anchor) -> 0.178 (§36 full90) -> 0.241 (§37 PURE). The single
largest step is published -> §36, where the ONLY thing that changed was the Stage-1
encoder's training set: 17-18 conditions (with D2/D3) became 90 conditions (without them).
Every hyperparameter was identical.

That step can only be an encoder effect. A binary model is trained on {X tubes, PBS tubes}
alone (cascadir/src/cascadir/train.py:469-470), so the number of conditions in the run
cannot reach a signature by any route except through the shared encoder.

The mechanism is that Stage 1 optimises the encoder for CELL-TYPE classification, for which
cytokine-induced variation is nuisance variance *within* a cell type. The more conditions
the encoder is shown, the more perturbation diversity it is explicitly trained to be
invariant to — so it discards exactly the signal the downstream method needs.

This sweep varies only that: how many distinct perturbations Stage 1 sees. Everything else
is pinned at the published values.

Discipline
----------
Like `_oes90_pure_config`, this module is written standalone and does NOT import
`_full90_config`, so the audited pair list and the published 24-cytokine panel are not in
scope. The encoder subsets and the readout panel are seeded random draws from the sorted
90; no cytokine is chosen for what it is. It reuses `_oes90_pure_config` only for
file/digest plumbing (`state_dict_sha256`, `read_json`, `log`, ...), which carries no
benchmark knowledge and keeps `assert_agnostic()` meaningful.

This module holds no method math. Every statistic comes from `cascadir`.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _oes90_pure_config as P  # noqa: E402

# Reused plumbing — no benchmark knowledge in any of these.
assert_agnostic = P.assert_agnostic
state_dict_sha256 = P.state_dict_sha256
chunk_conditions = P.chunk_conditions
read_json = P.read_json
write_json = P.write_json
log = P.log
mark_done = P.mark_done

REPO_ROOT = P.REPO_ROOT

# ---- data (read-only, shared with §36/§37) -------------------------------
SHARD_DIR = P.SHARD_DIR
MANIFEST_PATH = P.MANIFEST_PATH
OUT_DIR = REPO_ROOT / "results" / "encsweep"

CONTROL = P.CONTROL
VAL_DONORS = P.VAL_DONORS          # CLAUDE.md §16 — held out EVERYWHERE, Stage-1 included
CONDITION_COL = P.CONDITION_COL
DONOR_COL = P.DONOR_COL
CELLTYPE_COL = P.CELLTYPE_COL

# ---- tubes ---------------------------------------------------------------
# k=10: the full tube budget, matching the published anchor and §36. §37's k=4 is already
# known to be the worse of the two, so it is not a factor here.
MAIN_TUBE_INDICES = list(range(10))
RESERVE_TUBE_INDICES: list[int] = []   # no memorisation check in this sweep

# ---- model / training: PINNED at the published wide config ---------------
# scripts/train_oesinghaus_binary_missing16.py:108-115. No early stopping: the published
# encoder ran a fixed 20 epochs, and §37's val-loss early stop landed at epoch 4, which is
# a fourth deviation this sweep deliberately removes.
EMBED_DIM = 512
HIDDEN_DIMS = (512, 512)
ATTENTION_HIDDEN_DIM = 128
STAGE1_EPOCHS = 20
STAGE1_LR = 0.005
STAGE2_EPOCHS = 250
STAGE2_LR = 0.00003
MOMENTUM = 0.9
SEED = 42

# ---- signatures ----------------------------------------------------------
TOP_N = 50
N_IG_STEPS = 20

# ---- the sweep itself ----------------------------------------------------
# Total Stage-1 cells is held FIXED across arms. Without this, condition breadth would be
# confounded with cell count and therefore with gradient exposure, and the sweep would
# measure nothing. With epochs also fixed, every arm gets identical gradient exposure and
# the only variable is how many distinct perturbations the encoder must be invariant to.
#
# PBS is present in every arm (it is the negative class of every binary model, so the
# encoder must be competent on it) and counts as one of the groups the budget is split over.
# If the `pbs_only` arm cannot supply this many unique PBS cells, prepare lowers the budget
# for ALL arms to what PBS can supply and records it — a smaller controlled sweep is worth
# more than a larger uncontrolled one.
STAGE1_TOTAL_CELLS = 36000
STAGE1_CELL_SEED = 42

# Per-(condition, donor) caps for the unique-cell bank prepare builds once. PBS needs the
# most because the `pbs_only` arm spends the whole budget on it; the stimulus cap covers
# the hungriest stimulus arm (rand18, which needs budget/19 per condition).
PBS_CAP_PER_DONOR = 4000
STIM_CAP_PER_DONOR = 250

# Nested draws from a single seeded permutation of the sorted 90, so rand18 c rand45 c
# all90. Nesting removes subset-composition noise from the ladder: consecutive arms differ
# only by the conditions added, not by a fresh random set.
ENCODER_SUBSET_SEED = 20260825
ARM_N_CONDITIONS = {"pbs_only": 0, "rand18": 18, "rand45": 45, "all90": None}
ARMS = ("pbs_only", "rand18", "rand45", "all90")

# The readout panel: binary models are trained for these conditions in every arm.
PANEL_SEED = 424242
PANEL_SIZE = 24

N_CHUNKS = 4  # 24 panel cytokines / 4 tasks = 6 per task


def arm_dir(out_dir, arm: str) -> Path:
    """Directory for one arm's artefacts.

    Deliberately NOT validated against `ARMS`: this module is shared by the breadth sweep
    (CLAUDE.md §38.1) and the Stage-1-construction sweep (§38.4), whose arm names differ.
    What must still be rejected is a name that would escape `out_dir` or collide with the
    run-level files sitting beside the arm directories.
    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", arm or ""):
        raise ValueError(
            f"unsafe arm name {arm!r}; expected [A-Za-z0-9_]+ (it becomes a directory)"
        )
    return Path(out_dir) / arm


def draw_encoder_subset(all_conditions, arm: str) -> list:
    """The stimulus conditions this arm's Stage-1 encoder is allowed to see.

    A seeded permutation of the sorted stimulus list, truncated — so the arms are nested
    and the draw depends on nothing but the names and the committed seed.
    `pbs_only` returns [] (the control is added separately by the caller).
    """
    import numpy as np

    n = ARM_N_CONDITIONS[arm]
    conds = sorted(str(c) for c in all_conditions if str(c) != CONTROL)
    if n == 0:
        return []
    perm = np.random.default_rng(ENCODER_SUBSET_SEED).permutation(len(conds))
    take = len(conds) if n is None else min(n, len(conds))
    return sorted(conds[i] for i in perm[:take])


def draw_panel(all_conditions) -> list:
    """The fixed readout panel — a seeded random draw, never a benchmark list."""
    import numpy as np

    conds = sorted(str(c) for c in all_conditions if str(c) != CONTROL)
    perm = np.random.default_rng(PANEL_SEED).permutation(len(conds))
    return sorted(conds[i] for i in perm[:min(PANEL_SIZE, len(conds))])
