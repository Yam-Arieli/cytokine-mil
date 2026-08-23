"""Shared constants + helpers for the Oesinghaus full-90 DAG (CLAUDE.md §26/§28/§29).

One place for the values every stage must agree on. The hyperparameters are the
**published "wide" Oesinghaus config** (`scripts/train_oesinghaus_binary_missing16.py:108-115`
— the config behind `binary_ig_all24` and the published 88%), NOT `cascadir.TrainConfig`'s
packaged defaults, so the audited-pair regression check compares like with like.

This module holds no method math. Every statistic comes from `cascadir`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- data ----------------------------------------------------------------
MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
AUDITED_CSV = REPO_ROOT / "reports" / "cascade_pairs" / "cytokine_axes_audited.csv"
PUBLISHED_COUPLING_CSV = (
    REPO_ROOT / "reports" / "coupling_figures_draft" / "donor_coupling_hub_IG_vsPBS.csv"
)

CONTROL = "PBS"
VAL_DONORS = ["Donor2", "Donor3"]  # CLAUDE.md §16 — held out EVERYWHERE, Stage-1 included

CONDITION_COL = "cytokine"
DONOR_COL = "donor"
CELLTYPE_COL = "cell_type"

# ---- model / training (published "wide" config) --------------------------
EMBED_DIM = 512
HIDDEN_DIMS = (512, 512)
ATTENTION_HIDDEN_DIM = 128
STAGE1_EPOCHS = 20
STAGE1_LR = 0.005
STAGE2_EPOCHS = 250
STAGE2_LR = 0.00003
MOMENTUM = 0.9
SEED = 42

# ---- signatures / statistic ---------------------------------------------
TOP_N = 50
N_IG_STEPS = 20
MIN_CELLS = 10
N_NULL_PERMS = 100
NULL_SEED = 42

# ---- DAG ----------------------------------------------------------------
N_CHUNKS = 9  # 90 cytokines / 9 tasks = 10 per task
COUPLING_ALPHA = 0.05
FDR_QS = (0.05, 0.10)


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
    """Stable digest of a torch state_dict — the §27.6 guard.

    Every chunk task recomputes this over the encoder it loaded and asserts it equals the
    digest Stage 1 wrote. Signatures from different encoders are not comparable, and
    `cross_asym` compares signatures across cytokines, so a sharded encoder silently
    corrupts every downstream number.
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
    large/small conditions; the union over all chunks is exactly the input.
    """
    conds = sorted(all_conditions)
    if not 0 <= chunk_id < n_chunks:
        raise ValueError(f"chunk_id={chunk_id} out of range for n_chunks={n_chunks}")
    return [c for i, c in enumerate(conds) if i % n_chunks == chunk_id]


def load_audited_labels(path=None):
    """`(upstream, downstream)` pairs for the 17 `counts_in_benchmark=True` audited rows.

    `expected_sign` is the sign convention of the alphabetical pair (a, b): +1 means a is
    upstream, -1 means b is upstream (CLAUDE.md §26.1). `est.benchmark` wants explicit
    (upstream, downstream) tuples.
    """
    import csv

    out = []
    with open(path or AUDITED_CSV) as fh:
        for row in csv.DictReader(fh):
            if str(row.get("counts_in_benchmark")).lower() != "true":
                continue
            a, b = row["axis_a"], row["axis_b"]
            sign = int(float(row["expected_sign"]))
            out.append((a, b) if sign > 0 else (b, a))
    return out


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
