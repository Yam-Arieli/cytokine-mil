"""Config for Phase 1+2 of the controlled code-path comparison.

`reports/code_path_comparison/SPEC.md`. Phase 0 established three things that shape this:

  * 0a — both paths read **bit-identical** tubes (9100/9100, gene order identical), so the
    gap is not a data difference;
  * 0b — the two IG implementations return **identical** top-50 signatures on identical
    weights (Jaccard 1.000), so attribution is not the difference;
  * 0b's 2x2 — IG tube/baseline selection moves meanJ by only +0.016 against a 0.10-0.30
    gap, so configuration is not the difference either.

What is left is the **trained weights**. This run therefore holds everything else fixed and
crosses the two things that produce weights:

    arm            Stage-1 encoder     Stage-2 binary head
    cm_cm          cytokine_mil        cytokine_mil          (= pure cytokine_mil, Phase 1 P)
    cd_cd          cascadir            cascadir              (= pure cascadir,     Phase 1 C)
    cm_cd          cytokine_mil        cascadir              (Phase 2 T1)
    cd_cm          cascadir            cytokine_mil          (Phase 2 T2)

Phase 1's two reference arms are the diagonal, so running the square gives both phases at
once — and Phase 2 is uninterpretable without them, since the only pure-`cytokine_mil`
reference in existence (run B, meanJ 0.079) has no matching pure-`cascadir` arm on the
same protocol.

**Attribution is held fixed across all four arms** — one implementation, one tube set, one
baseline — which Phase 0 licenses and which removes the last non-weight variable.

THREE SEEDS, and they are not a formality. The claimed effect (0.065-0.077 vs 0.178-0.394)
has never been compared against within-path seed variance at fixed settings. If the
within-arm spread is comparable to the between-arm gap, the code-path hypothesis dies here
regardless of where the means land.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))

OUT_DIR = REPO_ROOT / "results" / "code_path" / "phase2"

# Inputs — all already on disk, none rebuilt here.
SHARD_DIR = REPO_ROOT / "results" / "oes_full90" / "tubes"
STAGE1_CELLS = REPO_ROOT / "results" / "oes90_dc" / "stage1_cells.h5ad"
MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"

CONTROL = "PBS"
VAL_DONORS = ["Donor2", "Donor3"]      # §16; the shards already exclude them
CELLTYPE_COL = "cell_type"

# The published-24 panel, locked with the user (SPEC.md §7) so the result lands in the
# same ladder §38.3/§38.4 and §40 are quoted on. Hardcoded rather than read from
# binary_ig_all24.parquet so no stage depends on a results file at runtime.
PANEL = [
    "CD30L", "Decorin", "GM-CSF", "IFN-beta", "IFN-gamma", "IFN-lambda1", "IFN-omega",
    "IL-1-beta", "IL-10", "IL-12", "IL-13", "IL-15", "IL-16", "IL-17A", "IL-2", "IL-27",
    "IL-35", "IL-36-alpha", "IL-6", "IL-9", "TGF-beta1", "TL1A", "TNF-alpha", "VEGF",
]

# Published "wide" hyperparameters — pinned at the anchor's values on both sides.
EMBED_DIM = 512
HIDDEN_DIMS = (512, 512)
ATTENTION_HIDDEN_DIM = 128
STAGE1_EPOCHS = 20
STAGE1_LR = 0.005
STAGE1_MOMENTUM = 0.9
STAGE1_BATCH = 256
STAGE2_EPOCHS = 250
STAGE2_LR = 0.00003
STAGE2_MOMENTUM = 0.9

TOP_N = 50          # the comparison cut, matching the §38 ladder
DEEP_N = 200        # derived depth; comparisons are cut to TOP_N
IG_STEPS = 20

SEEDS = [42, 123, 7]
PATHS = ["cm", "cd"]
ARMS = [(e, s) for e in PATHS for s in PATHS]   # (encoder_path, stage2_path)


def arm_name(encoder_path: str, stage2_path: str) -> str:
    return f"{encoder_path}_{stage2_path}"


def encoder_path_dir(path: str, seed: int) -> Path:
    return OUT_DIR / f"encoder_{path}_seed{seed}"


def arm_dir(encoder_path: str, stage2_path: str, seed: int) -> Path:
    return OUT_DIR / f"arm_{arm_name(encoder_path, stage2_path)}_seed{seed}"


def log(msg: str = "") -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def state_dict_sha256(state) -> str:
    """Digest over a state_dict, so an arm can never silently use another's encoder."""
    import hashlib

    import numpy as np
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def mark_done(out_dir: Path, stage: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"DONE_{stage}").write_text(datetime.now(timezone.utc).isoformat())
