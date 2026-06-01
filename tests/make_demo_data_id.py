"""
Generate simulated demo data for the Immune Dictionary adapter
(`scripts/build_pseudotubes_immune_dictionary.py`).

This represents the input to the ID adapter's pseudo-tube construction step —
i.e., the AnnData produced after Leiden cell typing and PBS relabeling, ready
for `build_pseudo_tubes_id()`.

In real use, the adapter loads 10x MTX files, applies preprocess_id(), runs
Leiden, and relabels PBS cells. For testing we synthesize a small AnnData
with the exact column convention the adapter's pseudo-tube function expects:

  obs columns (required):
    - mouse_id   : str   (e.g., "mouse_1", "mouse_2", "mouse_3")
    - cytokine   : str   (PBS-relabeled; e.g., "PBS", "IL-12", "IFN-g")
    - cell_type  : str   (e.g., "id_c0", "id_c1")

  X: log-normalized counts, dtype float32, shape (n_cells, n_genes).
     First 50 gene names include JAK-STAT pathway marker genes so that
     the §24 pathway_signatures library has hits to resolve against the demo.

Spec for the demo:
  - 5 stimuli + PBS = 6 classes:
      "IL-12", "IFN-g", "IL-1b", "IL-6", "TNF", "PBS"
  - 3 mice: "mouse_1", "mouse_2", "mouse_3"
    PBS cells are pooled from all mice (each mouse contributes PBS cells
    under the PBS-injected control design, like the real ID dataset).
  - 2 cell types: "id_c0", "id_c1"
  - 30 cells per (mouse, cytokine, cell_type) → 60 cells per tube
  - 200 simulated genes (log-normalized via Poisson + log1p)
    Gene names: Gene_0 through Gene_199.
    Positions 0–49 include canonical JAK-STAT pathway marker genes
    (mouse symbols) so that pathway_signatures.py can resolve them:
      Il2ra, Ifng, Gbp2, Socs3, Foxp3, Arg1, Smad7, Stat1, Stat3, Stat4,
      Stat5a, Ifnb1, Il6ra, Il12rb1, Il12rb2, Tnfrsf1a, Nfkbia, Il1r1,
      Irf1, Irf4, Irf7, Mx1, Mx2, Cxcl10, Ccl5, Ifit1, Ifit2, Ifit3,
      Rsad2, Il6st  (30 pathway genes in first 50 positions)
    Remaining gene names: Gene_030 through Gene_199.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import anndata as ad
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Demo design constants (exported for use in test_demo_id.py)
# ---------------------------------------------------------------------------
MICE = ["mouse_1", "mouse_2", "mouse_3"]
MICE_TRAIN = ["mouse_1", "mouse_2"]
MICE_VAL = ["mouse_3"]

# Cytokine names matching the §25 pre-registration canonical names.
# PBS is already relabeled (not raw control string) — represents the adapter
# output state that build_pseudo_tubes_id() consumes.
ACTIVE_CYTOKINES = ["IL-12", "IFN-g", "IL-1b", "IL-6", "TNF"]
ALL_CLASSES = ACTIVE_CYTOKINES + ["PBS"]

CELL_TYPES = ["id_c0", "id_c1"]

N_GENES = 200
N_CELLS_PER_COMBO = 30  # per (mouse, cytokine, cell_type)

# ---------------------------------------------------------------------------
# Pathway gene names (mouse symbols) embedded in the first 50 gene slots.
# The §24/§25 directional-asymmetry test resolves these from the gene panel.
# ---------------------------------------------------------------------------
PATHWAY_GENE_NAMES: List[str] = [
    # JAK-STAT family markers (STAT1, STAT3, STAT4, STAT5, STAT6 targets)
    "Il2ra",     # IL-2Rα — STAT5 target (γ-chain cytokines)
    "Ifng",      # IFN-γ — STAT1 inducer
    "Gbp2",      # IFN-γ STAT1 target (GBP family)
    "Socs3",     # STAT3 feedback inhibitor
    "Foxp3",     # STAT5 / Treg marker
    "Arg1",      # IL-4/IL-13 STAT6 target (alternative activation)
    "Smad7",     # TGF-β / SMAD pathway negative regulator
    "Stat1",     # JAK-STAT1 signal transducer
    "Stat3",     # JAK-STAT3 signal transducer
    "Stat4",     # IL-12 / STAT4 signal transducer
    "Stat5a",    # γ-chain cytokine STAT5 signal transducer
    # IFN pathway (type I and II)
    "Ifnb1",     # IFN-β (IRF3 direct target)
    "Mx1",       # IFNAR-induced ISG
    "Mx2",       # IFNAR-induced ISG
    "Ifit1",     # IFNAR-induced ISG
    "Ifit2",     # IRF3-direct / IFNAR ISG
    "Ifit3",     # IRF3-direct / IFNAR ISG
    "Rsad2",     # IFNAR-induced ISG (Viperin)
    "Irf1",      # IFN-γ STAT1 → IRF1 axis
    "Irf4",      # STAT6 / IL-4 cascade transcription factor
    "Irf7",      # IFNAR-induced secondary IRF
    "Cxcl10",    # IRF3-direct / STAT1 chemokine
    "Ccl5",      # IRF3-direct chemokine
    # IL-6 / STAT3 markers
    "Il6ra",     # IL-6 receptor α (STAT3 upstream)
    "Il6st",     # gp130 — shared IL-6 family signal transducer
    # IL-12 / STAT4 markers
    "Il12rb1",   # IL-12Rβ1
    "Il12rb2",   # IL-12Rβ2 (STAT4-specific arm)
    # NF-κB / TNF markers
    "Tnfrsf1a",  # TNFR1
    "Nfkbia",    # IκBα — NF-κB canonical target / inhibitor
    "Il1r1",     # IL-1 receptor (NF-κB upstream)
]

assert len(PATHWAY_GENE_NAMES) == 30, "Expected exactly 30 pathway gene names"

# Fill positions 30–199 with generic gene names
_FILLER_GENE_NAMES = [f"Gene_{i:03d}" for i in range(30, N_GENES)]
ALL_GENE_NAMES: List[str] = PATHWAY_GENE_NAMES + _FILLER_GENE_NAMES

assert len(ALL_GENE_NAMES) == N_GENES


# ---------------------------------------------------------------------------
# Demo AnnData builder
# ---------------------------------------------------------------------------

def make_demo_adata_id(seed: int = 42) -> ad.AnnData:
    """
    Build the synthetic input AnnData for the ID adapter's pseudo-tube step.

    Returns an AnnData with obs[mouse_id, cytokine, cell_type] and a small
    log-normalized count matrix (Poisson + log1p). PBS cells are included for
    all mice (pooled PBS design matching the real ID dataset). No further
    PBS relabeling needed — `cytokine` is already the final form.

    Shape: (n_cells, 200) where n_cells = N_MICE * N_CLASSES * N_CELL_TYPES * N_CELLS_PER_COMBO.
    """
    rng = np.random.default_rng(seed)

    rows = []
    Xs = []

    # Simulated per-cytokine signal: each cytokine has a mild up-regulation
    # on a small set of pathway genes to give weak but detectable class signal.
    cyt_signal_genes = {
        "PBS":   slice(0, 0),          # no signal above resting
        "IL-12": slice(7, 12),         # Stat1, Stat3, Stat4 cluster
        "IFN-g": slice(1, 5),          # Ifng, Gbp2, Socs3, Foxp3 cluster
        "IL-1b": slice(27, 30),        # Nfkbia, Il1r1 cluster
        "IL-6":  slice(23, 26),        # Il6ra, Il6st, Il12rb1 cluster
        "TNF":   slice(27, 29),        # Tnfrsf1a, Nfkbia
    }

    for mouse_id in MICE:
        for cyt in ALL_CLASSES:
            for ct in CELL_TYPES:
                n = N_CELLS_PER_COMBO
                # Baseline log-normalized expression (Poisson-derived, log1p)
                counts = rng.poisson(lam=2.0, size=(n, N_GENES)).astype(np.float32)
                X = np.log1p(counts)

                # Add mild cytokine-specific signal on pathway genes
                sig_slice = cyt_signal_genes.get(cyt, slice(0, 0))
                if sig_slice.start != sig_slice.stop:
                    X[:, sig_slice] += rng.normal(
                        0.3, 0.05, size=(n, sig_slice.stop - sig_slice.start)
                    ).astype(np.float32)

                # Add mild cell-type signal on the last 5 genes to help Leiden
                ct_idx = CELL_TYPES.index(ct)
                X[:, -(ct_idx + 1)] += 0.5

                Xs.append(X)
                for _ in range(n):
                    rows.append({
                        "mouse_id": mouse_id,
                        "cytokine": cyt,
                        "cell_type": ct,
                    })

    X_full = np.concatenate(Xs, axis=0)
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i:07d}" for i in range(len(obs))]

    var = pd.DataFrame(index=ALL_GENE_NAMES)
    var.index.name = None

    adata = ad.AnnData(X=X_full, obs=obs, var=var)

    # Shuffle cell order so class / cell_type are not implicit in row order
    perm = rng.permutation(adata.n_obs)
    adata = adata[perm].copy()
    return adata


def write_demo_pseudotubes(
    out_dir: str,
    seed: int = 42,
    n_per_cell_type: int = N_CELLS_PER_COMBO,
    min_cells_threshold: int = 5,
    n_pseudo_tubes: int = 2,
) -> str:
    """
    Build and write pseudo-tubes from the demo AnnData to `out_dir`.

    Returns the path to the written manifest.json.

    Uses `build_pseudo_tubes_id` from the adapter script directly. The adapter
    is imported at call time (not at module import time) so this module can be
    imported from tests without requiring the scripts/ directory on sys.path.
    """
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import build_pseudotubes_immune_dictionary as id_adapter  # noqa: F401

    adata = make_demo_adata_id(seed=seed)
    rng = np.random.default_rng(seed)
    id_adapter.build_pseudo_tubes_id(
        adata,
        base_path=out_dir,
        n_per_cell_type=n_per_cell_type,
        min_cells_threshold=min_cells_threshold,
        n_pseudo_tubes=n_pseudo_tubes,
        rng=rng,
    )
    return str(Path(out_dir) / "manifest.json")


if __name__ == "__main__":
    import sys
    import tempfile

    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="id_demo_")
    path = write_demo_pseudotubes(out_dir)
    print(f"ID demo manifest written to: {path}")
