"""Registry of every saved Oesinghaus-90 Stage-1 encoder with a MEASURED outcome.

This is data, not logic. Each entry pairs a frozen encoder checkpoint with the signature
table that encoder produced, so `probe_encoder_gene_geometry.py` can ask whether the
encoder's gene-space geometry predicts how specific its signatures turned out.

Two things every consumer must respect:

  * `panel` — mean between-cytokine Jaccard is only comparable WITHIN a panel. The eight
    sweep arms share one seeded-random 24 (`sweep24`) and are the confound-free correlation
    set; `published24` and `recurrent45` are different panels and are reported separately
    (CLAUDE.md §38.3 handled exactly this confound by panel-matching).
  * `embed_dim` — twelve fits are 512-wide, the two §37 fits are 1024-wide. Raw
    participation ratio is NOT comparable across those; use PR/d.

`published_runA` is deliberately absent: three candidate run directories exist under
`results/oesinghaus_binary/` and none is unambiguously the one merged into
`binary_ig_all24`, so it has no exact encoder-to-outcome pairing. Run B and the §31 seeds do.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SHARD_DIR = REPO_ROOT / "results" / "oes_full90" / "tubes"

# Run B's 16 conditions (scripts/train_oesinghaus_binary_missing16.py:87-104). The published
# parquet merges run A and run B; only these 16 were produced under run B's encoder, and
# §38.5 measured this subset at meanJ 0.079.
RUN_B_CONDITIONS = [
    "IFN-gamma", "IFN-omega", "IFN-lambda1", "IL-15", "IL-17A", "IL-36-alpha",
    "IL-9", "IL-13", "IL-27", "IL-16", "CD30L", "Decorin", "VEGF", "GM-CSF",
    "TL1A", "IL-35",
]

_PUBLISHED_IG = "results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet"


def _fit(key, encoder, code_path, panel, *, signatures=None, conditions=None,
         embed_dim=512, diagnostic_only=False, note=""):
    return {
        "key": key,
        "encoder": encoder,
        "code_path": code_path,
        "panel": panel,
        "signatures": signatures,
        "conditions": conditions,
        "embed_dim": embed_dim,
        "diagnostic_only": diagnostic_only,
        "note": note,
    }


FITS = [
    # ---- the healthy side: the cytokine_mil training path ------------------
    _fit("published_runB",
         "results/oesinghaus_binary_missing16/run_20260530_191127_pid213865/encoder_shared_stage1.pt",
         "cytokine_mil", "published24",
         signatures=_PUBLISHED_IG, conditions=RUN_B_CONDITIONS,
         note="the published 88% anchor's second run; Stage-1 saw 17 conditions incl. D2/D3"),
    _fit("recurrent_ig_s42", "results/recurrent_ig/seed_42/encoder_shared_stage1.pt",
         "cytokine_mil", "recurrent45",
         signatures="results/recurrent_ig/seed_42/final_signatures.parquet",
         note="§31, 45 conditions, single shared Stage-1 encoder, wide config"),
    _fit("recurrent_ig_s123", "results/recurrent_ig/seed_123/encoder_shared_stage1.pt",
         "cytokine_mil", "recurrent45",
         signatures="results/recurrent_ig/seed_123/final_signatures.parquet",
         note="§31 seed 123"),
    _fit("recurrent_ig_s7", "results/recurrent_ig/seed_7/encoder_shared_stage1.pt",
         "cytokine_mil", "recurrent45",
         signatures="results/recurrent_ig/seed_7/final_signatures.parquet",
         note="§31 seed 7"),

    # ---- the collapsed side: the cascadir training path --------------------
    _fit("oes_full90", "results/oes_full90/encoder.pt", "cascadir", "published24",
         signatures="results/oes_full90/signatures_all90.parquet",
         note="§36; 90 conditions, 512-wide, 20 Stage-1 epochs"),
    _fit("oes90_pure", "results/oes90_pure/encoder.pt", "cascadir", "published24",
         signatures="results/oes90_pure/signatures_main.parquet", embed_dim=1024,
         note="§37 PURE; 1024-wide, best-val encoder at epoch 4, top_n=100"),
    _fit("oes90_pure_ep50", "results/oes90_pure_ep50/encoder.pt", "cascadir", "published24",
         signatures="results/oes90_pure_ep50/signatures_main.parquet", embed_dim=1024,
         note="§37 re-run with Stage-2 stopped at epoch 50"),

    # ---- the eight sweep arms: ONE shared panel, everything else pinned ----
    _fit("encsweep_pbs_only", "results/encsweep/pbs_only/encoder.pt", "cascadir", "sweep24",
         signatures="results/encsweep/pbs_only/signatures.parquet",
         note="§38.3 breadth arm; Stage-1 saw PBS only"),
    _fit("encsweep_rand18", "results/encsweep/rand18/encoder.pt", "cascadir", "sweep24",
         signatures="results/encsweep/rand18/signatures.parquet",
         note="§38.3 breadth arm; 18 seeded-random stimuli"),
    _fit("encsweep_rand45", "results/encsweep/rand45/encoder.pt", "cascadir", "sweep24",
         signatures="results/encsweep/rand45/signatures.parquet",
         note="§38.3 breadth arm; 45 seeded-random stimuli"),
    _fit("encsweep_all90", "results/encsweep/all90/encoder.pt", "cascadir", "sweep24",
         signatures="results/encsweep/all90/signatures.parquet",
         note="§38.3 breadth arm; all 90 stimuli"),
    _fit("s1sweep_pub_replica", "results/s1sweep/pub_replica/encoder.pt", "cascadir", "sweep24",
         signatures="results/s1sweep/pub_replica/signatures.parquet", diagnostic_only=True,
         note="§38.4; one tube per condition, D2/D3 INCLUDED — violates §16 by design, "
              "diagnostic only, must never seed a production fit"),
    _fit("s1sweep_pub_replica_clean", "results/s1sweep/pub_replica_clean/encoder.pt",
         "cascadir", "sweep24",
         signatures="results/s1sweep/pub_replica_clean/signatures.parquet",
         note="§38.4; one tube per condition, D2/D3 excluded"),
    _fit("s1sweep_vol_small", "results/s1sweep/vol_small/encoder.pt", "cascadir", "sweep24",
         signatures="results/s1sweep/vol_small/signatures.parquet",
         note="§38.4; donor-balanced, ~7.5K Stage-1 cells"),
    _fit("s1sweep_vol_large", "results/s1sweep/vol_large/encoder.pt", "cascadir", "sweep24",
         signatures="results/s1sweep/vol_large/signatures.parquet",
         note="§38.4; donor-balanced, 36K Stage-1 cells (reproduces encsweep_all90 exactly)"),
]

BY_KEY = {f["key"]: f for f in FITS}
