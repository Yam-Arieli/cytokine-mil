"""Tests for the Phase 1+2 arm runner.

The comparison rests on the four arms differing ONLY in which code produced the weights.
`phase2_train_arm.train_head_cm` turns `train_mil`'s per-epoch dynamics logging down to
once, purely to save time — so that had better not change the weights. If it did, the
`cm` arms would differ from the published anchor's configuration for a reason that has
nothing to do with the code path, and the whole run would be measuring the wrong thing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "cascadir" / "src", REPO_ROOT / "scripts",
           REPO_ROOT / "tests"):
    sys.path.insert(0, str(_p))


def _tiny_binary_dataset(tmp_path: Path):
    import make_demo_data as mdd
    from cytokine_mil.data.dataset import PseudoTubeDataset
    from cytokine_mil.experiment_setup import make_binary_manifest

    mdd.N_PSEUDO_TUBES = 2
    manifest_path = mdd.make_demo_data(str(tmp_path / "demo"))
    manifest = json.loads(Path(manifest_path).read_text())
    bin_manifest, label_enc = make_binary_manifest(manifest, "IL-2", control="PBS")
    mp = tmp_path / "bin.json"
    mp.write_text(json.dumps(bin_manifest))
    import scanpy as sc
    genes = [str(g) for g in sc.read_h5ad(manifest[0]["path"]).var_names]
    return PseudoTubeDataset(str(mp), label_enc, gene_names=genes, preload=True), label_enc, genes


def _train(ds, label_enc, genes, log_every, seed=42, epochs=4):
    from cytokine_mil.experiment_setup import build_encoder, build_mil_model
    from cytokine_mil.training.train_mil import train_mil

    torch.manual_seed(0)
    enc = build_encoder(n_input_genes=len(genes), n_cell_types=5, embed_dim=16)
    model = build_mil_model(enc, embed_dim=16, attention_hidden_dim=8,
                            n_classes=label_enc.n_classes(), encoder_frozen=True)
    train_mil(model, ds, n_epochs=epochs, lr=1e-3, momentum=0.9,
              log_every_n_epochs=log_every, device=torch.device("cpu"),
              seed=seed, verbose=False)
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def test_log_frequency_does_not_change_the_trained_weights(tmp_path):
    ds, le, genes = _tiny_binary_dataset(tmp_path)
    a = _train(ds, le, genes, log_every=1)
    b = _train(ds, le, genes, log_every=99)
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_allclose(
            a[k], b[k], rtol=0, atol=0,
            err_msg=f"{k} differs between log_every=1 and log_every=99 — dynamics logging "
                    "is not read-only, so turning it down changes the experiment.",
        )


def test_the_weight_comparison_can_fail(tmp_path):
    """A different seed must give different weights, or the test above proves nothing."""
    ds, le, genes = _tiny_binary_dataset(tmp_path)
    a = _train(ds, le, genes, log_every=1, seed=42)
    c = _train(ds, le, genes, log_every=1, seed=7)
    assert any(not np.array_equal(a[k], c[k]) for k in a), (
        "two different seeds produced byte-identical weights — the comparison is vacuous."
    )
