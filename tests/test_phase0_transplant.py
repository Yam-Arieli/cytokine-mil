"""Unit tests for Phase 0 of the controlled code-path comparison.

The load-bearing claim in `scripts/phase0_ig_transplant.py` is that a `cytokine_mil`
binary AB-MIL `state_dict` can be pushed into a `cascadir.AbMil` and compute the SAME
function. If that were quietly false — a partial load leaving random weights in place —
every IG number the script produces would be fabricated. These tests pin it down on
small random models, so a regression in either package's module naming fails here rather
than after a cluster job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase0_ig_transplant import (  # noqa: E402
    _sig_rows,
    assert_forward_equivalent,
    build_cascadir_model,
)
from run_binary_ig_probe import _build_binary_mil  # noqa: E402

G, N_CT, EMBED, H0, H1, ATT = 60, 4, 16, 32, 24, 8


def _cm_model():
    torch.manual_seed(0)
    return _build_binary_mil(
        n_input_genes=G, n_cell_types=N_CT, embed_dim=EMBED,
        hidden_dims=(H0, H1), attention_hidden_dim=ATT, device=torch.device("cpu"),
    )


def test_key_sets_match_exactly_between_paths():
    cm = _cm_model()
    cd = build_cascadir_model(cm.state_dict(), G, "cpu")
    assert set(cm.state_dict()) == set(cd.state_dict())


def test_transplanted_model_computes_the_same_function():
    cm = _cm_model().eval()
    cd = build_cascadir_model(cm.state_dict(), G, "cpu")
    delta = assert_forward_equivalent(cm, cd, G, "cpu")
    assert delta < 1e-5
    # ...and on a second, independent input, so we are not testing one lucky draw.
    X = torch.rand(23, G, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        np.testing.assert_allclose(
            cm(X)[0].numpy(), cd(X)[0].numpy(), rtol=0, atol=1e-5
        )


def test_transplant_is_not_vacuous_a_different_model_disagrees():
    """The equivalence check must be able to FAIL — otherwise it proves nothing."""
    cm = _cm_model().eval()
    torch.manual_seed(999)
    other = _build_binary_mil(
        n_input_genes=G, n_cell_types=N_CT, embed_dim=EMBED, hidden_dims=(H0, H1),
        attention_hidden_dim=ATT, device=torch.device("cpu"),
    ).eval()
    with pytest.raises(AssertionError, match="disagree"):
        assert_forward_equivalent(cm, other, G, "cpu")


@pytest.mark.parametrize("key", ["attention.V.bias", "encoder.res1.fc1.weight"])
def test_partial_state_dict_is_refused_not_silently_loaded(key):
    """A dropped non-shape key must be caught by the 1:1 key-set check."""
    cm = _cm_model()
    broken = dict(cm.state_dict())
    broken.pop(key)
    with pytest.raises(AssertionError, match="1:1"):
        build_cascadir_model(broken, G, "cpu")


@pytest.mark.parametrize("key", ["attention.V.weight", "encoder.down2.fc1.weight",
                                 "classifier.classifier.weight"])
def test_missing_shape_key_fails_readably_not_with_a_raw_keyerror(key):
    """Dropping a shape-DEFINING key must still refuse, and say why.

    These are read during HP inference, before any key-set comparison is possible, so
    without an explicit guard the failure is an opaque KeyError from inside the probe.
    """
    cm = _cm_model()
    broken = dict(cm.state_dict())
    broken.pop(key)
    with pytest.raises(AssertionError, match="shape-defining"):
        build_cascadir_model(broken, G, "cpu")


def test_unexpected_key_is_refused():
    cm = _cm_model()
    broken = dict(cm.state_dict())
    broken["attention.bogus"] = torch.zeros(3)
    with pytest.raises(AssertionError, match="1:1"):
        build_cascadir_model(broken, G, "cpu")


def test_sig_rows_are_rank_ordered_and_truncated():
    from cascadir.types import Signature

    sig = Signature(condition="X", genes=tuple("abcde"),
                    ig_scores=(5.0, 4.0, 3.0, 2.0, 1.0), top_n=5)
    rows = _sig_rows("X", sig, keep=3)
    assert [r["gene"] for r in rows] == ["a", "b", "c"]
    assert [r["rank_ig"] for r in rows] == [0, 1, 2]
    assert all(r["cytokine"] == "X" for r in rows)


def test_derive_signature_runs_on_a_transplanted_model():
    """End-to-end smoke: transplanted model -> cascadir IG -> a Signature."""
    from cascadir.signatures import derive_signature
    from cascadir.types import PseudoTube, PseudoTubeSet

    cd = build_cascadir_model(_cm_model().state_dict(), G, "cpu")
    rng = np.random.default_rng(0)

    def tube(cond, idx):
        X = rng.random((12, G), dtype=np.float32)
        return PseudoTube(X=X, condition=cond, donor="D1",
                          cell_types=tuple("t" for _ in range(12)),
                          cell_types_included=("t",), tube_idx=idx)

    ts = PseudoTubeSet(tubes=[tube("A", 0), tube("A", 1), tube("PBS", 0)],
                       gene_names=tuple(f"g{i}" for i in range(G)),
                       control_label="PBS")
    sig = derive_signature(cd, ts, "A", top_n=10, n_steps=4, device="cpu")
    assert len(sig.genes) == 10
    assert len(set(sig.genes)) == 10
    assert list(sig.ig_scores) == sorted(sig.ig_scores, reverse=True)
