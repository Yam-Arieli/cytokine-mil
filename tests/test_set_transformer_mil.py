"""
Unit tests for the §34 self-attention MIL model (CytokineSelfAttnMIL).

Verifies the CytokineABMIL-compatible forward contract (so train_mil runs
unchanged), that cells actually interact (N×N attention, not identity), and that
the extractor helper methods (pool_from_H / interaction_from_H / readout_from_H)
agree with a full forward_with_interaction.
"""

import torch

from cytokine_mil.experiment_setup import build_encoder, build_selfattn_model
from cytokine_mil.models.cytokine_abmil_v2 import CytokineABMIL_V2

G, N, EMBED, NCT, K = 20, 12, 16, 3, 5


def _model():
    enc = build_encoder(n_input_genes=G, n_cell_types=NCT, embed_dim=EMBED)
    return build_selfattn_model(enc, embed_dim=EMBED, attention_hidden_dim=8,
                                n_classes=K, encoder_frozen=True, sab_heads=4, sab_layers=1)


def test_forward_contract_matches_abmil():
    m = _model()
    X = torch.randn(N, G)
    y_hat, a, H = m(X)
    assert y_hat.shape == (K,)
    assert a.shape == (N,)
    assert H.shape == (N, EMBED)
    assert torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-5)


def test_not_v2_so_train_mil_uses_standard_branch():
    assert isinstance(_model(), CytokineABMIL_V2) is False


def test_interaction_is_NxN_rows_sum_to_one():
    m = _model()
    X = torch.randn(N, G)
    _, _, _, A = m.forward_with_interaction(X)
    assert A.shape == (N, N)
    row_sums = A.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-4)


def test_cells_actually_interact():
    """A must not be the identity — every cell attends to others (off-diagonal>0)."""
    m = _model()
    X = torch.randn(N, G)
    _, _, _, A = m.forward_with_interaction(X)
    off = A.sum() - torch.diagonal(A).sum()
    assert float(off) > 0.0
    assert not torch.allclose(A, torch.eye(N), atol=1e-3)


def test_forward_returns_original_frozen_H():
    """3rd output is the ORIGINAL encoder embedding (not the interacted H')."""
    m = _model()
    X = torch.randn(N, G)
    _, _, H = m(X)
    assert torch.allclose(H, m.encoder(X), atol=1e-6)


def test_extractor_helpers_match_full_forward():
    m = _model().eval()
    X = torch.randn(N, G)
    with torch.no_grad():
        y, a_full, H, A_full = m.forward_with_interaction(X)
        a_r, A_r = m.readout_from_H(H)
        a_p, Hp = m.pool_from_H(H)
        A_i = m.interaction_from_H(H)
    assert torch.allclose(a_full, a_r, atol=1e-5)
    assert torch.allclose(a_full, a_p, atol=1e-5)
    assert torch.allclose(A_full, A_r, atol=1e-5)
    assert torch.allclose(A_full, A_i, atol=1e-5)
    assert Hp.shape == (N, EMBED)


def test_encoder_freeze_unfreeze():
    m = _model()
    assert m.encoder_frozen is True
    assert all(not p.requires_grad for p in m.encoder.parameters())
    m.unfreeze_encoder()
    assert m.encoder_frozen is False
    assert all(p.requires_grad for p in m.encoder.parameters())


def test_sab_and_pool_are_trainable_when_encoder_frozen():
    m = _model()
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any(n.startswith("sab_layers") for n in trainable)
    assert any(n.startswith("attention") for n in trainable)
    assert any(n.startswith("classifier") for n in trainable)
    assert not any(n.startswith("encoder.") for n in trainable)
