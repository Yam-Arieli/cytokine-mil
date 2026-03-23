"""
TwoLayerAttentionModule: two independent attention layers (SA and CA).

SA layer: standard Ilse et al. (2018) attention.
    a_SA_i = softmax( w_sa^T * tanh(V_sa * h_i) )

CA layer: conditioned on the SA aggregate z_SA.
    a_CA_i = softmax( w_ca^T * tanh(V_ca * h_i + U_ca * z_SA) )

No shared weights between SA and CA. Zero dropout — stability of attention
weights is required for dynamics tracking across epochs.
"""

from typing import Tuple

import torch
import torch.nn as nn


class TwoLayerAttentionModule(nn.Module):
    """
    Two independent attention layers over cell embeddings.

    SA layer produces a first-pass attention aggregate z_SA.
    CA layer conditions on z_SA to produce a second attention distribution.

    No shared weights between SA and CA layers.
    No dropout — stability of attention weights is required for dynamics tracking.

    Input:  H in R^(N x embed_dim)
    Output: (a_SA, z_SA, a_CA, z_CA)
        a_SA: (N,)         SA attention weights summing to 1.
        z_SA: (embed_dim,) SA attention-weighted aggregate.
        a_CA: (N,)         CA attention weights summing to 1.
        z_CA: (embed_dim,) CA attention-weighted aggregate.
    """

    def __init__(self, embed_dim: int = 128, attention_hidden_dim: int = 64) -> None:
        super().__init__()
        # SA parameters — independent set
        self.V_sa = nn.Linear(embed_dim, attention_hidden_dim)
        self.w_sa = nn.Linear(attention_hidden_dim, 1, bias=False)
        # CA parameters — independent set (no shared weights with SA)
        self.V_ca = nn.Linear(embed_dim, attention_hidden_dim)
        self.w_ca = nn.Linear(attention_hidden_dim, 1, bias=False)
        self.U_ca = nn.Linear(embed_dim, attention_hidden_dim, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.V_sa.weight)
        nn.init.zeros_(self.V_sa.bias)
        nn.init.xavier_uniform_(self.w_sa.weight)
        nn.init.xavier_uniform_(self.V_ca.weight)
        nn.init.zeros_(self.V_ca.bias)
        nn.init.xavier_uniform_(self.w_ca.weight)
        nn.init.xavier_uniform_(self.U_ca.weight)

    def _compute_sa(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SA attention weights and aggregate.

        Args:
            H: (N, embed_dim) cell embeddings.
        Returns:
            a_SA: (N,) SA attention weights summing to 1.
            z_SA: (embed_dim,) SA attention-weighted aggregate.
        """
        # H: (N, embed_dim)
        scores = self.w_sa(torch.tanh(self.V_sa(H)))   # (N, 1)
        a_SA = torch.softmax(scores, dim=0).squeeze(1)  # (N,)
        z_SA = (a_SA.unsqueeze(1) * H).sum(dim=0)       # (embed_dim,)
        return a_SA, z_SA

    def _compute_ca(
        self, H: torch.Tensor, z_SA: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CA attention weights and aggregate, conditioned on z_SA.

        Args:
            H:   (N, embed_dim) cell embeddings.
            z_SA: (embed_dim,) SA aggregate used for conditioning.
        Returns:
            a_CA: (N,) CA attention weights summing to 1.
            z_CA: (embed_dim,) CA attention-weighted aggregate.
        """
        # V_ca * h_i + U_ca * z_SA  (broadcast U_ca * z_SA over N cells)
        # V_sa(H): (N, attention_hidden_dim); U_ca(z_SA): (attention_hidden_dim,)
        scores = self.w_ca(
            torch.tanh(self.V_ca(H) + self.U_ca(z_SA).unsqueeze(0))
        )                                                # (N, 1)
        a_CA = torch.softmax(scores, dim=0).squeeze(1)  # (N,)
        z_CA = (a_CA.unsqueeze(1) * H).sum(dim=0)       # (embed_dim,)
        return a_CA, z_CA

    def forward(
        self, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            H: (N, embed_dim) cell embeddings.
        Returns:
            a_SA: (N,)         SA attention weights summing to 1.
            z_SA: (embed_dim,) SA attention-weighted aggregate.
            a_CA: (N,)         CA attention weights summing to 1.
            z_CA: (embed_dim,) CA attention-weighted aggregate.
        """
        a_SA, z_SA = self._compute_sa(H)
        a_CA, z_CA = self._compute_ca(H, z_SA)
        return a_SA, z_SA, a_CA, z_CA
