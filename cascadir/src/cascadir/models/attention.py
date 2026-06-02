"""AttentionModule: learnable attention aggregation over cell embeddings.

Implements the Ilse et al. (2018) gated-free attention mechanism:
    ``a_i = softmax( w^T * tanh(V * h_i) )``

No dropout is applied — stable attention weights are required so that the
discovered per-condition signatures are reproducible. Pure tensor module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """Learnable attention aggregation over cell embeddings.

    Input:  ``H`` in R^(N x embed_dim)
    Output: ``a`` in R^N  (non-negative, sums to 1)
    """

    def __init__(self, embed_dim: int = 128, attention_hidden_dim: int = 64) -> None:
        super().__init__()
        self.V = nn.Linear(embed_dim, attention_hidden_dim)
        self.w = nn.Linear(attention_hidden_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.zeros_(self.V.bias)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Args: H ``(N, embed_dim)``. Returns: a ``(N,)`` summing to 1."""
        scores = self.w(torch.tanh(self.V(H)))  # (N, 1)
        a = torch.softmax(scores, dim=0)         # (N, 1)
        return a.squeeze(1)                      # (N,)
