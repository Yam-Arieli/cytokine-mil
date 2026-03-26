"""
Stage3CAModel: wraps a trained CytokineABMIL (v1) and adds a trainable CA layer.

Encoder, SA attention, and classifier are completely frozen. Only the CA layer
is trainable, initialized with small weights (std=0.01) so it starts as near-zero
noise. Output = classifier(z_SA + z_CA), reusing the existing BagClassifier
(input dim = embed_dim = 128).

Hypothesis: if SA + classifier already solved classification, CA will remain
noise (weights stay near zero, entropy stays high). CA only learns if it
provides genuine additional signal.
"""

from typing import Tuple

import torch
import torch.nn as nn

from cytokine_mil.models.cytokine_abmil import CytokineABMIL


class Stage3CAModel(nn.Module):
    """
    Wraps a fully trained CytokineABMIL (v1) and adds a single trainable CA layer.

    Encoder, SA attention, and classifier are frozen. Only CA parameters
    (V_ca, w_ca, U_ca) are trainable. CA is initialized with near-zero weights
    (std=0.01) so any learned signal must come from the classification objective.

    Forward pass returns (y_hat, a_SA, a_CA, H) for dynamics compatibility.

    Input:  X in R^(N x G)
    Output: y_hat in R^K, a_SA in R^N, a_CA in R^N, H in R^(N x embed_dim)
    """

    def __init__(
        self,
        frozen_model: CytokineABMIL,
        embed_dim: int = 128,
        attention_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        # Store frozen submodules
        self.encoder = frozen_model.encoder
        self.sa_attention = frozen_model.attention
        self.classifier = frozen_model.classifier

        # Freeze all parameters of the frozen submodules
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.sa_attention.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False

        # Trainable CA parameters (same structure as TwoLayerAttentionModule)
        self.V_ca = nn.Linear(embed_dim, attention_hidden_dim)
        self.w_ca = nn.Linear(attention_hidden_dim, 1, bias=False)
        self.U_ca = nn.Linear(embed_dim, attention_hidden_dim, bias=False)

        self._init_ca_weights()

    def _init_ca_weights(self) -> None:
        """Initialize CA parameters with small normal weights (near-zero noise)."""
        nn.init.normal_(self.V_ca.weight, std=0.01)
        nn.init.zeros_(self.V_ca.bias)
        nn.init.normal_(self.w_ca.weight, std=0.01)
        nn.init.normal_(self.U_ca.weight, std=0.01)

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X: (N, G) pseudo-tube expression matrix (one tube at a time).
        Returns:
            y_hat: (K,) class logits.
            a_SA:  (N,) SA attention weights (frozen, constant across epochs).
            a_CA:  (N,) CA attention weights (trainable).
            H:     (N, embed_dim) cell embeddings (frozen).
        """
        # Frozen forward: encoder -> SA
        H = self.encoder(X)                               # (N, embed_dim)
        a_SA = self.sa_attention(H)                       # (N,)
        z_SA = (a_SA.unsqueeze(1) * H).sum(dim=0)        # (embed_dim,)

        # Trainable CA computation (mirrors TwoLayerAttentionModule._compute_ca)
        # V_ca(H): (N, attention_hidden_dim); U_ca(z_SA): (attention_hidden_dim,)
        scores = self.w_ca(
            torch.tanh(self.V_ca(H) + self.U_ca(z_SA).unsqueeze(0))
        )                                                  # (N, 1)
        a_CA = torch.softmax(scores, dim=0).squeeze(1)   # (N,)
        z_CA = (a_CA.unsqueeze(1) * H).sum(dim=0)        # (embed_dim,)

        # Frozen classifier on sum of SA and CA aggregates (input dim = embed_dim)
        y_hat = self.classifier(z_SA + z_CA)              # (K,)

        return y_hat, a_SA, a_CA, H

    def ca_weight_norm(self) -> float:
        """L2 norm of all CA parameters concatenated."""
        norms = [
            p.data.norm(2)
            for p in [
                self.V_ca.weight,
                self.V_ca.bias,
                self.w_ca.weight,
                self.U_ca.weight,
            ]
        ]
        return float(torch.stack(norms).norm(2))
