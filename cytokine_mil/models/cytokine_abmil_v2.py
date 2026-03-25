"""
CytokineABMIL_V2: full AB-MIL pipeline with two-layer attention.

Forward pass returns (y_hat, a_SA, a_CA, H). The bag representation is the
concatenation of z_SA and z_CA (256-dim), passed to a linear BagClassifier.
"""

from typing import Tuple

import torch
import torch.nn as nn

from cytokine_mil.models.instance_encoder import InstanceEncoder
from cytokine_mil.models.two_layer_attention import TwoLayerAttentionModule


class CytokineABMIL_V2(nn.Module):
    """
    Full AB-MIL pipeline with two-layer attention:
    InstanceEncoder -> TwoLayerAttentionModule -> BagClassifier(256).

    Accepts a pre-trained InstanceEncoder. encoder_frozen controls whether
    encoder weights are updated during MIL training.

    The bag representation is the concatenation of z_SA and z_CA
    (embed_dim + embed_dim = 256), passed to a linear classifier.

    Input:  X in R^(N x G)
    Output: y_hat in R^K      (class logits)
            a_SA  in R^N      (SA attention weights, sums to 1)
            a_CA  in R^N      (CA attention weights, sums to 1)
            H     in R^(N x embed_dim)  (cell embeddings)
    """

    def __init__(
        self,
        encoder: InstanceEncoder,
        attention: TwoLayerAttentionModule,
        n_classes: int = 91,
        embed_dim: int = 128,
        encoder_frozen: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        # Input dim = 2 * embed_dim because z_SA and z_CA are concatenated.
        self.classifier = nn.Linear(2 * embed_dim, n_classes)
        # Auxiliary heads for SA-only and CA-only classification (used for aux loss during training).
        self.sa_head = nn.Linear(embed_dim, n_classes)
        self.ca_head = nn.Linear(embed_dim, n_classes)
        self._init_classifier()
        self._init_aux_heads()
        if encoder_frozen:
            self._freeze_encoder()

    def _init_classifier(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _init_aux_heads(self) -> None:
        for head in (self.sa_head, self.ca_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def _freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_encoder(self) -> None:
        """Disable encoder weight updates."""
        self._freeze_encoder()

    def unfreeze_encoder(self) -> None:
        """Enable encoder weight updates (Stage 3 fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X: (N, G) pseudo-tube expression matrix (one tube at a time).
        Returns:
            y_hat: (K,)         class logits (from combined z_SA + z_CA).
            a_SA:  (N,)         SA attention weights.
            a_CA:  (N,)         CA attention weights.
            H:     (N, embed_dim) cell embeddings.
        """
        H = self.encoder(X)                              # (N, embed_dim)
        a_SA, z_SA, a_CA, z_CA = self.attention(H)      # (N,), (D,), (N,), (D,)
        z_tube = torch.cat([z_SA, z_CA], dim=0)         # (2*embed_dim,)
        y_hat = self.classifier(z_tube)                  # (K,)
        return y_hat, a_SA, a_CA, H

    def forward_with_aux(
        self, X: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """
        Extended forward pass returning SA and CA auxiliary logits for regularized training.

        Args:
            X: (N, G) pseudo-tube expression matrix (one tube at a time).
        Returns:
            y_hat:    (K,) class logits from combined representation.
            a_SA:     (N,) SA attention weights.
            a_CA:     (N,) CA attention weights.
            H:        (N, embed_dim) cell embeddings.
            y_hat_sa: (K,) class logits from SA representation alone (auxiliary).
            y_hat_ca: (K,) class logits from CA representation alone (auxiliary).
        """
        H = self.encoder(X)
        a_SA, z_SA, a_CA, z_CA = self.attention(H)
        z_tube = torch.cat([z_SA, z_CA], dim=0)
        y_hat = self.classifier(z_tube)
        y_hat_sa = self.sa_head(z_SA)
        y_hat_ca = self.ca_head(z_CA)
        return y_hat, a_SA, a_CA, H, y_hat_sa, y_hat_ca
