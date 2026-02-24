"""
CytokineABMIL: full AB-MIL pipeline combining all three components.

Forward pass returns (y_hat, a, H) so dynamics metrics can be computed
without a second pass.
"""

from typing import Tuple

import torch
import torch.nn as nn

from cytokine_mil.models.attention import AttentionModule
from cytokine_mil.models.bag_classifier import BagClassifier
from cytokine_mil.models.instance_encoder import InstanceEncoder


class CytokineABMIL(nn.Module):
    """
    Full AB-MIL pipeline: InstanceEncoder -> AttentionModule -> BagClassifier.

    Accepts a pre-trained InstanceEncoder. encoder_frozen controls whether
    encoder weights are updated during MIL training.

    Input:  X in R^(N x G)    (pseudo-tube: N cells, G genes)
    Output: y_hat in R^K      (class logits)
            a in R^N          (attention weights, for dynamics tracking)
            H in R^(N x 128)  (cell embeddings, for instance-level confidence)
    """

    def __init__(
        self,
        encoder: InstanceEncoder,
        attention: AttentionModule,
        classifier: BagClassifier,
        encoder_frozen: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.classifier = classifier
        if encoder_frozen:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Enable encoder weight updates (Stage 3 fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X: (N, G) pseudo-tube expression matrix (one tube at a time).
        Returns:
            y_hat: (K,) class logits.
            a:     (N,) attention weights.
            H:     (N, embed_dim) cell embeddings.
        """
        H = self.encoder(X)                          # (N, embed_dim)
        a = self.attention(H)                         # (N,)
        z_tube = (a.unsqueeze(1) * H).sum(dim=0)     # (embed_dim,)
        y_hat = self.classifier(z_tube)               # (K,)
        return y_hat, a, H
