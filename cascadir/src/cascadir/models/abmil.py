"""AbMil: the full attention-based MIL pipeline (encoder -> attention -> classifier).

Forward returns ``(y_hat, a, H)`` in one pass so that Integrated-Gradients
attribution (on ``y_hat``) and attention inspection need no second forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cascadir.models.attention import AttentionModule
from cascadir.models.bag_classifier import BagClassifier
from cascadir.models.instance_encoder import InstanceEncoder


class AbMil(nn.Module):
    """Full AB-MIL pipeline: InstanceEncoder -> AttentionModule -> BagClassifier.

    Accepts a pre-trained :class:`InstanceEncoder`. ``encoder_frozen`` controls
    whether encoder weights update during MIL training (frozen by default so the
    discovered signatures reflect the bag-level head, not a re-fit encoder).

    Input:  ``X`` in R^(N x G)   (one pseudo-tube: N cells, G genes)
    Output: ``y_hat`` in R^K, ``a`` in R^N (attention), ``H`` in R^(N x embed_dim)
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
        self.encoder_frozen: bool = encoder_frozen
        if encoder_frozen:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        """Enable encoder weight updates (optional fine-tuning)."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder_frozen = False

    def forward_from_H(
        self, H: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bag head only: attention -> weighted pool -> classifier, given cell embeddings.

        Args: H ``(N, embed_dim)`` (encoder output). Returns: (y_hat ``(K,)``, a ``(N,)``, H).

        Used to train the attention/classifier on **pre-encoded** cells when the encoder
        is frozen (its output ``H`` is then constant across epochs). Produces the exact
        same computation as ``forward`` from the point ``H`` is available, so training on
        cached ``H`` is bit-identical to re-running the encoder every step.
        """
        a = self.attention(H)                     # (N,)
        z_tube = (a.unsqueeze(1) * H).sum(dim=0)  # (embed_dim,)
        y_hat = self.classifier(z_tube)           # (K,)
        return y_hat, a, H

    def forward(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: X ``(N, G)``. Returns: (y_hat ``(K,)``, a ``(N,)``, H ``(N, embed_dim)``)."""
        H = self.encoder(X)                       # (N, embed_dim)
        return self.forward_from_H(H)
