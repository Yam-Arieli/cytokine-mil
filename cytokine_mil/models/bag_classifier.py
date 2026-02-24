"""
BagClassifier: linear classifier on the attention-aggregated tube representation.
"""

import torch
import torch.nn as nn


class BagClassifier(nn.Module):
    """
    Linear classifier on the aggregated pseudo-tube representation.

    Input:  z_tube in R^embed_dim  (attention-weighted sum of cell embeddings)
    Output: y_hat in R^K           (K = number of cytokine classes)
    """

    def __init__(self, embed_dim: int = 128, n_classes: int = 91) -> None:
        super().__init__()
        self.classifier = nn.Linear(embed_dim, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, z_tube: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_tube: (embed_dim,) aggregated tube embedding.
        Returns:
            y_hat: (K,) class logits.
        """
        return self.classifier(z_tube)
